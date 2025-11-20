import os
from datetime import datetime

from dotenv import load_dotenv
from pymongo import MongoClient
from bson.objectid import ObjectId

from services.processor.finbert_model import analyze_headline

# Load .env from project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB", "financial_news")


def get_collection():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    return db["news_events"]


def fetch_unscored_batch(coll, batch_size=50):
    """
    Fetch a batch of documents where sentiment_score is None.
    """
    cursor = coll.find(
        {"sentiment_score": None},
        {"headline": 1},  # only need headline + _id
    ).sort("published_at", 1).limit(batch_size)

    return list(cursor)


def process_batch(batch_size=50):
    coll = get_collection()
    docs = fetch_unscored_batch(coll, batch_size=batch_size)

    if not docs:
        print("No unscored documents found.")
        return 0

    print(f"Processing batch of {len(docs)} documents...")

    updated_count = 0

    for doc in docs:
        _id = doc["_id"]
        headline = doc.get("headline", "")

        label, score = analyze_headline(headline)

        coll.update_one(
            {"_id": ObjectId(_id)},
            {
                "$set": {
                    "sentiment_label": label,
                    "sentiment_score": score,
                    "sentiment_updated_at": datetime.utcnow(),
                }
            },
        )
        updated_count += 1

    print(f"Updated {updated_count} documents in this batch.")
    return updated_count


def run_worker(max_loops=100, batch_size=50):
    """
    Run in a loop until no more unscored docs or max_loops reached.
    """
    total_updated = 0

    for i in range(max_loops):
        print(f"\n--- Loop {i + 1} ---")
        updated = process_batch(batch_size=batch_size)
        total_updated += updated

        if updated == 0:
            break

    print(f"\nSentiment processing complete. Total documents updated: {total_updated}")


if __name__ == "__main__":
    run_worker()
