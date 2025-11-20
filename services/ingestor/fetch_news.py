import os
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING

# Load .env from project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB", "financial_news")

# Small dev ticker set – we can expand later
TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA"]


def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    coll = db["news_events"]

    # Ensure unique documents
    coll.create_index(
        [("ticker", ASCENDING), ("headline", ASCENDING), ("published_at", ASCENDING)],
        unique=True,
        name="uniq_ticker_headline_published",
    )

    return coll


def fetch_news_for_ticker(ticker: str, days_back: int = 30):
    if not FINNHUB_API_KEY:
        raise RuntimeError("FINNHUB_API_KEY is not set in .env")

    url = "https://finnhub.io/api/v1/company-news"
    today = datetime.utcnow().date()
    start = today - timedelta(days=days_back)

    params = {
        "symbol": ticker,
        "from": start.isoformat(),
        "to": today.isoformat(),
        "token": FINNHUB_API_KEY,
    }

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def insert_news_batch(coll, ticker: str, items: list):
    inserted = 0

    for item in items:
        headline = item.get("headline") or ""
        if not headline:
            continue

        # Convert timestamp → timezone-aware datetime in UTC
        published_at = datetime.fromtimestamp(item["datetime"], tz=timezone.utc)
        url = item.get("url")
        source = item.get("source", "unknown")

        doc = {
            "source": source,
            "ticker": ticker,
            "headline": headline,
            "summary": None,
            "sentiment_label": None,
            "sentiment_score": None,
            "published_at": published_at,
            "ingested_at": datetime.now(timezone.utc),  # FIXED
            "url": url,
        }

        result = coll.update_one(
            {
                "ticker": ticker,
                "headline": headline,
                "published_at": published_at,
            },
            {"$setOnInsert": doc},
            upsert=True,
        )

        if result.upserted_id is not None:
            inserted += 1

    return inserted


def run_ingestor():
    coll = get_mongo_collection()

    for ticker in TICKERS:
        print(f"Fetching news for {ticker}...")
        data = fetch_news_for_ticker(ticker)
        inserted = insert_news_batch(coll, ticker, data)
        print(f"{ticker}: fetched {len(data)} items, inserted {inserted} new docs.")


if __name__ == "__main__":
    run_ingestor()
