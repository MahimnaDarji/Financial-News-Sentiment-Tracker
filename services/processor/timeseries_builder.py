import os
from datetime import datetime, timedelta, UTC
from typing import Dict, Any, List

from dotenv import load_dotenv
from pymongo import MongoClient

# Load .env from project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB", "financial_news")


def get_daily_collection():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    return db["daily_ticker_metrics"]


def get_latest_date_for_ticker(coll, ticker: str):
    """
    Return the latest date (as a date object) for which we have daily metrics.
    """
    doc = coll.find_one(
        {"ticker": ticker},
        sort=[("date", -1)],
        projection={"date": 1},
    )
    if not doc:
        return None

    d: datetime = doc["date"]
    return d.date()


def build_ticker_timeseries(
    ticker: str,
    days_back: int = 30,
) -> Dict[str, Any]:
    """
    Build a JSON-ready time series structure for a given ticker.

    Each point contains:
      - date (YYYY-MM-DD string)
      - close_price
      - daily_return
      - avg_sentiment
      - dominant_sentiment
      - article_count
      - rolling_corr_7d
    """
    coll = get_daily_collection()

    latest_date = get_latest_date_for_ticker(coll, ticker)
    if latest_date is None:
        return {"ticker": ticker, "points": []}

    start_date = latest_date - timedelta(days=days_back - 1)
    start_dt = datetime(start_date.year, start_date.month, start_date.day)

    cursor = coll.find(
        {
            "ticker": ticker,
            "date": {"$gte": start_dt},
        },
        sort=[("date", 1)],
    )

    points: List[Dict[str, Any]] = []

    for doc in cursor:
        d: datetime = doc["date"]  
        points.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "close_price": doc.get("close_price"),
                "daily_return": doc.get("daily_return"),
                "avg_sentiment": doc.get("avg_sentiment_score"),
                "dominant_sentiment": doc.get("dominant_sentiment_label"),
                "article_count": doc.get("article_count", 0),
                "rolling_corr_7d": doc.get("rolling_corr_7d"),
            }
        )

    return {
        "ticker": ticker,
        "from_date": start_date.strftime("%Y-%m-%d"),
        "to_date": latest_date.strftime("%Y-%m-%d"),
        "points": points,
    }


if __name__ == "__main__":
    data = build_ticker_timeseries("AAPL", days_back=7)
    print(
        f"Built time series for {data['ticker']} from "
        f"{data.get('from_date')} to {data.get('to_date')}"
    )
    for p in data["points"]:
        print(p)
