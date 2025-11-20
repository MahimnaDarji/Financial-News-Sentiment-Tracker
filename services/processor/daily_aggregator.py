import os
from datetime import datetime, timedelta, date
from collections import Counter

from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING

# Load .env from project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB", "financial_news")

# Same ticker set as before
TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA"]


def get_collections():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    news = db["news_events"]
    prices = db["price_snapshots"]
    daily = db["daily_ticker_metrics"]

    daily.create_index(
        [("ticker", ASCENDING), ("date", ASCENDING)],
        unique=True,
        name="uniq_ticker_date",
    )

    return news, prices, daily


def start_of_day(d: date) -> datetime:
    """Return naive UTC midnight for a given date."""
    return datetime(d.year, d.month, d.day)


def build_price_series(prices_coll, ticker: str):
    """
    Build a dict: date -> {close, daily_return}
    using price_snapshots (daily closes).
    """
    cursor = prices_coll.find(
        {"ticker": ticker},
        sort=[("ts", ASCENDING)],
    )

    rows = list(cursor)
    if not rows:
        return {}

    series = []
    for doc in rows:
        ts: datetime = doc["ts"]  # naive datetime from yfinance
        d = ts.date()
        close = float(doc["close"])
        series.append((d, close))

    result = {}
    prev_close = None
    for d, close in series:
        if prev_close is None:
            daily_ret = None
        else:
            daily_ret = (close - prev_close) / prev_close if prev_close != 0 else None
        result[d] = {
            "close_price": close,
            "daily_return": daily_ret,
        }
        prev_close = close

    return result


def aggregate_sentiment_for_day(news_coll, ticker: str, d: date):
    """
    Aggregate sentiment for (ticker, date) from news_events.
    Returns (avg_sentiment_score, dominant_label, article_count)
    """
    day_start = start_of_day(d)
    day_end = day_start + timedelta(days=1)

    cursor = news_coll.find(
        {
            "ticker": ticker,
            "published_at": {"$gte": day_start, "$lt": day_end},
            "sentiment_score": {"$ne": None},
        },
        {"sentiment_score": 1, "sentiment_label": 1},
    )

    docs = list(cursor)
    if not docs:
        return None, None, 0

    scores = [float(doc["sentiment_score"]) for doc in docs if "sentiment_score" in doc]
    labels = [doc.get("sentiment_label") for doc in docs if doc.get("sentiment_label")]

    avg_score = sum(scores) / len(scores) if scores else None

    if labels:
        counter = Counter(labels)
        dominant = counter.most_common(1)[0][0]
    else:
        dominant = None

    return avg_score, dominant, len(docs)


def upsert_daily_metric(
    daily_coll,
    ticker: str,
    d: date,
    close_price,
    daily_return,
    avg_sentiment,
    dominant_label,
    article_count,
):
    """
    Upsert one document into daily_ticker_metrics.
    Store `date` as a datetime at midnight (Mongo-friendly).
    """
    date_dt = start_of_day(d)

    daily_coll.update_one(
        {
            "ticker": ticker,
            "date": date_dt,
        },
        {
            "$set": {
                "ticker": ticker,
                "date": date_dt,
                "close_price": close_price,
                "daily_return": daily_return,
                "avg_sentiment_score": avg_sentiment,
                "dominant_sentiment_label": dominant_label,
                "article_count": article_count,
                "updated_at": datetime.now(UTC),
            }
        },
        upsert=True,
    )


def run_daily_aggregation(days_back: int = 90):
    news_coll, prices_coll, daily_coll = get_collections()

    today = datetime.utcnow().date()
    start_date = today - timedelta(days=days_back)

    for ticker in TICKERS:
        print(f"\n=== Aggregating daily metrics for {ticker} ===")
        price_series = build_price_series(prices_coll, ticker)

        if not price_series:
            print(f"[WARN] No price data for {ticker}, skipping.")
            continue

        for d in sorted(price_series.keys()):
            if d < start_date:
                continue

            price_info = price_series[d]
            close_price = price_info["close_price"]
            daily_return = price_info["daily_return"]

            avg_sentiment, dominant_label, article_count = aggregate_sentiment_for_day(
                news_coll, ticker, d
            )

            upsert_daily_metric(
                daily_coll,
                ticker,
                d,
                close_price,
                daily_return,
                avg_sentiment,
                dominant_label,
                article_count,
            )

        print(f"Finished {ticker}.")


if __name__ == "__main__":
    run_daily_aggregation(days_back=30) 
