import os
from datetime import datetime, timedelta, UTC
from typing import List, Tuple, Optional

from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING

# Load .env from project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB", "financial_news")

# Same ticker set
TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA"]


def get_daily_collection():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    coll = db["daily_ticker_metrics"]

    coll.create_index(
        [("ticker", ASCENDING), ("date", ASCENDING)],
        name="idx_ticker_date",
    )

    return coll


def pearson_corr(pairs: List[Tuple[float, float]]) -> Optional[float]:
    """
    Compute Pearson correlation for a list of (x, y) pairs.
    Returns None if fewer than 2 valid points.
    """
    n = len(pairs)
    if n < 2:
        return None

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)

    if den_x == 0 or den_y == 0:
        return None

    return num / (den_x * den_y) ** 0.5


def update_rolling_corr_for_ticker(coll, ticker: str, window: int = 7):
    """
    For a single ticker:
      - sort daily rows by date
      - compute rolling 7-day correlation between daily_return and avg_sentiment_score
      - store result in field rolling_corr_7d
    """
    cursor = coll.find({"ticker": ticker}).sort("date", ASCENDING)
    docs = list(cursor)

    if not docs:
        print(f"[WARN] No daily metrics for {ticker}")
        return

    history = []  # list of dicts: {"ret": float or None, "sent": float or None, "date": datetime}

    updated = 0

    for doc in docs:
        d = doc["date"]
        daily_ret = doc.get("daily_return")
        sent = doc.get("avg_sentiment_score")

        history.append({"date": d, "ret": daily_ret, "sent": sent})
        if len(history) > window:
            history.pop(0)

        # Build list of valid (ret, sent) in the current window
        pairs = [
            (h["ret"], h["sent"])
            for h in history
            if h["ret"] is not None and h["sent"] is not None
        ]

        corr = pearson_corr(pairs) if pairs else None

        coll.update_one(
            {"_id": doc["_id"]},
            {
                "$set": {
                    "rolling_corr_7d": corr,
                    "corr_updated_at": datetime.now(UTC),
                }
            },
        )
        updated += 1

    print(f"{ticker}: updated rolling_corr_7d for {updated} daily rows.")


def run_rolling_corr(window: int = 7):
    coll = get_daily_collection()

    for ticker in TICKERS:
        print(f"\n=== Computing {window}-day rolling correlation for {ticker} ===")
        update_rolling_corr_for_ticker(coll, ticker, window=window)


if __name__ == "__main__":
    run_rolling_corr(window=7)
