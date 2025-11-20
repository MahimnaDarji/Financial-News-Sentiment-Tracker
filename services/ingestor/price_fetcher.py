import os
import time
from datetime import datetime

import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING

# Load .env from project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB", "financial_news")

# Same ticker set as news
TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA"]


def get_price_collection():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    coll = db["price_snapshots"]

    coll.create_index(
        [("ticker", ASCENDING), ("ts", ASCENDING)],
        unique=True,
        name="uniq_ticker_ts",
    )

    return coll


def fetch_daily_prices_yf(ticker: str, period: str = "90d"):
    """
    Fetch daily OHLCV data from Yahoo Finance using the Ticker.history API.
    This avoids the MultiIndex / group_by weirdness of yf.download.
    """
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval="1d", auto_adjust=False)

    if df.empty:
        print(f"[WARN] No price data returned for {ticker} from yfinance.history().")
        return []

    # Ensure simple string column names
    df.columns = [str(c) for c in df.columns]

    # Typical columns: Open, High, Low, Close, Volume, Dividends, Stock Splits
    close_col = None
    if "Close" in df.columns:
        close_col = "Close"
    elif "Adj Close" in df.columns:
        close_col = "Adj Close"
    else:
        print(f"[WARN] No 'Close' or 'Adj Close' column for {ticker}. Columns: {list(df.columns)}")
        return []

    candles = []
    for idx, row in df.iterrows():
        # idx is a pandas Timestamp
        dt = datetime.utcfromtimestamp(idx.timestamp())

        close_val = row.get(close_col, None)
        if pd.isna(close_val):
            continue

        open_val = row.get("Open", None)
        high_val = row.get("High", None)
        low_val = row.get("Low", None)
        vol_val = row.get("Volume", 0.0)

        if any(pd.isna(v) for v in [open_val, high_val, low_val]):
            continue

        try:
            vol_val = float(vol_val)
        except Exception:
            vol_val = 0.0

        candles.append(
            {
                "ticker": ticker,
                "ts": dt,
                "open": float(open_val),
                "high": float(high_val),
                "low": float(low_val),
                "close": float(close_val),
                "volume": vol_val,
                "source": "yfinance_history",
            }
        )

    return candles


def upsert_candles(coll, candles):
    inserted = 0
    for candle in candles:
        result = coll.update_one(
            {
                "ticker": candle["ticker"],
                "ts": candle["ts"],
            },
            {"$setOnInsert": candle},
            upsert=True,
        )
        if result.upserted_id is not None:
            inserted += 1
    return inserted


def run_price_fetcher(period: str = "90d", sleep_between: float = 0.5):
    coll = get_price_collection()

    for ticker in TICKERS:
        print(f"Fetching {period} of daily prices for {ticker} from yfinance.history()...")
        candles = fetch_daily_prices_yf(ticker, period=period)
        print(f"{ticker}: fetched {len(candles)} candles from yfinance.history().")
        if not candles:
            continue
        inserted = upsert_candles(coll, candles)
        print(f"{ticker}: inserted {inserted} new documents into Mongo.")
        time.sleep(sleep_between)


if __name__ == "__main__":
    run_price_fetcher()
