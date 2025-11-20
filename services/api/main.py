import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# Import the timeseries builder
from services.processor.timeseries_builder import build_ticker_timeseries

# ------------------------------
# Load .env from project root
# ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB", "financial_news")

# Central ticker list
TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA"]


# ------------------------------
# Pydantic response models
# ------------------------------
class TimeSeriesPoint(BaseModel):
    date: str
    close_price: Optional[float] = None
    daily_return: Optional[float] = None
    avg_sentiment: Optional[float] = None
    dominant_sentiment: Optional[str] = None
    article_count: int
    rolling_corr_7d: Optional[float] = None


class TimeSeriesResponse(BaseModel):
    ticker: str
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    points: List[TimeSeriesPoint]


# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(
    title="Financial News Sentiment Tracker API",
    version="1.0",
    description="API for price, sentiment, and correlation metrics.",
)


# ------------------------------
# Root landing page
# ------------------------------
@app.get("/")
def root():
    return {
        "message": "Financial News Sentiment Tracker API",
        "description": "Use /docs to explore all available endpoints.",
        "endpoints": {
            "/health": "Health check endpoint",
            "/tickers": "List supported tickers",
            "/ticker/{ticker}/timeseries?days=7": "Get sentiment + price timeseries",
            "/docs": "Swagger API Docs",
            "/redoc": "ReDoc API Docs"
        },
    }


# ------------------------------
# Health
# ------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}


# ------------------------------
# Ticker list
# ------------------------------
@app.get("/tickers")
def list_tickers():
    return {"tickers": TICKERS}


# ------------------------------
# Time series for a ticker
# ------------------------------
@app.get("/ticker/{ticker}/timeseries", response_model=TimeSeriesResponse)
def get_ticker_timeseries(
    ticker: str,
    days: int = Query(
        7,
        ge=1,
        le=90,
        description="How many days to return (default: 7, max: 90)",
    ),
):
    ticker = ticker.upper()

    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} is not supported")

    data = build_ticker_timeseries(ticker, days_back=days)

    if not data.get("points"):
        raise HTTPException(
            status_code=404,
            detail=f"No daily metrics found for ticker {ticker}",
        )

    return data
