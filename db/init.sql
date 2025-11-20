-- Table for news articles and their sentiment
CREATE TABLE IF NOT EXISTS news_events (
    id SERIAL PRIMARY KEY,
    source            VARCHAR(50),
    ticker            VARCHAR(10),
    headline          TEXT NOT NULL,
    summary           TEXT,
    sentiment_label   VARCHAR(20),
    sentiment_score   NUMERIC,
    published_at      TIMESTAMPTZ NOT NULL,
    ingested_at       TIMESTAMPTZ DEFAULT NOW(),
    url               TEXT
);

-- Table for price snapshots (intraday or EOD)
CREATE TABLE IF NOT EXISTS price_snapshots (
    id SERIAL PRIMARY KEY,
    ticker       VARCHAR(10) NOT NULL,
    price        NUMERIC NOT NULL,
    ts           TIMESTAMPTZ NOT NULL
);


CREATE TABLE IF NOT EXISTS daily_ticker_metrics (
    id SERIAL PRIMARY KEY,
    ticker                  VARCHAR(10) NOT NULL,
    date                    DATE NOT NULL,
    avg_sentiment_score     NUMERIC,
    sentiment_label_dominant VARCHAR(20),
    daily_return            NUMERIC,
    correlation_7d          NUMERIC,
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes to speed up queries
CREATE INDEX IF NOT EXISTS idx_news_ticker_time
    ON news_events (ticker, published_at);

CREATE INDEX IF NOT EXISTS idx_price_ticker_time
    ON price_snapshots (ticker, ts);

CREATE INDEX IF NOT EXISTS idx_daily_metrics_ticker_date
    ON daily_ticker_metrics (ticker, date);
