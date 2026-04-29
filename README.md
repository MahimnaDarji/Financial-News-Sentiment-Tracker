## Step 1: Go to project folder

```powershell
cd "C:\Users\darji\Desktop\financial-news-sentiment-tracker"
```

## Step 2: Activate virtual environment

```powershell
.\.venv\Scripts\activate
```

## Step 3: Start Docker services

```powershell
docker compose up -d
```

## Step 4: Check containers

```powershell
docker ps --format "{{.Names}}"
```

You should see:

```text
fnst_postgres
fnst_mongo
fnst_mongo_express
fnst_adminer
```

## Step 5: Initialize Postgres tables

```powershell
Get-Content db/init.sql | docker exec -i fnst_postgres psql -U finuser -d findb
```

## Step 6: Verify tables

```powershell
docker exec -it fnst_postgres psql -U finuser -d findb
```

Inside Postgres:

```sql
\dt
```

Exit:

```sql
\q
```

## Step 7: Fetch news data

```powershell
python -m services.ingestor.fetch_news
```

## Step 8: Fetch price data

```powershell
python -m services.ingestor.price_fetcher
```

## Step 9: Run sentiment scoring

```powershell
python -m services.processor.sentiment_worker
```

## Step 10: Build daily metrics

```powershell
python -m services.processor.daily_aggregator
```

## Step 11: Build rolling correlation

```powershell
python -m services.processor.daily_correlation
```

## Step 12: Build time series output

```powershell
python -m services.processor.timeseries_builder
```

## Step 13: Start FastAPI backend

Keep this terminal running:

```powershell
uvicorn services.api.main:app --reload --host 127.0.0.1 --port 8000
```

Check API:

```text
http://127.0.0.1:8000/tickers
```

## Step 14: Start Streamlit dashboard in a new terminal

```powershell
cd "C:\Users\darji\Desktop\financial-news-sentiment-tracker"
.\.venv\Scripts\activate
streamlit run services/dashboard/app.py
```

Dashboard opens at:

```text
http://localhost:8501
```
