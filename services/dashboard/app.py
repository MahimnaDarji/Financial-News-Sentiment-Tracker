import os
from typing import List, Dict, Any

import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

API_BASE_URL = os.getenv("FNST_API_BASE_URL", "http://127.0.0.1:8000")
NEON_GREEN = "#39ff14"
RED_FALL = "#f97373"
NEUTRAL_FLAT = "#9ca3af"


# --------------- API HELPERS ---------------

def fetch_tickers() -> List[str]:
    url = f"{API_BASE_URL}/tickers"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("tickers", [])


def fetch_timeseries(ticker: str, days: int) -> Dict[str, Any]:
    url = f"{API_BASE_URL}/ticker/{ticker}/timeseries"
    params = {"days": days}
    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code == 404:
        return {"ticker": ticker, "points": []}
    resp.raise_for_status()
    return resp.json()


def build_dataframe(timeseries: Dict[str, Any]) -> pd.DataFrame:
    points = timeseries.get("points", [])
    if not points:
        return pd.DataFrame()

    df = pd.DataFrame(points)

    numeric_cols = [
        "close_price",
        "daily_return",
        "avg_sentiment",
        "article_count",
        "rolling_corr_7d",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# --------------- SMALL HELPERS ---------------

def sentiment_label(value: float) -> str:
    if pd.isna(value):
        return "Unknown"
    if value >= 0.2:
        return "Bullish"
    if value <= -0.2:
        return "Bearish"
    return "Neutral"


def sentiment_class(value: float) -> str:
    if pd.isna(value):
        return "badge-neutral"
    if value >= 0.2:
        return "badge-bullish"
    if value <= -0.2:
        return "badge-bearish"
    return "badge-neutral"


def apply_chart_theme(fig, dark_mode: bool):
    """Apply transparent background and neon compatible style."""
    if dark_mode:
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
        )
    else:
        fig.update_layout(
            template="plotly_white",
            plot_bgcolor="rgba(255,255,255,0)",
            paper_bgcolor="rgba(255,255,255,0)",
            font_color="#111827",
        )
    return fig


def make_segmented_line(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    dark_mode: bool,
    line_width: int = 3,
) -> go.Figure:
    """Build a line chart where each segment is colored by direction."""
    fig = go.Figure()

    # Need at least 2 points for a segment
    if len(df) < 2 or y_col not in df.columns or x_col not in df.columns:
        return fig

    for i in range(1, len(df)):
        x_pair = [df[x_col].iloc[i - 1], df[x_col].iloc[i]]
        y_prev = df[y_col].iloc[i - 1]
        y_curr = df[y_col].iloc[i]

        if pd.isna(y_prev) or pd.isna(y_curr):
            continue

        if y_curr > y_prev:
            color = NEON_GREEN
        elif y_curr < y_prev:
            color = RED_FALL
        else:
            color = NEUTRAL_FLAT

        fig.add_trace(
            go.Scatter(
                x=x_pair,
                y=[y_prev, y_curr],
                mode="lines",
                line=dict(color=color, width=line_width),
                showlegend=False,
            )
        )

    fig.update_layout(margin=dict(l=10, r=10, t=25, b=10))
    fig = apply_chart_theme(fig, dark_mode)
    return fig


# --------------- WATCHLIST RENDER ---------------

def render_ticker_watchlist(tickers: List[str], days: int, dark_mode: bool):
    if not tickers:
        return

    top_n = min(4, len(tickers))
    watchlist_tickers = tickers[:top_n]

    st.markdown('<div class="glass-card card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-title" style="margin-bottom:0.8rem;">Market watchlist</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(top_n)

    for i, tk in enumerate(watchlist_tickers):
        with cols[i]:
            try:
                ts_card = fetch_timeseries(tk, days)
                df_card = build_dataframe(ts_card)
                if df_card.empty:
                    st.markdown(
                        f'<div class="ticker-title">{tk}</div>'
                        '<div class="ticker-sub">No data</div>',
                        unsafe_allow_html=True,
                    )
                    continue

                latest = df_card.iloc[-1]
                price = latest.get("close_price", float("nan"))
                daily_ret = latest.get("daily_return", float("nan"))
                daily_ret_pct = daily_ret * 100 if not pd.isna(daily_ret) else float("nan")

                st.markdown(
                    f"""
                    <div class="ticker-card-header">
                        <div class="ticker-title">{tk}</div>
                        <div class="ticker-price">{price:.2f}</div>
                        <div class="ticker-change {'pos' if daily_ret_pct >= 0 else 'neg'}">
                            {daily_ret_pct:+.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                spark_df = df_card[["date", "close_price"]].copy()
                fig_spark = px.line(
                    spark_df,
                    x="date",
                    y="close_price",
                    color_discrete_sequence=[NEON_GREEN],
                )
                fig_spark.update_traces(line_width=2)
                fig_spark.update_layout(
                    showlegend=False,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    height=110,
                )
                fig_spark = apply_chart_theme(fig_spark, dark_mode)
                st.plotly_chart(fig_spark, use_container_width=True)
            except Exception:
                st.markdown(
                    f'<div class="ticker-title">{tk}</div>'
                    '<div class="ticker-sub">Error loading</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)


# --------------- MAIN UI ---------------

def main():
    st.set_page_config(
        page_title="Financial News Sentiment Tracker",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True

    # --------------- SIDEBAR ---------------

    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:1rem;">
                <div style="
                    width:36px; height:36px; border-radius:999px;
                    display:flex; align-items:center; justify-content:center;
                    background: radial-gradient(circle at 30% 30%, #ffffff, #22c55e, #0f172a);
                    box-shadow: 0 0 20px rgba(34,197,94,0.8);
                    font-weight:700; color:#020617; font-size:0.9rem;
                ">
                    FN
                </div>
                <div style="display:flex; flex-direction:column;">
                    <span style="font-weight:600; font-size:0.95rem;">Financial News Sentiment Tracker</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Controls")

        dark_mode = st.checkbox("Dark mode", value=st.session_state.dark_mode)
        st.session_state.dark_mode = dark_mode

        try:
            tickers = fetch_tickers()
        except Exception as e:
            st.error(f"Failed to load tickers from API: {e}")
            return

        if not tickers:
            st.error("No tickers available from API.")
            return

        ticker = st.selectbox("Select ticker", options=tickers, index=0)
        days = st.slider("Days of history", min_value=3, max_value=60, value=14)
        st.markdown("---")
        st.caption(f"API base: `{API_BASE_URL}`")

    # --------------- THEME CSS ---------------

    if dark_mode:
        bg_color = "#020617"
        text_color = "#e5e7eb"
        subtext_color = "#9ca3af"
        card_border = "rgba(148,163,184,0.35)"
        shadow = "0 18px 45px rgba(0,0,0,0.65)"
    else:
        bg_color = "#e5e7eb"
        text_color = "#111827"
        subtext_color = "#4b5563"
        card_border = "rgba(148,163,184,0.45)"
        shadow = "0 18px 40px rgba(148,163,184,0.45)"

    st.markdown(
        f"""
        <style>
        body {{
            background-color: {bg_color};
        }}
        .stApp {{
            background-color: {bg_color};
        }}
        .main {{
            background-color: {bg_color};
        }}
        .app-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            margin-bottom: 1rem;
        }}
        .app-title {{
            font-size: 1.8rem;
            font-weight: 700;
            color: {text_color};
        }}
        .app-subtitle {{
            font-size: 0.95rem;
            color: {subtext_color};
        }}
        .app-meta {{
            font-size: 0.85rem;
            color: {subtext_color};
            text-align: right;
        }}
        .card {{
            border-radius: 1rem;
            padding: 1.1rem 1.3rem;
            margin-bottom: 1.2rem;
        }}
        .glass-card {{
            background: linear-gradient(
                135deg,
                rgba(15,23,42,0.91),
                rgba(15,23,42,0.72)
            );
            border: 1px solid {card_border};
            box-shadow: {shadow};
            backdrop-filter: blur(22px);
        }}
        .card-title {{
            font-size: 1rem;
            font-weight: 600;
            color: {text_color};
            margin-bottom: 0.6rem;
        }}
        .badge {{
            display: inline-block;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .badge-bullish {{
            background: rgba(34,197,94,0.1);
            color: {NEON_GREEN};
            border: 1px solid rgba(34,197,94,0.6);
            box-shadow: 0 0 12px rgba(34,197,94,0.6);
        }}
        .badge-bearish {{
            background: rgba(248,113,113,0.08);
            color: #f97373;
            border: 1px solid rgba(248,113,113,0.5);
        }}
        .badge-neutral {{
            background: rgba(148,163,184,0.08);
            color: {subtext_color};
            border: 1px solid rgba(148,163,184,0.5);
        }}
        .kpi-label {{
            font-size: 0.8rem;
            color: {subtext_color};
        }}
        .kpi-value {{
            font-size: 1.3rem;
            font-weight: 600;
            color: {text_color};
        }}
        .kpi-sub {{
            font-size: 0.8rem;
            color: {subtext_color};
        }}
        .ticker-card-header {{
            display:flex;
            flex-direction:column;
            gap:0.05rem;
            margin-bottom:0.25rem;
        }}
        .ticker-title {{
            font-size:0.8rem;
            font-weight:600;
            color:{subtext_color};
            text-transform:uppercase;
            letter-spacing:0.04em;
        }}
        .ticker-price {{
            font-size:1.1rem;
            font-weight:600;
            color:{text_color};
        }}
        .ticker-change {{
            font-size:0.8rem;
            font-weight:500;
        }}
        .ticker-change.pos {{
            color:{NEON_GREEN};
        }}
        .ticker-change.neg {{
            color:{RED_FALL};
        }}
        .ticker-sub {{
            font-size:0.8rem;
            color:{subtext_color};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --------------- FETCH MAIN DATA ---------------

    with st.spinner(f"Loading data for {ticker} ..."):
        try:
            ts = fetch_timeseries(ticker, days)
        except Exception as e:
            st.error(f"Failed to load time series for {ticker}: {e}")
            return

    df = build_dataframe(ts)

    if df.empty:
        st.warning("No data returned for this ticker and window.")
        return

    from_date = ts.get("from_date", "")
    to_date = ts.get("to_date", "")

    latest = df.iloc[-1]
    latest_sentiment = latest.get("avg_sentiment", float("nan"))
    sentiment_tag = sentiment_label(latest_sentiment)
    sentiment_css_class = sentiment_class(latest_sentiment)

    # --------------- HEADER ---------------

    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <div class="app-title">📈 {ticker} News Sentiment Dashboard</div>
            </div>
            <div class="app-meta">
                <div style="margin-top:0.3rem;">
                    Current sentiment:
                    <span class="badge {sentiment_css_class}">{sentiment_tag}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------- TICKER WATCHLIST ---------------

    render_ticker_watchlist(tickers, days=7, dark_mode=dark_mode)

    # --------------- KPI CARDS ---------------

    st.markdown('<div class="glass-card card">', unsafe_allow_html=True)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown('<div class="kpi-label">Close price</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="kpi-value">{latest["close_price"]:.2f}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="kpi-sub">Last trading session</div>',
            unsafe_allow_html=True,
        )

    with kpi2:
        daily_ret_pct = latest["daily_return"] * 100 if not pd.isna(latest["daily_return"]) else 0
        st.markdown('<div class="kpi-label">Daily return</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="kpi-value">{daily_ret_pct:+.2f}%</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="kpi-sub">Change vs previous close</div>',
            unsafe_allow_html=True,
        )

    with kpi3:
        st.markdown('<div class="kpi-label">Average sentiment</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="kpi-value">{latest_sentiment:.3f}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="kpi-sub">{sentiment_tag} news tone</div>',
            unsafe_allow_html=True,
        )

    with kpi4:
        st.markdown('<div class="kpi-label">Articles today</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="kpi-value">{int(latest["article_count"])}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="kpi-sub">Ingested news items</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------- TABS ---------------

    tab_charts, tab_table = st.tabs(["Detailed charts", "Raw data"])

    # --------------- DETAILED CHARTS TAB ---------------

    with tab_charts:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="glass-card card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Close price over time</div>', unsafe_allow_html=True)
            fig_price = make_segmented_line(df, "date", "close_price", dark_mode, line_width=3)
            st.plotly_chart(fig_price, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="glass-card card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Average sentiment over time</div>', unsafe_allow_html=True)
            fig_sent = make_segmented_line(df, "date", "avg_sentiment", dark_mode, line_width=3)
            st.plotly_chart(fig_sent, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        c3, c4 = st.columns(2)

        with c3:
            st.markdown('<div class="glass-card card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Article count per day</div>', unsafe_allow_html=True)
            fig_count = px.bar(
                df,
                x="date",
                y="article_count",
                color_discrete_sequence=[NEON_GREEN],
            )
            fig_count.update_layout(margin=dict(l=10, r=10, t=25, b=10))
            fig_count = apply_chart_theme(fig_count, dark_mode)
            st.plotly_chart(fig_count, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c4:
            st.markdown('<div class="glass-card card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">7 day rolling correlation</div>', unsafe_allow_html=True)
            fig_corr = make_segmented_line(df, "date", "rolling_corr_7d", dark_mode, line_width=3)
            st.plotly_chart(fig_corr, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # --------------- RAW DATA TAB ---------------

    with tab_table:
        st.markdown('<div class="glass-card card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Raw daily metrics</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
