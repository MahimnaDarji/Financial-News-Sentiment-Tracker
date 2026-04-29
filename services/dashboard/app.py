import os
from datetime import datetime
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
    """Apply institutional chart styling."""
    if dark_mode:
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
        )
    else:
        fig.update_layout(
            plot_bgcolor="rgba(255,255,255,0)",
            paper_bgcolor="rgba(255,255,255,0)",
            font_color="#111827",
        )

    fig.update_layout(
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0f172a" if dark_mode else "#ffffff",
            bordercolor="rgba(148,163,184,0.25)",
            font_size=12,
            font_family="Arial, sans-serif",
        ),
        margin=dict(l=10, r=10, t=16, b=10),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="rgba(148,163,184,0.18)",
            tickfont=dict(size=11, color="#94a3b8" if dark_mode else "#64748b"),
            ticks="outside",
            ticklen=4,
            tickcolor="rgba(148,163,184,0.24)",
            tickformat="%b %d",
            fixedrange=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.08)",
            zeroline=False,
            showline=False,
            tickfont=dict(size=11, color="#94a3b8" if dark_mode else "#64748b"),
            ticks="outside",
            ticklen=4,
            tickcolor="rgba(148,163,184,0.16)",
            fixedrange=False,
        ),
    )
    return fig


def apply_signal_axis_style(fig, dark_mode: bool, zero_line: bool = True):
    fig = apply_chart_theme(fig, dark_mode)
    yaxis = dict(
        range=[-1.15, 1.15],
        tickmode="array",
        tickvals=[-1, -0.5, 0, 0.5, 1],
        tickfont=dict(size=11, color="#94a3b8" if dark_mode else "#64748b"),
        showgrid=True,
        gridcolor="rgba(148,163,184,0.08)",
        zeroline=False,
    )
    if zero_line:
        fig.add_hline(y=0, line_width=1, line_color="rgba(148,163,184,0.35)")
    fig.update_yaxes(**yaxis)
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
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig = apply_chart_theme(fig, dark_mode)
    return fig


def make_price_chart(df: pd.DataFrame, dark_mode: bool, height: int = 430) -> go.Figure:
    fig = go.Figure()

    if len(df) < 2 or "date" not in df.columns or "close_price" not in df.columns:
        return apply_chart_theme(fig, dark_mode)

    df_plot = df[["date", "close_price", "daily_return"]].copy()

    fig.add_trace(
        go.Scatter(
            x=df_plot["date"],
            y=df_plot["close_price"],
            mode="lines",
            line=dict(color="rgba(148,163,184,0.72)", width=1.6),
            fill="tozeroy",
            fillcolor="rgba(57,255,20,0.08)",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Close: %{y:.2f}<extra>Latest movement</extra>",
            name="Price",
            showlegend=False,
        )
    )

    for i in range(1, len(df_plot)):
        x_pair = [df_plot["date"].iloc[i - 1], df_plot["date"].iloc[i]]
        y_pair = [df_plot["close_price"].iloc[i - 1], df_plot["close_price"].iloc[i]]
        prev_val = df_plot["close_price"].iloc[i - 1]
        curr_val = df_plot["close_price"].iloc[i]

        if pd.isna(prev_val) or pd.isna(curr_val):
            continue

        move = curr_val - prev_val
        if abs(move) < 0.001:
            continue

        color = NEON_GREEN if move > 0 else RED_FALL
        fig.add_trace(
            go.Scatter(
                x=x_pair,
                y=y_pair,
                mode="lines",
                line=dict(color=color, width=2.4),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    latest_idx = df_plot.index[-1]
    latest_date = df_plot.loc[latest_idx, "date"]
    latest_price = df_plot.loc[latest_idx, "close_price"]
    fig.add_trace(
        go.Scatter(
            x=[latest_date],
            y=[latest_price],
            mode="markers+text",
            text=["Current"],
            textposition="top center",
            marker=dict(size=10, color=NEON_GREEN, line=dict(color="#081120", width=1.5)),
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Close: %{y:.2f}<extra>Current price</extra>",
            showlegend=False,
        )
    )

    fig.update_layout(height=height)
    fig = apply_chart_theme(fig, dark_mode)
    fig.update_yaxes(tickformat=".2f")
    return fig


def make_sparkline(df: pd.DataFrame, dark_mode: bool) -> go.Figure:
    fig = px.line(df, x="date", y="close_price")
    fig.update_traces(line=dict(color=NEON_GREEN, width=1.8), hoverinfo="skip")
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        height=72,
    )
    return apply_chart_theme(fig, dark_mode)


def make_volume_bars(df: pd.DataFrame, dark_mode: bool, height: int = 180) -> go.Figure:
    fig = px.bar(df, x="date", y="article_count", color_discrete_sequence=[NEON_GREEN])
    fig.update_traces(
        marker_line_width=0,
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Articles: %{y}<extra>Volume</extra>",
    )
    fig.update_layout(height=height)
    fig = apply_chart_theme(fig, dark_mode)
    fig.update_yaxes(tickformat="d")
    return fig


def make_signal_band(
    df: pd.DataFrame,
    value_col: str,
    dark_mode: bool,
    height: int = 180,
    line_width: int = 3,
) -> go.Figure:
    fig = make_segmented_line(df, "date", value_col, dark_mode, line_width=line_width)
    fig.update_layout(height=height)
    return fig


def make_sentiment_chart(df: pd.DataFrame, dark_mode: bool, height: int = 210) -> go.Figure:
    fig = go.Figure()
    if len(df) < 2 or "date" not in df.columns or "avg_sentiment" not in df.columns:
        return apply_signal_axis_style(fig, dark_mode)

    min_x = df["date"].iloc[0]
    max_x = df["date"].iloc[-1]

    shapes = [
        dict(type="rect", xref="x", yref="y", x0=min_x, x1=max_x, y0=0.2, y1=1.15, fillcolor="rgba(34,197,94,0.08)", line_width=0, layer="below"),
        dict(type="rect", xref="x", yref="y", x0=min_x, x1=max_x, y0=-0.2, y1=0.2, fillcolor="rgba(148,163,184,0.08)", line_width=0, layer="below"),
        dict(type="rect", xref="x", yref="y", x0=min_x, x1=max_x, y0=-1.15, y1=-0.2, fillcolor="rgba(248,113,113,0.08)", line_width=0, layer="below"),
    ]

    fig.update_layout(shapes=shapes)

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["avg_sentiment"],
            mode="lines",
            line=dict(color="#cbd5e1", width=1.8),
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Sentiment: %{y:.3f}<extra>Sentiment trend</extra>",
            showlegend=False,
        )
    )

    latest_row = df.iloc[-1]
    fig.add_trace(
        go.Scatter(
            x=[latest_row["date"]],
            y=[latest_row["avg_sentiment"]],
            mode="markers+text",
            text=["Current sentiment"],
            textposition="top center",
            marker=dict(size=10, color=NEON_GREEN if latest_row["avg_sentiment"] >= 0 else RED_FALL, line=dict(color="#081120", width=1.5)),
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Sentiment: %{y:.3f}<extra>Current sentiment marker</extra>",
            showlegend=False,
        )
    )

    fig.update_layout(height=height)
    fig = apply_signal_axis_style(fig, dark_mode)
    fig.update_yaxes(tickformat=".1f")
    return fig


def make_correlation_chart(df: pd.DataFrame, dark_mode: bool, height: int = 200) -> go.Figure:
    fig = go.Figure()
    if len(df) < 2 or "date" not in df.columns or "rolling_corr_7d" not in df.columns:
        return apply_signal_axis_style(fig, dark_mode)

    valid = df[["date", "rolling_corr_7d"]].copy()

    fig.add_trace(
        go.Scatter(
            x=valid["date"],
            y=valid["rolling_corr_7d"],
            mode="lines",
            line=dict(color="#cbd5e1", width=1.8),
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Correlation: %{y:.3f}<extra>Signal strength</extra>",
            showlegend=False,
        )
    )

    latest_row = valid.iloc[-1]
    fig.add_trace(
        go.Scatter(
            x=[latest_row["date"]],
            y=[latest_row["rolling_corr_7d"]],
            mode="markers+text",
            text=["Current"],
            textposition="top center",
            marker=dict(size=10, color=NEON_GREEN if latest_row["rolling_corr_7d"] >= 0 else RED_FALL, line=dict(color="#081120", width=1.5)),
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Correlation: %{y:.3f}<extra>Current signal</extra>",
            showlegend=False,
        )
    )

    fig.update_layout(height=height)
    fig = apply_signal_axis_style(fig, dark_mode)
    fig.update_yaxes(tickformat=".2f")
    return fig


def build_signal_text(value: float, label: str, positive_hint: str, negative_hint: str) -> tuple[str, str]:
    if pd.isna(value):
        return "No signal", "Data missing"
    if value > 0:
        return label, positive_hint
    if value < 0:
        return label, negative_hint
    return "Balanced", "Neutral pressure"


def build_actionable_signals(df: pd.DataFrame) -> List[Dict[str, str]]:
    if df.empty:
        return []

    latest = df.iloc[-1]
    latest_price = latest.get("close_price", float("nan"))
    latest_return = latest.get("daily_return", float("nan"))
    latest_return_pct = latest_return * 100 if not pd.isna(latest_return) else float("nan")
    latest_sentiment = latest.get("avg_sentiment", float("nan"))
    latest_articles = latest.get("article_count", float("nan"))
    latest_corr = latest.get("rolling_corr_7d", float("nan"))

    recent_window = df.tail(min(5, len(df)))
    recent_articles_avg = recent_window["article_count"].mean() if "article_count" in recent_window else float("nan")
    recent_return_avg = recent_window["daily_return"].mean() if "daily_return" in recent_window else float("nan")

    signals: List[Dict[str, str]] = []

    if not pd.isna(latest_return_pct) and not pd.isna(latest_sentiment) and latest_return_pct < 0 and abs(latest_sentiment) < 0.15:
        signals.append(
            {
                "label": "Price / tone mismatch",
                "text": "Price dropped while sentiment remained neutral.",
                "tone": "warning",
            }
        )
    elif not pd.isna(latest_return_pct) and latest_return_pct > 0 and not pd.isna(latest_sentiment) and latest_sentiment > 0.15:
        signals.append(
            {
                "label": "Price / tone alignment",
                "text": "Price and sentiment are aligned to the upside.",
                "tone": "good",
            }
        )
    else:
        signals.append(
            {
                "label": "Price / tone check",
                "text": "Price and sentiment are not strongly aligned right now.",
                "tone": "neutral",
            }
        )

    if not pd.isna(latest_articles) and not pd.isna(recent_articles_avg) and recent_articles_avg > 0:
        if latest_articles > recent_articles_avg * 1.2:
            signals.append(
                {
                    "label": "Volume regime",
                    "text": "Article volume is elevated compared to the recent average.",
                    "tone": "good",
                }
            )
        else:
            signals.append(
                {
                    "label": "Volume regime",
                    "text": "Article volume is close to its recent average.",
                    "tone": "neutral",
                }
            )
    else:
        signals.append(
            {
                "label": "Volume regime",
                "text": "Article volume is not yet stable enough to compare cleanly.",
                "tone": "neutral",
            }
        )

    if not pd.isna(latest_corr):
        if abs(latest_corr) < 0.25:
            signals.append(
                {
                    "label": "Signal strength",
                    "text": "Correlation is weak, so sentiment is not currently explaining returns.",
                    "tone": "warning",
                }
            )
        else:
            signals.append(
                {
                    "label": "Signal strength",
                    "text": "Correlation is active enough to matter in the current window.",
                    "tone": "good",
                }
            )

    if len(signals) < 3:
        signals.append(
            {
                "label": "Recent behavior",
                "text": f"Latest price sits at {latest_price:.2f} with a {latest_return_pct:+.2f}% move.",
                "tone": "neutral",
            }
        )

    return signals[:3]


def choose_segmented(label: str, options: List[str], default_index: int = 0, key: str | None = None):
    if hasattr(st, "segmented_control"):
        return st.segmented_control(label, options=options, default=options[default_index], key=key)
    return st.radio(
        label,
        options=options,
        index=default_index,
        horizontal=True,
        key=key,
        label_visibility="collapsed",
    )


# --------------- WATCHLIST RENDER ---------------

def render_ticker_watchlist(tickers: List[str], days: int, dark_mode: bool, selected_ticker: str):
    if not tickers:
        return

    top_n = min(4, len(tickers))
    watchlist_tickers = [selected_ticker] + [tk for tk in tickers if tk != selected_ticker]
    watchlist_tickers = watchlist_tickers[:top_n]

    st.markdown('<div class="glass-card card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">Market strip</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-note">Compact market context across the tracked names.</div>',
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
                trend_label = "Up" if not pd.isna(daily_ret_pct) and daily_ret_pct >= 0 else "Down"
                is_selected = tk == selected_ticker
                card_class = "market-card selected" if is_selected else "market-card"

                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <div class="ticker-card-topline">
                            <div class="ticker-title">{tk}</div>
                            <div class="ticker-price">{price:.2f}</div>
                        </div>
                        <div class="ticker-change {'pos' if daily_ret_pct >= 0 else 'neg'}">
                            {daily_ret_pct:+.2f}%
                        </div>
                        <div class="ticker-subline">
                            {trend_label} | {'Selected' if is_selected else 'Market watch'}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                spark_df = df_card[["date", "close_price"]].copy()
                fig_spark = make_sparkline(spark_df, dark_mode)
                st.plotly_chart(fig_spark, use_container_width=True, config={"displayModeBar": False})
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
            <div class="rail-brand">
                <div class="rail-logo">FN</div>
                <div>
                    <div class="rail-title">Market Dashboard</div>
                    <div class="rail-subtitle">Financial News Sentiment Tracker</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        dark_choice = choose_segmented("Theme", ["Dark", "Light"], default_index=0 if st.session_state.dark_mode else 1, key="theme_choice")
        dark_mode = dark_choice == "Dark"
        st.session_state.dark_mode = dark_mode

        try:
            tickers = fetch_tickers()
        except Exception as e:
            st.error(f"Failed to load tickers from API: {e}")
            return

        if not tickers:
            st.error("No tickers available from API.")
            return

        ticker_index = st.session_state.get("ticker_index", 0)
        if ticker_index >= len(tickers):
            ticker_index = 0

        ticker = st.selectbox("Ticker", options=tickers, index=ticker_index, key="ticker_choice")
        st.session_state.ticker_index = tickers.index(ticker)

        window_choice = choose_segmented("Window", ["7D", "14D", "30D"], default_index=1, key="window_choice")
        days = {"7D": 7, "14D": 14, "30D": 30}[window_choice]

        st.markdown('<div class="rail-section">', unsafe_allow_html=True)
        st.markdown('<div class="rail-label">Control summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="rail-chip">{window_choice}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        status_slot = st.empty()

        st.caption(f"API base: {API_BASE_URL}")

    # --------------- THEME CSS ---------------

    if dark_mode:
        bg_color = "#050b16"
        text_color = "#e5e7eb"
        subtext_color = "#9ca3af"
        panel_bg = "#0b1424"
        panel_bg_alt = "#101b2e"
        card_border = "rgba(148,163,184,0.18)"
        shadow = "0 10px 26px rgba(0,0,0,0.28)"
    else:
        bg_color = "#e5e7eb"
        text_color = "#111827"
        subtext_color = "#4b5563"
        panel_bg = "#f8fafc"
        panel_bg_alt = "#eef2f7"
        card_border = "rgba(148,163,184,0.22)"
        shadow = "0 10px 22px rgba(15,23,42,0.08)"

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
        .block-container {{
            padding-top: 0.7rem;
            padding-bottom: 0.8rem;
        }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #09111f 0%, #0b1424 100%);
            border-right: 1px solid rgba(148,163,184,0.14);
        }}
        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1rem;
            padding-bottom: 1rem;
        }}
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
            margin-bottom: 0.2rem;
        }}
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stRadio label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}
        .app-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
            padding: 1.15rem 1.25rem;
        }}
        .app-title {{
            font-size: 2.1rem;
            font-weight: 700;
            color: {text_color};
            line-height: 1.05;
        }}
        .app-subtitle {{
            font-size: 0.95rem;
            color: {subtext_color};
            margin-top: 0.25rem;
        }}
        .app-meta {{
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            justify-content: flex-end;
            font-size: 0.85rem;
            color: {subtext_color};
        }}
        .card {{
            border-radius: 1rem;
            padding: 0.85rem 1rem;
            margin-bottom: 0.8rem;
        }}
        .glass-card {{
            background: linear-gradient(180deg, {panel_bg}, {panel_bg_alt});
            border: 1px solid {card_border};
            box-shadow: {shadow};
        }}
        .card-title {{
            font-size: 1rem;
            font-weight: 600;
            color: {text_color};
            margin-bottom: 0.6rem;
        }}
        .section-label {{
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: {subtext_color};
            margin-bottom: 0.25rem;
        }}
        .section-note {{
            font-size: 0.88rem;
            color: {subtext_color};
            margin-bottom: 0.7rem;
        }}
        .chart-caption {{
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: {subtext_color};
            margin-bottom: 0.15rem;
        }}
        .chart-label {{
            font-size: 0.98rem;
            font-weight: 600;
            color: {text_color};
            margin-bottom: 0.35rem;
        }}
        .badge {{
            display: inline-block;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .badge-bullish {{
            background: rgba(34,197,94,0.08);
            color: {NEON_GREEN};
            border: none;
            box-shadow: none;
        }}
        .badge-bearish {{
            background: rgba(248,113,113,0.07);
            color: #f97373;
            border: none;
        }}
        .badge-neutral {{
            background: rgba(148,163,184,0.06);
            color: {subtext_color};
            border: none;
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
        .market-card {{
            min-height: 146px;
            padding: 0.55rem 0.7rem 0.45rem 0.7rem;
            border-radius: 0.95rem;
            border: 1px solid rgba(148,163,184,0.18);
            background: linear-gradient(180deg, rgba(15,23,42,0.18), rgba(15,23,42,0.08));
            margin-bottom: 0.25rem;
            display:flex;
            flex-direction:column;
            justify-content:space-between;
            transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
        }}
        .market-card:hover {{
            transform: translateY(-1px);
            border-color: rgba(57,255,20,0.28);
            box-shadow: 0 8px 18px rgba(0,0,0,0.16);
        }}
        .market-card.selected {{
            border-color: rgba(57,255,20,0.85);
            box-shadow: 0 0 0 1px rgba(57,255,20,0.14), 0 10px 22px rgba(0,0,0,0.18);
        }}
        .ticker-card-topline {{
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:0.6rem;
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
            font-size:1.22rem;
            font-weight:700;
            color:{text_color};
            line-height:1.05;
            margin-bottom:0.08rem;
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
        .ticker-subline {{
            font-size:0.76rem;
            color:{subtext_color};
            margin-bottom:0.2rem;
        }}
        .hero-shell {{
            display:block;
            gap:1rem;
            align-items:stretch;
            padding: 0.75rem 0.85rem;
            margin-bottom: 0.55rem;
        }}
        .hero-title {{
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: {text_color};
            margin-bottom: 0.15rem;
        }}
        .hero-subtitle {{
            color: {subtext_color};
            font-size: 0.95rem;
            max-width: 52rem;
            margin-top: 0.3rem;
        }}
        .hero-main {{
            display:flex;
            flex-direction:column;
            justify-content:space-between;
            min-height: 100%;
        }}
        .hero-value {{
            font-size: 2.55rem;
            line-height: 0.95;
            font-weight: 800;
            color: {text_color};
            letter-spacing: -0.03em;
        }}
        .hero-state-row {{
            display:flex;
            gap:0.55rem;
            flex-wrap:wrap;
            margin-top: 0.55rem;
        }}
        .hero-pill {{
            min-width: 6.7rem;
            padding: 0.6rem 0.75rem;
            border-radius: 0.95rem;
            border: 1px solid rgba(148,163,184,0.18);
            background: rgba(15,23,42,0.14);
        }}
        .hero-pill-label {{
            display:block;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {subtext_color};
            margin-bottom: 0.2rem;
        }}
        .hero-pill-value {{
            display:block;
            font-size: 1rem;
            font-weight: 700;
            color: {text_color};
        }}
        .hero-pill-value.pos {{ color: {NEON_GREEN}; }}
        .hero-pill-value.neg {{ color: {RED_FALL}; }}
        .hero-inline-sentiment {{
            display:flex;
            align-items:center;
            gap:0.45rem;
            padding: 0.12rem 0;
        }}
        .hero-inline-label {{
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {subtext_color};
        }}
        .hero-sentiment-text {{
            font-size: 0.9rem;
            font-weight: 600;
            color: {text_color};
        }}
        .insight-row {{
            margin-bottom: 0.75rem;
        }}
        .insight-card {{
            padding: 0.75rem 0.8rem;
            border-radius: 1rem;
            border: 1px solid rgba(148,163,184,0.15);
            background: rgba(15,23,42,0.14);
            height: 100%;
        }}
        .signal-strip {{
            margin-bottom: 0.75rem;
        }}
        .signal-card {{
            padding: 0.7rem 0.8rem;
            border-radius: 0.95rem;
            border: 1px solid rgba(148,163,184,0.16);
            background: rgba(15,23,42,0.12);
            min-height: 126px;
            width: 100%;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }}
        .signal-label {{
            font-size: 0.72rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: {subtext_color};
            margin-bottom: 0.3rem;
        }}
        .signal-text {{
            font-size: 0.9rem;
            line-height: 1.35;
            color: {text_color};
        }}
        .signal-card.good {{ border-color: rgba(148,163,184,0.16); }}
        .signal-card.warning {{ border-color: rgba(148,163,184,0.16); }}
        .signal-card.neutral {{ border-color: rgba(148,163,184,0.16); }}
        .insight-label {{
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {subtext_color};
            margin-bottom: 0.35rem;
        }}
        .insight-value {{
            font-size: 1.25rem;
            font-weight: 700;
            color: {text_color};
            line-height: 1.1;
            margin-bottom: 0.25rem;
        }}
        .insight-note {{
            font-size: 0.86rem;
            color: {subtext_color};
        }}
        .workspace-side {{
            display:flex;
            flex-direction:column;
            gap:0.7rem;
        }}
        .rail-brand {{
            display:flex;
            align-items:center;
            gap:0.7rem;
            margin-bottom:0.9rem;
        }}
        .rail-logo {{
            width:38px;
            height:38px;
            border-radius:12px;
            display:flex;
            align-items:center;
            justify-content:center;
            background: linear-gradient(135deg, rgba(57,255,20,0.22), rgba(15,23,42,0.95));
            color:{NEON_GREEN};
            border: 1px solid rgba(57,255,20,0.28);
            font-weight:800;
            letter-spacing:0.04em;
        }}
        .rail-title {{
            font-size:1rem;
            font-weight:700;
            color:{text_color};
            line-height:1.15;
        }}
        .rail-subtitle {{
            font-size:0.8rem;
            color:{subtext_color};
            margin-top:0.1rem;
        }}
        .rail-section {{
            margin-top:0.85rem;
            padding-top:0.7rem;
            border-top:1px solid rgba(148,163,184,0.12);
        }}
        .rail-label {{
            font-size:0.72rem;
            letter-spacing:0.12em;
            text-transform:uppercase;
            color:{subtext_color};
            margin-bottom:0.45rem;
        }}
        .rail-status-card {{
            padding:0.7rem 0.75rem;
            border-radius:0.9rem;
            background: rgba(15,23,42,0.12);
            border:1px solid rgba(148,163,184,0.14);
            margin-bottom:0.55rem;
        }}
        .rail-status-row {{
            display:flex;
            justify-content:space-between;
            gap:0.6rem;
            align-items:center;
            font-size:0.85rem;
        }}
        .rail-status-name {{ color:{subtext_color}; }}
        .rail-status-value {{ color:{text_color}; font-weight:600; text-align:right; }}
        .rail-status-value.good {{ color:{NEON_GREEN}; }}
        .rail-status-value.bad {{ color:{RED_FALL}; }}
        .rail-chip {{
            display:inline-flex;
            align-items:center;
            justify-content:center;
            min-width:2.8rem;
            padding:0.28rem 0.5rem;
            border-radius:999px;
            font-size:0.78rem;
            font-weight:700;
            border:1px solid rgba(148,163,184,0.16);
            background: rgba(15,23,42,0.12);
            color:{text_color};
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
    latest_price = latest.get("close_price", float("nan"))
    latest_return = latest.get("daily_return", float("nan"))
    latest_return_pct = latest_return * 100 if not pd.isna(latest_return) else float("nan")
    latest_articles = int(latest.get("article_count", 0))
    latest_corr = latest.get("rolling_corr_7d", float("nan"))
    price_change_class = "pos" if not pd.isna(latest_return_pct) and latest_return_pct >= 0 else "neg"
    price_move_label = "Uptrend" if not pd.isna(latest_return_pct) and latest_return_pct >= 0 else "Downtrend"
    corr_label, corr_note = build_signal_text(
        latest_corr,
        "Correlation",
        "News and price are moving together",
        "News and price are diverging",
    )
    sentiment_label_text, sentiment_note = build_signal_text(
        latest_sentiment,
        "Sentiment",
        "Positive tone dominates",
        "Negative tone dominates",
    )
    rule_signals = build_actionable_signals(df)
    total_articles_loaded = int(df["article_count"].fillna(0).sum()) if "article_count" in df.columns else 0
    scored_headlines = int(df["avg_sentiment"].notna().sum()) if "avg_sentiment" in df.columns else 0
    price_feed_active = bool(df["close_price"].notna().any()) if "close_price" in df.columns else False

    with status_slot.container():
        st.markdown('<div class="rail-section">', unsafe_allow_html=True)
        st.markdown('<div class="rail-label">Data quality</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="rail-status-card">
                <div class="rail-status-row">
                    <span class="rail-status-name">API</span>
                    <span class="rail-status-value good">Connected</span>
                </div>
                <div class="rail-status-row">
                    <span class="rail-status-name">Articles loaded</span>
                    <span class="rail-status-value">{total_articles_loaded}</span>
                </div>
                <div class="rail-status-row">
                    <span class="rail-status-name">Scored headlines</span>
                    <span class="rail-status-value">{scored_headlines}</span>
                </div>
                <div class="rail-status-row">
                    <span class="rail-status-name">Price feed</span>
                    <span class="rail-status-value {'good' if price_feed_active else 'bad'}">{ 'Active' if price_feed_active else 'Inactive' }</span>
                </div>
                <div class="rail-status-row">
                    <span class="rail-status-name">Last refresh</span>
                    <span class="rail-status-value">{datetime.now().strftime('%I:%M %p')}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # --------------- HEADER ---------------

    st.markdown(
        f"""
        <div class="glass-card hero-shell">
            <div class="hero-main">
                <div>
                    <div class="hero-title">Selected ticker</div>
                    <div class="hero-value">{ticker}</div>
                    <div class="hero-subtitle">News sentiment and market behavior across {days} days. Use the workspace below to inspect price action, tone, volume, and correlation without flattening the hierarchy.</div>
                </div>
                <div class="hero-state-row">
                    <div class="hero-pill">
                        <span class="hero-pill-label">Price</span>
                        <span class="hero-pill-value">{latest_price:.2f}</span>
                    </div>
                    <div class="hero-pill">
                        <span class="hero-pill-label">Return</span>
                        <span class="hero-pill-value {price_change_class}">{latest_return_pct:+.2f}%</span>
                    </div>
                    <div class="hero-pill">
                        <span class="hero-pill-label">Sentiment</span>
                        <span class="hero-pill-value">{sentiment_tag}</span>
                    </div>
                    <div class="hero-pill">
                        <span class="hero-pill-label">Articles</span>
                        <span class="hero-pill-value">{latest_articles}</span>
                    </div>
                    <div class="hero-pill">
                        <span class="hero-pill-label">Date range</span>
                        <span class="hero-pill-value">{from_date} to {to_date}</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------- TICKER WATCHLIST ---------------

    render_ticker_watchlist(tickers, days=days, dark_mode=dark_mode, selected_ticker=ticker)

    # --------------- ACTIONABLE SIGNALS ---------------

    st.markdown('<div class="signal-strip">', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)

    signal_columns = [s1, s2, s3]
    for idx, signal in enumerate(rule_signals):
        with signal_columns[idx]:
            st.markdown(
                f"""
                <div class="signal-card {signal['tone']}">
                    <div class="signal-label">{signal['label']}</div>
                    <div class="signal-text">{signal['text']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)

    # --------------- CHART WORKSPACE ---------------

    st.markdown('<div class="glass-card card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Chart workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">One dominant market chart on the left with supporting context stacked on the right.</div>', unsafe_allow_html=True)

    workspace_left, workspace_right = st.columns([1.75, 1.0])

    with workspace_left:
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-caption">Latest movement</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-label">Price action</div>', unsafe_allow_html=True)
        fig_price = make_price_chart(df, dark_mode, height=430)
        st.plotly_chart(fig_price, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with workspace_right:
        st.markdown('<div class="workspace-side">', unsafe_allow_html=True)

        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-caption">Sentiment trend</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-label">Tone regime</div>', unsafe_allow_html=True)
        fig_sent = make_sentiment_chart(df, dark_mode, height=210)
        st.plotly_chart(fig_sent, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-caption">News flow</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-label">Article volume</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note" style="margin-bottom:0.45rem;">How much reporting is feeding the signal set.</div>', unsafe_allow_html=True)
        fig_count = make_volume_bars(df, dark_mode, height=165)
        st.plotly_chart(fig_count, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-caption">Signal strength</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-label">7d rolling correlation</div>', unsafe_allow_html=True)
        fig_corr = make_correlation_chart(df, dark_mode, height=165)
        st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # --------------- RAW DATA ---------------

    with st.expander("Raw data", expanded=False):
        st.markdown('<div class="glass-card card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Raw daily metrics</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
