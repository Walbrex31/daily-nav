from __future__ import annotations

from pathlib import Path
import datetime as _dt
from typing import Optional

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


st.set_page_config(page_title="UCITS MMFs Dashboard", layout="wide")

ROOT = Path(__file__).resolve().parent


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _mtime_str(path: Path) -> str | None:
    try:
        ts = path.stat().st_mtime
        dt = _dt.datetime.fromtimestamp(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


st.title("UCITS Money Market Funds – Performance Comparison")
st.caption("Interactive Plotly charts rendered from CSV outputs.")

eur_csv_path = ROOT / "nav_eur.csv"
usd_csv_path = ROOT / "nav_usd.csv"

eur_tab, usd_tab = st.tabs(["EUR", "USD"])

def _load_nav_df(csv_path: Path) -> Optional[pd.DataFrame]:
    try:
        if not csv_path.exists():
            return None
        df = pd.read_csv(csv_path, encoding="utf-8")
        if df.empty:
            return None
        # Expect a column named 'day'
        if "day" not in df.columns:
            return None
        df["day"] = pd.to_datetime(df["day"], errors="coerce")
        df = df.dropna(subset=["day"]).set_index("day")
        # Ensure all value columns are floats
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # Drop columns that are entirely NA
        df = df.dropna(axis=1, how="all")
        # Sort by day
        df = df.sort_index()
        return df
    except Exception:
        return None


def _normalize_col(col: pd.Series) -> pd.Series:
    non_na = col.dropna()
    if non_na.empty:
        return col
    base = float(non_na.iloc[0])
    if base == 0.0:
        return col
    return col / base


def _compute_annualized(col: pd.Series, start_dt: _dt.date) -> Optional[float]:
    non_na = col.dropna()
    if non_na.empty:
        return None
    first_val = float(non_na.iloc[0])
    last_val = float(non_na.iloc[-1])
    if first_val == 0.0:
        return None
    r = last_val / first_val - 1.0
    last_date = non_na.index[-1]
    try:
        days = (last_date.date() - start_dt).days
        if days <= 0:
            return None
        return (1.0 + r) ** (360.0 / float(days)) - 1.0
    except Exception:
        return None


def _build_figure(df: pd.DataFrame, title: str, group_currency: str, start_dt: _dt.date) -> go.Figure:
    # Drop all-NA rows
    df = df.dropna(how="all")
    df_norm = df.apply(_normalize_col, axis=0)

    fig = go.Figure()

    spiko_blue = "#2376FB"
    grey_shades = [
        "#5C6670",
        "#6F7881",
        "#88919A",
        "#9CA5AD",
        "#B0B8BF",
        "#C4CCD2",
        "#D8DFE3",
        "#A7AFB6",
        "#8C949B",
        "#737B82",
    ]

    x_values = [pd.Timestamp(d) for d in df_norm.index]

    for idx, col in enumerate(df_norm.columns):
        series = df_norm[col]
        ann = _compute_annualized(series, start_dt)
        ann_str = f" | <b>Ann: {ann*100:.2f}%</b>" if ann is not None else ""
        is_spiko = isinstance(col, str) and "SPIKO" in col.upper()
        if is_spiko:
            line_color = spiko_blue
            line_width = 2.8
        else:
            base_grey = grey_shades[idx % len(grey_shades)]
            line_color = base_grey
            line_width = 1.9
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=series,
                mode="lines",
                name=f"{col}{ann_str}",
                line=dict(color=line_color, width=line_width),
                connectgaps=True,
            )
        )

    y_axis_label = f"<i>Normalized NAV ({group_currency})</i>"

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.03, xanchor="left", font=dict(size=20)),
        template="plotly_white",
        legend=dict(title_text=""),
        font=dict(family="Rubik, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif", size=13),
        yaxis_title=y_axis_label,
        yaxis_title_font=dict(size=12),
        xaxis_title=None,
        margin=dict(l=60, r=20, t=90, b=50),
    )

    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"])],
        rangeslider=dict(visible=False),
        rangeselector=dict(
            y=1.02,
            yanchor="top",
            x=0,
            xanchor="left",
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(step="all", label="All"),
            ],
        ),
    )

    return fig


with eur_tab:
    st.subheader("EUR Funds")
    updated = _mtime_str(eur_csv_path)
    if updated:
        st.caption(f"Last updated: {updated} (file: `nav_eur.csv`)")
    eur_df = _load_nav_df(eur_csv_path)
    if eur_df is None or eur_df.empty:
        st.warning("`nav_eur.csv` not found or empty. Run `python fetch_nav.py` to generate it.")
    else:
        min_dt = eur_df.index.min()
        start_dt = _dt.date(min_dt.year, 1, 1)
        fig = _build_figure(eur_df, "UCITS EUR MMFs Performance Comparison", "EUR", start_dt)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

with usd_tab:
    st.subheader("USD Funds")
    updated = _mtime_str(usd_csv_path)
    if updated:
        st.caption(f"Last updated: {updated} (file: `nav_usd.csv`)")
    usd_df = _load_nav_df(usd_csv_path)
    if usd_df is None or usd_df.empty:
        st.warning("`nav_usd.csv` not found or empty. Run `python fetch_nav.py` to generate it.")
    else:
        min_dt = usd_df.index.min()
        start_dt = _dt.date(min_dt.year, 1, 1)
        fig = _build_figure(usd_df, "UCITS USD MMFs Performance Comparison", "USD", start_dt)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


with st.expander("Downloads"):
    col1, col2, col3, col4 = st.columns(4)
    for col, rel in zip(
        (col1, col2, col3, col4),
        ("nav_eur.csv", "nav_usd.csv", "metadata_eur.csv", "metadata_usd.csv"),
    ):
        path = ROOT / rel
        try:
            data = path.read_bytes()
        except Exception:
            data = None
        label = f"Download {rel}"
        if data is None:
            col.button(label, disabled=True)
        else:
            col.download_button(label, data=data, file_name=rel)

