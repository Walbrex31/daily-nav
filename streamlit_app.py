from __future__ import annotations
# pyright: reportMissingTypeStubs=false, reportMissingImports=false

from pathlib import Path
import datetime as _dt
from typing import Optional

import streamlit as st
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore


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
st.caption("Money market funds investing exclusively in short-term sovereign debt instruments.")
st.caption("Source: Yahoo Finance, Spiko public API")

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

    # Order traces by decreasing latest normalized NAV (for hover list ordering)
    latest_norm_values: list[tuple[str, float]] = []
    for col in df_norm.columns:
        s = df_norm[col].dropna()
        if s.empty:
            continue
        latest_norm_values.append((str(col), float(s.iloc[-1])))
    sorted_cols = [name for name, _ in sorted(latest_norm_values, key=lambda t: t[1], reverse=True)]

    for idx, col in enumerate(sorted_cols):
        series = df_norm[col]
        ann = _compute_annualized(series, start_dt)
        ann_str = f" | <b>Ann. YTD: {ann*100:.2f}%</b>" if ann is not None else ""
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
                hovertemplate=
                    "Date: %{x|%Y-%m-%d}<br>" +
                    f"{col}" + ": Norm NAV %{y:.6f}<extra></extra>",
            )
        )

    y_axis_label = f"<i>Normalized NAV ({group_currency})</i>"

    fig.update_layout(
        title=dict(text=""),
        template="plotly_white",
        legend=dict(title_text=""),
        font=dict(family="Rubik, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif", size=13),
        yaxis_title=y_axis_label,
        yaxis_title_font=dict(size=12),
        xaxis_title=None,
        margin=dict(l=60, r=20, t=40, b=50),
        hovermode="x unified",
        hoverlabel=dict(namelength=-1),
        spikedistance=-1,
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
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor="#A0A0A0",
    )

    fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor")

    return fig


with eur_tab:
    st.subheader("EUR MMFs")
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
        # Yields table
        def _compute_periodic_yields(df_src: pd.DataFrame) -> pd.DataFrame:
            records: list[dict] = []
            for col in df_src.columns:
                s = df_src[col].dropna()
                if s.empty:
                    continue
                last_date = s.index[-1]
                last_val = float(s.iloc[-1])
                # Previous available NAV
                prev_ann = None
                if len(s) >= 2:
                    prev_date = s.index[-2]
                    prev_val = float(s.iloc[-2])
                    days = max((last_date - prev_date).days, 0)
                    if prev_val != 0.0 and days > 0:
                        r = last_val / prev_val - 1.0
                        prev_ann = (1.0 + r) ** (360.0 / float(days)) - 1.0
                # 7-day
                seven_ann = None
                try:
                    target7 = last_date - _dt.timedelta(days=7)
                    s7 = s.loc[:target7]
                    if not s7.empty:
                        d7 = s7.index[-1]
                        v7 = float(s7.iloc[-1])
                        days7 = max((last_date - d7).days, 0)
                        if v7 != 0.0 and days7 > 0:
                            r7 = last_val / v7 - 1.0
                            seven_ann = (1.0 + r7) ** (360.0 / float(days7)) - 1.0
                except Exception:
                    pass
                # 30-day
                thirty_ann = None
                try:
                    target30 = last_date - _dt.timedelta(days=30)
                    s30 = s.loc[:target30]
                    if not s30.empty:
                        d30 = s30.index[-1]
                        v30 = float(s30.iloc[-1])
                        days30 = max((last_date - d30).days, 0)
                        if v30 != 0.0 and days30 > 0:
                            r30 = last_val / v30 - 1.0
                            thirty_ann = (1.0 + r30) ** (360.0 / float(days30)) - 1.0
                except Exception:
                    pass
                records.append({
                    "Fund": col,
                    "Ann. daily yield": prev_ann,
                    "Ann. 7d yield": seven_ann,
                    "Ann. 30d yield": thirty_ann,
                })
            out = pd.DataFrame.from_records(records)
            if not out.empty:
                out = out.set_index("Fund")
            return out

        eur_yields = _compute_periodic_yields(eur_df)
        if eur_yields is not None and not eur_yields.empty:
            eur_yields = eur_yields.sort_values(by="Ann. 30d yield", ascending=False, na_position="last")
            st.markdown("**Annualized yields**")
            st.dataframe(eur_yields.map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "-"))

with usd_tab:
    st.subheader("USD MMFs")
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
        # Yields table
        def _compute_periodic_yields(df_src: pd.DataFrame) -> pd.DataFrame:
            records: list[dict] = []
            for col in df_src.columns:
                s = df_src[col].dropna()
                if s.empty:
                    continue
                last_date = s.index[-1]
                last_val = float(s.iloc[-1])
                prev_ann = None
                if len(s) >= 2:
                    prev_date = s.index[-2]
                    prev_val = float(s.iloc[-2])
                    days = max((last_date - prev_date).days, 0)
                    if prev_val != 0.0 and days > 0:
                        r = last_val / prev_val - 1.0
                        prev_ann = (1.0 + r) ** (360.0 / float(days)) - 1.0
                seven_ann = None
                try:
                    target7 = last_date - _dt.timedelta(days=7)
                    s7 = s.loc[:target7]
                    if not s7.empty:
                        d7 = s7.index[-1]
                        v7 = float(s7.iloc[-1])
                        days7 = max((last_date - d7).days, 0)
                        if v7 != 0.0 and days7 > 0:
                            r7 = last_val / v7 - 1.0
                            seven_ann = (1.0 + r7) ** (360.0 / float(days7)) - 1.0
                except Exception:
                    pass
                thirty_ann = None
                try:
                    target30 = last_date - _dt.timedelta(days=30)
                    s30 = s.loc[:target30]
                    if not s30.empty:
                        d30 = s30.index[-1]
                        v30 = float(s30.iloc[-1])
                        days30 = max((last_date - d30).days, 0)
                        if v30 != 0.0 and days30 > 0:
                            r30 = last_val / v30 - 1.0
                            thirty_ann = (1.0 + r30) ** (360.0 / float(days30)) - 1.0
                except Exception:
                    pass
                records.append({
                    "Fund": col,
                    "Ann. daily yield": prev_ann,
                    "Ann. 7d yield": seven_ann,
                    "Ann. 30d yield": thirty_ann,
                })
            out = pd.DataFrame.from_records(records)
            if not out.empty:
                out = out.set_index("Fund")
            return out

        usd_yields = _compute_periodic_yields(usd_df)
        if usd_yields is not None and not usd_yields.empty:
            usd_yields = usd_yields.sort_values(by="Ann. 30d yield", ascending=False, na_position="last")
            st.markdown("**Annualized yields**")
            st.dataframe(usd_yields.map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "-"))


with st.expander("Downloads"):
    col1, col2, col3, col4 = st.columns(4)
    for col, rel in zip(
        (col1, col2, col3, col4),
        ("nav_eur.csv", "nav_usd.csv", "metadata_eur.csv", "metadata_usd.csv"),
    ):
        path = ROOT / rel
        data_bytes: Optional[bytes] = None
        try:
            data_bytes = path.read_bytes()
        except Exception:
            pass
        label = f"Download {rel}"
        if data_bytes is None:
            col.button(label, disabled=True)
        else:
            col.download_button(label, data=data_bytes, file_name=rel)

