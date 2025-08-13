#!/usr/bin/env python3
"""
Build EUR and USD fund NAV datasets from competitors.csv and generate charts.

Behavior:
- Reads competitors.csv and splits funds into two groups by Currency: EUR and USD
- For each fund, uses its "Yahoo ticker" to fetch daily Close (NAV) from 2025-01-01 to latest
- Fetches AUM from Yahoo fund metadata for reference

Outputs in the current directory:
- nav_eur.csv and nav_usd.csv: rows = dates, columns = fund Name, values = NAV
- metadata_eur.csv and metadata_usd.csv: fund name, ticker, AUM, inception, YTD return, annualized YTD return
- nav_eur.html and nav_usd.html: Plotly charts of normalized NAVs (base=1 EUR or 1 USD)

Usage:
  python3 fetch_nav.py
"""

from __future__ import annotations

import sys
import argparse
from typing import Dict, Optional, Tuple, List, Iterable
from datetime import date, datetime
import csv
import os
from pathlib import Path
import json
from urllib.request import urlopen, Request
from urllib.parse import urlencode

import yfinance as yf  # type: ignore[import-not-found]
import pandas as pd  # type: ignore


DEFAULT_TICKER = "0P00000CFB.F"


def get_latest_nav(yahoo_ticker: str) -> Tuple[str, float]:
    """Return (date_iso, nav) for the most recent daily close.

    For mutual funds on Yahoo Finance, the daily Close typically reflects the NAV.
    """
    ticker = yf.Ticker(yahoo_ticker)

    for period in ("1mo", "3mo", "6mo"):
        hist = ticker.history(period=period, interval="1d", auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist:
            continue
        close = hist["Close"].dropna()
        if close.empty:
            continue
        last_ts = close.index[-1]
        last_value = float(close.iloc[-1])
        # Just output the calendar date (no timezone/time component)
        return last_ts.date().isoformat(), last_value

    # Fallback: try fast_info if history failed (less reliable for funds)
    fast_info = getattr(ticker, "fast_info", None)
    if fast_info is not None:
        price = getattr(fast_info, "last_price", None)
        if price is None:
            price = getattr(fast_info, "previous_close", None)
        if price is not None:
            return date.today().isoformat(), float(price)

    raise RuntimeError(f"Could not retrieve NAV for ticker: {yahoo_ticker}")


def get_first_nav_on_or_after(yahoo_ticker: str, start_date: date) -> Optional[Tuple[str, float]]:
    """Return (date_iso, nav) for the first available daily close on or after start_date.

    Returns None if no data is available in a reasonable window after start_date.
    """
    ticker = yf.Ticker(yahoo_ticker)
    # Fetch a window from start_date to roughly two weeks later to ensure we pick
    # the first trading day after the requested start.
    # Initial window: from start_date through end of month
    hist = ticker.history(
        start=start_date.isoformat(),
        end=None,
        interval="1d",
        auto_adjust=False,
    )
    if hist is None or hist.empty or "Close" not in hist:
        # Try a broader window (1 month)
        hist = ticker.history(start=start_date.isoformat(), period="1mo", interval="1d", auto_adjust=False)

    if hist is None or hist.empty or "Close" not in hist:
        return None

    close = hist["Close"].dropna()
    if close.empty:
        return None

    first_ts = close.index[0]
    first_value = float(close.iloc[0])
    return first_ts.date().isoformat(), first_value


def compute_ytd_return_from_history(yahoo_ticker: str, year: int) -> Optional[float]:
    """Compute YTD return as (latest_close / first_close_after_year_start - 1).

    Returns None if required data is unavailable.
    """
    start_of_year = date(year, 1, 1)
    first = get_first_nav_on_or_after(yahoo_ticker, start_of_year)
    if not first:
        return None
    latest_date, latest_nav = get_latest_nav(yahoo_ticker)
    _, first_nav = first
    if first_nav == 0:
        return None
    return latest_nav / first_nav - 1.0


def get_fund_metadata(yahoo_ticker: str) -> Dict[str, Optional[str]]:
    """Fetch available fund metadata fields from Yahoo and normalize a bit."""
    ticker = yf.Ticker(yahoo_ticker)
    try:
        info = ticker.get_info()
        if not isinstance(info, dict):
            info = {}
    except Exception:
        info = {}

    def get_number(key: str) -> Optional[float]:
        try:
            val = info.get(key)
            if val is None:
                return None
            return float(val)
        except Exception:
            return None

    def get_int(key: str) -> Optional[int]:
        try:
            val = info.get(key)
            if val is None:
                return None
            return int(val)
        except Exception:
            return None

    inception_ts = get_int("fundInceptionDate")
    inception_iso: Optional[str] = None
    if inception_ts:
        try:
            # Use timezone-aware UTC conversion, compatible across Python versions
            from datetime import timezone

            inception_iso = datetime.fromtimestamp(inception_ts, tz=timezone.utc).date().isoformat()
        except Exception:
            inception_iso = None

    # Prefer longName then shortName
    long_name = info.get("longName") or info.get("shortName")

    # Try multiple possible fields for fund family/manager
    fund_family = info.get("fundFamily") or info.get("fund_fundFamily") or info.get("fundManager")

    # YTD return provided by Yahoo (if present)
    ytd_fields = [
        "ytdReturn",
        "trailingYtdReturn",
        "trailingReturnYtd",
    ]
    ytd_from_info: Optional[float] = None
    for f in ytd_fields:
        val = get_number(f)
        if val is not None:
            ytd_from_info = val
            break

    metadata: Dict[str, Optional[str]] = {
        "name": str(long_name) if long_name is not None else None,
        "asset_manager": str(fund_family) if fund_family is not None else None,
        "category": info.get("category"),
        "legal_type": info.get("legalType"),
        "inception_date": inception_iso,
        "aum_total_assets": str(get_number("totalAssets")) if get_number("totalAssets") is not None else None,
        "expense_ratio": str(get_number("annualReportExpenseRatio")) if get_number("annualReportExpenseRatio") is not None else None,
        "yield": str(get_number("fundYield") or get_number("yield")) if (get_number("fundYield") or get_number("yield")) is not None else None,
        "ytd_return_from_info": str(ytd_from_info) if ytd_from_info is not None else None,
    }

    return metadata


def _normalize_header_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def read_competitors(filepath: Path) -> List[dict]:
    rows: List[dict] = []
    with filepath.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            # Normalize fieldnames to ease access
            normalized = [_normalize_header_name(h) if h is not None else "" for h in reader.fieldnames]
            reader.fieldnames = normalized
        for row in reader:
            if not row:
                continue
            # Skip completely empty lines
            if all((v or "").strip() == "" for v in row.values()):
                continue
            rows.append(row)
    return rows


def fetch_history_close_series(yahoo_ticker: str, start_dt: date) -> Optional[pd.Series]:
    try:
        t = yf.Ticker(yahoo_ticker)
        hist = t.history(start=start_dt.isoformat(), end=None, interval="1d", auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist:
            return None
        close = hist["Close"].dropna()
        if close.empty:
            return None
        # Ensure DatetimeIndex normalized to date (no time component)
        close.index = pd.to_datetime(close.index).tz_localize(None)
        # Convert index to date for nicer CSV display
        close.index = close.index.date
        close.name = yahoo_ticker
        return close.astype(float)
    except Exception:
        return None


def build_group_dataframe(funds: Iterable[dict], start_dt: date) -> tuple[pd.DataFrame, list[dict]]:
    series_list: List[pd.Series] = []
    metadata_rows: list[dict] = []

    for fund in funds:
        name = (fund.get("name") or "").strip()
        ticker = (fund.get("yahoo_ticker") or fund.get("yahoo_ticker_") or fund.get("yahoo_ticker__") or fund.get("yahoo_ticker___") or fund.get("yahoo_ticker____") or fund.get("yahoo ticker") or fund.get("yahoo_ticker\ufeff") or "").strip()
        if not name or not ticker:
            continue

        s = fetch_history_close_series(ticker, start_dt)
        if s is not None:
            s = s.rename(name)
            series_list.append(s)

        # Fetch AUM via metadata
        meta = get_fund_metadata(ticker)
        raw_aum_opt = meta.get("aum_total_assets")
        aum_val: Optional[float]
        if raw_aum_opt is not None:
            try:
                aum_val = float(raw_aum_opt)
            except Exception:
                aum_val = None
        else:
            aum_val = None

        # Compute YTD and annualized YTD from available series
        ytd_return_val: Optional[float] = None
        annualized_ytd_val: Optional[float] = None
        if s is not None:
            non_na = s.dropna()
            if not non_na.empty:
                first_val = float(non_na.iloc[0])
                last_val = float(non_na.iloc[-1])
                if first_val != 0.0:
                    ytd_return_val = last_val / first_val - 1.0
                    latest_dt = non_na.index[-1]
                    try:
                        days_elapsed = (latest_dt - start_dt).days
                        if days_elapsed > 0:
                            annualized_ytd_val = (1.0 + ytd_return_val) ** (360.0 / float(days_elapsed)) - 1.0
                    except Exception:
                        pass

        metadata_rows.append(
            {
                "fund_name": name,
                "yahoo_ticker": ticker,
                "aum_total_assets": aum_val,
                "inception_date": meta.get("inception_date"),
                "ytd_return": ytd_return_val,
                "annualized_ytd_return": annualized_ytd_val,
            }
        )

    if not series_list:
        return pd.DataFrame(), metadata_rows

    # Outer-join on dates to get union of all dates
    df = pd.concat(series_list, axis=1, join="outer").sort_index()
    # Keep only rows on/after start date (defensive)
    df = df.loc[df.index >= start_dt]
    return df, metadata_rows


def write_nav_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        # Write only header row with day if nothing
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["day"])  # no fund columns
        return
    df_to_write = df.copy()
    # Keep date index as-is; set index label for CSV
    df_to_write.to_csv(path, encoding="utf-8", index_label="day")


def write_metadata_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            simple_writer = csv.writer(f)
            simple_writer.writerow(["fund_name", "yahoo_ticker", "aum_total_assets", "inception_date", "ytd_return", "annualized_ytd_return"])
        return
    fieldnames = ["fund_name", "yahoo_ticker", "aum_total_assets", "inception_date", "ytd_return", "annualized_ytd_return"]
    with path.open("w", newline="", encoding="utf-8") as f:
        dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
        dict_writer.writeheader()
        for r in rows:
            dict_writer.writerow(r)


def write_plotly_html_normalized(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
    group_currency: str,
    start_dt: date,
) -> None:
    try:
        import plotly.graph_objects as go  # type: ignore[import-not-found]
        from plotly.offline import plot  # type: ignore[import-not-found]
    except Exception:
        sys.stderr.write("Plotly not installed; skipping HTML chart generation.\n")
        return

    if df.empty:
        # Write a minimal HTML mentioning no data
        html = f"<html><body><h3>{title}</h3><p>No data available.</p></body></html>"
        output_path.write_text(html, encoding="utf-8")
        return

    # Drop rows where all values are NA to avoid empty-space-only dates
    df = df.dropna(how="all")

    # Normalize each column to its first non-NA value
    def normalize_col(col: pd.Series) -> pd.Series:
        non_na = col.dropna()
        if non_na.empty:
            return col
        base = float(non_na.iloc[0])
        if base == 0.0:
            return col
        return col / base

    df_norm = df.apply(normalize_col, axis=0)

    fig = go.Figure()

    # Define colors: Spiko in brand blue, others in grey shades
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

    def hex_to_rgba(hex_color: str, alpha: float) -> str:
        hex_clean = hex_color.lstrip("#")
        r = int(hex_clean[0:2], 16)
        g = int(hex_clean[2:4], 16)
        b = int(hex_clean[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    x_values = [pd.Timestamp(d) for d in df_norm.index]

    # Compute annualized YTD inline for legend labels
    def compute_annualized(col: pd.Series) -> Optional[float]:
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
            days = (last_date - start_dt).days
            if days <= 0:
                return None
            return (1.0 + r) ** (360.0 / float(days)) - 1.0
        except Exception:
            return None

    y_axis_label = f"<i>Normalized NAV ({group_currency})</i>"

    for idx, col in enumerate(df_norm.columns):
        series = df_norm[col]
        ann = compute_annualized(series)
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

    # Configure layout: title bold and larger and slightly right-aligned, y-axis smaller font (italic via HTML), no legend title
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

    # X-axis: remove weekend gaps, move range selector down, disable rangeslider
    # Also add a proper slider widget that adjusts the end of the visible range.
    # min_dt/max_dt can be derived from df_norm when needed

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

    # Note: Plotly does not support a single slider with two handles without the mini-chart rangeslider.
    # As requested, we remove custom sliders and keep only the range selector buttons.

    plot(fig, filename=str(output_path), auto_open=False, include_plotlyjs="cdn")


def fetch_spiko_series(fund_code: str, start_dt: date, display_name: str) -> Optional[pd.Series]:
    """Fetch NAV series from Spiko public API for given fund code (e.g., 'EUTBL', 'USTBL').

    Returns a pandas Series indexed by date with float values and column name = display_name.
    """
    try:
        base_url = "https://public-api.spiko.io/v0/net-asset-values/" + fund_code
        params = {"startDay": start_dt.isoformat()}
        url = base_url + "?" + urlencode(params)
        req = Request(url, headers={"User-Agent": "nav-fetcher/1.0"})
        with urlopen(req) as resp:  # noqa: S310 - trusted public API
            data = json.loads(resp.read().decode("utf-8"))
        if not isinstance(data, list):
            return None
        records: list[tuple[date, float]] = []
        for item in data:
            try:
                day_str = item.get("day")
                amount = item.get("amount") or {}
                value_str = amount.get("value")
                if not day_str or value_str is None:
                    continue
                # Accept both 'YYYY-MM-DD' and with time components
                day_only = day_str.split("T")[0]
                y, m, d = [int(x) for x in day_only.split("-")]
                dt = date(y, m, d)
                val = float(value_str)
                records.append((dt, val))
            except Exception:
                continue
        if not records:
            return None
        # Build Series
        records.sort(key=lambda x: x[0])
        idx = [r[0] for r in records]
        vals = [r[1] for r in records]
        s = pd.Series(data=vals, index=idx, name=display_name, dtype=float)
        return s
    except Exception:
        return None


def _extract_latest_navs(df: pd.DataFrame) -> list[dict]:
    """Return list of {name, day_iso, nav} for the last available value of each column."""
    results: list[dict] = []
    if df is None or df.empty:
        return results
    for col in df.columns:
        try:
            series = df[col].dropna()
            if series.empty:
                continue
            last_day = series.index[-1]
            # Ensure ISO date string
            try:
                day_iso = last_day.isoformat()
            except Exception:
                day_iso = str(last_day)
            nav_val = float(series.iloc[-1])
            results.append({"name": str(col), "day_iso": day_iso, "nav": nav_val})
        except Exception:
            continue
    # Sort alphabetically by fund name for stable message
    results.sort(key=lambda r: r["name"].upper())
    return results


def _format_slack_text(eur_df: pd.DataFrame, usd_df: pd.DataFrame) -> str:
    eur_latest = _extract_latest_navs(eur_df)
    usd_latest = _extract_latest_navs(usd_df)
    lines: list[str] = []
    lines.append("*Daily NAV Update*")
    if eur_latest:
        lines.append("\n*EUR funds*:")
        for r in eur_latest:
            lines.append(f"• {r['name']}: {r['nav']:.6f} EUR (as of {r['day_iso']})")
    else:
        lines.append("\n*EUR funds*: no data")
    if usd_latest:
        lines.append("\n*USD funds*:")
        for r in usd_latest:
            lines.append(f"• {r['name']}: {r['nav']:.6f} USD (as of {r['day_iso']})")
    else:
        lines.append("\n*USD funds*: no data")
    return "\n".join(lines)


def _post_to_slack(webhook_url: str, text: str) -> None:
    """Post a simple text payload to a Slack Incoming Webhook."""
    try:
        payload = json.dumps({"text": text}).encode("utf-8")
        req = Request(webhook_url, data=payload, headers={"Content-Type": "application/json", "User-Agent": "nav-fetcher/1.0"})
        with urlopen(req):  # noqa: S310 - posting to configured webhook URL
            pass
    except Exception as exc:
        sys.stderr.write(f"Failed to post Slack message: {exc}\n")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NAV datasets and optionally notify Slack")
    parser.add_argument("--slack", action="store_true", help="Send Slack message with latest NAVs")
    parser.add_argument(
        "--slack-webhook",
        default=os.environ.get("SLACK_WEBHOOK_URL"),
        help="Slack Incoming Webhook URL (or set SLACK_WEBHOOK_URL env var)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    start_year = 2025
    start_date_obj = date(start_year, 1, 1)

    cwd = Path(os.getcwd())
    competitors_path = cwd / "competitors.csv"
    if not competitors_path.exists():
        sys.stderr.write(f"competitors.csv not found at: {competitors_path}\n")
        return 1

    rows = read_competitors(competitors_path)
    # Group by currency
    eur_funds = [r for r in rows if (r.get("currency") or "").strip().upper() == "EUR"]
    usd_funds = [r for r in rows if (r.get("currency") or "").strip().upper() == "USD"]

    # Build dataframes
    eur_df, eur_meta = build_group_dataframe(eur_funds, start_date_obj)
    usd_df, usd_meta = build_group_dataframe(usd_funds, start_date_obj)

    # Append Spiko funds via public API (competitors.csv may exclude them due to Yahoo inaccuracies)
    spiko_eur_name = "SPIKO EU T-BILLS MONEY MARKET FUND"
    spiko_usd_name = "SPIKO US T-BILLS MONEY MARKET FUND"

    spiko_eur = fetch_spiko_series("EUTBL", start_date_obj, spiko_eur_name)
    if spiko_eur is not None:
        eur_df = (pd.concat([eur_df, spiko_eur], axis=1, join="outer") if not eur_df.empty else spiko_eur.to_frame()).sort_index()
        # Compute YTD and annualized
        ytd_return_val: Optional[float] = None
        annualized_ytd_val: Optional[float] = None
        non_na = spiko_eur.dropna()
        if not non_na.empty:
            first_val = float(non_na.iloc[0])
            last_val = float(non_na.iloc[-1])
            if first_val != 0.0:
                ytd_return_val = last_val / first_val - 1.0
                latest_dt = non_na.index[-1]
                days_elapsed = (latest_dt - start_date_obj).days
                if days_elapsed > 0:
                    annualized_ytd_val = (1.0 + ytd_return_val) ** (360.0 / float(days_elapsed)) - 1.0
        eur_meta.append(
            {
                "fund_name": spiko_eur_name,
                "yahoo_ticker": "SPIKO:EUTBL",
                "aum_total_assets": None,
                "inception_date": None,
                "ytd_return": ytd_return_val,
                "annualized_ytd_return": annualized_ytd_val,
            }
        )

    spiko_usd = fetch_spiko_series("USTBL", start_date_obj, spiko_usd_name)
    if spiko_usd is not None:
        usd_df = (pd.concat([usd_df, spiko_usd], axis=1, join="outer") if not usd_df.empty else spiko_usd.to_frame()).sort_index()
        # Compute YTD and annualized
        ytd_return_val2: Optional[float] = None
        annualized_ytd_val2: Optional[float] = None
        non_na2 = spiko_usd.dropna()
        if not non_na2.empty:
            first_val2 = float(non_na2.iloc[0])
            last_val2 = float(non_na2.iloc[-1])
            if first_val2 != 0.0:
                ytd_return_val2 = last_val2 / first_val2 - 1.0
                latest_dt2 = non_na2.index[-1]
                days_elapsed2 = (latest_dt2 - start_date_obj).days
                if days_elapsed2 > 0:
                    annualized_ytd_val2 = (1.0 + ytd_return_val2) ** (360.0 / float(days_elapsed2)) - 1.0
        usd_meta.append(
            {
                "fund_name": spiko_usd_name,
                "yahoo_ticker": "SPIKO:USTBL",
                "aum_total_assets": None,
                "inception_date": None,
                "ytd_return": ytd_return_val2,
                "annualized_ytd_return": annualized_ytd_val2,
            }
        )

    # Write CSV outputs
    write_nav_csv(eur_df, cwd / "nav_eur.csv")
    write_nav_csv(usd_df, cwd / "nav_usd.csv")

    # Write AUM/metadata per group for reference
    write_metadata_csv(eur_meta, cwd / "metadata_eur.csv")
    write_metadata_csv(usd_meta, cwd / "metadata_usd.csv")

    # Plotly HTML outputs (normalized) with improved design
    write_plotly_html_normalized(
        eur_df,
        title="UCITS EUR MMFs Performance Comparison",
        output_path=cwd / "nav_eur.html",
        group_currency="EUR",
        start_dt=start_date_obj,
    )
    write_plotly_html_normalized(
        usd_df,
        title="UCITS USD MMFs Performance Comparison",
        output_path=cwd / "nav_usd.html",
        group_currency="USD",
        start_dt=start_date_obj,
    )

    # Optionally send Slack notification
    if args.slack:
        webhook = args.slack_webhook
        if not webhook:
            sys.stderr.write("--slack set but no webhook configured (use --slack-webhook or SLACK_WEBHOOK_URL)\n")
        else:
            text = _format_slack_text(eur_df, usd_df)
            _post_to_slack(webhook, text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

