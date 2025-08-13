from __future__ import annotations

from pathlib import Path
import datetime as _dt

import streamlit as st
import streamlit.components.v1 as components


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
st.caption("This dashboard displays the same charts as the generated HTML files.")

eur_html_path = ROOT / "nav_eur.html"
usd_html_path = ROOT / "nav_usd.html"

eur_tab, usd_tab = st.tabs(["EUR", "USD"])

with eur_tab:
    st.subheader("EUR Funds")
    updated = _mtime_str(eur_html_path)
    if updated:
        st.caption(f"Last updated: {updated} (file: `nav_eur.html`)")
    html = _read_text(eur_html_path)
    if html is None:
        st.warning("`nav_eur.html` not found. Run `python fetch_nav.py` to generate it.")
    else:
        components.html(html, height=800, scrolling=True)

with usd_tab:
    st.subheader("USD Funds")
    updated = _mtime_str(usd_html_path)
    if updated:
        st.caption(f"Last updated: {updated} (file: `nav_usd.html`)")
    html = _read_text(usd_html_path)
    if html is None:
        st.warning("`nav_usd.html` not found. Run `python fetch_nav.py` to generate it.")
    else:
        components.html(html, height=800, scrolling=True)


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

