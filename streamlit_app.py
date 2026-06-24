"""Live paper-trading dashboard (Streamlit).

Run locally:   streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud, main file = streamlit_app.py, with
               ALPACA_API_KEY / ALPACA_SECRET_KEY set in the app's Secrets.

Reads the Alpaca PAPER account for current equity/positions and the committed
tracking/equity_log.csv for progression over time. Strictly read-only — it never
places orders (that is the scheduled trader's job).
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

# Bridge Streamlit secrets -> environment so the shared config loader works.
for _k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_PAPER", "TRADE_SECTOR", "TRADE_STRATEGY"):
    if _k in st.secrets:
        os.environ[_k] = str(st.secrets[_k])

st.set_page_config(page_title="Paper-Trading Tracker", page_icon="📈", layout="wide")
st.title("📈 Paper-Trading Tracker")
st.caption("Live Alpaca **paper** account — fake money, real prices. Tracking only; "
           "trades are placed by the scheduled job, not here.")

LOG_PATH = "tracking/equity_log.csv"


@st.cache_data(ttl=300)
def load_progression() -> pd.DataFrame:
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()
    df = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    return df.sort_values("timestamp")


@st.cache_data(ttl=900)
def load_spy(start_iso: str, end_iso: str):
    """SPY closes over the tracked window (cached). Returns None on any failure."""
    from collections import deque

    from src.data import DataHandler

    try:
        dh = DataHandler.from_yfinance(["SPY"], start_iso, end_iso, deque(), cache_dir=None)
        spy = dh.close_frame()["SPY"]
        return spy if not spy.empty else None
    except Exception:
        return None


def get_broker():
    from src.live.broker import AlpacaBroker

    return AlpacaBroker()


# --- account panel ----------------------------------------------------------------
broker = None
try:
    broker = get_broker()
    snap = broker.account_snapshot()
    pnl_today = snap["equity"] - snap["last_equity"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Equity", f"${snap['equity']:,.2f}", f"{pnl_today:+,.2f} today")
    c2.metric("Cash", f"${snap['cash']:,.2f}")
    c3.metric("Buying power", f"${snap['buying_power']:,.2f}")
except Exception as exc:  # no keys yet, or alpaca-py missing
    st.info(
        "No live account connected yet. Set **ALPACA_API_KEY** and **ALPACA_SECRET_KEY** "
        "in this app's Secrets (or a local `.env`) to see your paper account. "
        f"\n\n_Details: {exc}_"
    )

# --- equity progression vs SPY ----------------------------------------------------
st.subheader("Equity progression vs SPY")
prog = load_progression()
if not prog.empty:
    from src.live.dashboard_utils import (
        BENCHMARK_LABEL,
        STRATEGY_LABEL,
        combine_equity_and_benchmark,
        normalize_benchmark,
    )

    start_equity = float(prog["equity"].iloc[0])
    start_iso = pd.to_datetime(prog["timestamp"].min(), utc=True).date().isoformat()
    end_iso = (pd.to_datetime(prog["timestamp"].max(), utc=True) + pd.Timedelta(days=1)).date().isoformat()

    spy = load_spy(start_iso, end_iso)
    bench = normalize_benchmark(spy, start_equity) if spy is not None else None
    combined = combine_equity_and_benchmark(prog, bench)
    st.line_chart(combined)

    # Contextual caption: who's ahead since inception?
    if BENCHMARK_LABEL in combined.columns:
        s_ret = combined[STRATEGY_LABEL].dropna().iloc[-1] / start_equity - 1
        b_ret = combined[BENCHMARK_LABEL].dropna().iloc[-1] / start_equity - 1
        verdict = "ahead of" if s_ret > b_ret else "behind"
        st.caption(f"Since inception: strategy **{s_ret * 100:+.1f}%** vs SPY **{b_ret * 100:+.1f}%** "
                   f"— currently **{verdict}** buy-and-hold.")
    else:
        st.caption("SPY benchmark unavailable right now; showing strategy equity only.")
else:
    st.write("No history logged yet — the scheduled trader appends to "
             f"`{LOG_PATH}` each run, and the SPY line will appear alongside it.")

# --- positions --------------------------------------------------------------------
if broker is not None:
    st.subheader("Open positions")
    try:
        details = broker.position_details()
        if details:
            st.dataframe(pd.DataFrame(details), use_container_width=True)
        else:
            st.write("Flat — no open positions.")
    except Exception as exc:
        st.warning(f"Could not load positions: {exc}")

# --- today's signals --------------------------------------------------------------
st.subheader("What the strategy wants today")
sector = os.getenv("TRADE_SECTOR", "tech")
strategy = os.getenv("TRADE_STRATEGY", "donchian")
st.caption(f"Universe: **{sector}**  ·  strategy: **{strategy}**")
if st.button("Compute current signals"):
    from src.live.signals import desired_holdings, fetch_frames
    from src.universe import resolve

    with st.spinner("Fetching data and running the strategy..."):
        symbols = resolve(sector)
        frames = fetch_frames(symbols)
        holdings, _ = desired_holdings(symbols, strategy, frames)
    st.success(f"Long: {sorted(holdings) or '(nothing)'}")
