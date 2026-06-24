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

# --- equity progression -----------------------------------------------------------
st.subheader("Equity progression")
prog = load_progression()
if not prog.empty:
    st.line_chart(prog.set_index("timestamp")["equity"])
else:
    st.write("No history logged yet — the scheduled trader appends to "
             f"`{LOG_PATH}` each run.")

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
    from src.live.signals import desired_holdings
    from src.universe import resolve

    with st.spinner("Fetching data and running the strategy..."):
        holdings, _ = desired_holdings(resolve(sector), strategy)
    st.success(f"Long: {sorted(holdings) or '(nothing)'}")
