# Going live (paper) + dashboard + landing page

Everything here is **paper trading only** — fake money, real prices. There is no
demonstrated edge; this exists to track the system live and as a portfolio piece.
**Never commit your keys.** `.env` and `.streamlit/secrets.toml` are git-ignored.

## 1. Get Alpaca paper keys (2 min)
1. Sign up / log in at https://app.alpaca.markets/
2. Switch to **Paper Trading** (toggle, top-left).
3. **API Keys → Generate** → copy the Key ID and Secret (the secret shows once).

## 2. Run it locally
```bash
cd trading-strategy-repo
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # then paste your PAPER keys into .env

# SAFE first step — no keys used, nothing submitted, just shows intended orders:
python -m src.live.trade --dry-run

# Live paper trading (reads .env, submits paper orders, logs equity):
python -m src.live.trade

# The dashboard:
streamlit run streamlit_app.py
```

## 3. Deploy the dashboard — Streamlit Community Cloud (free public URL)
1. Go to https://share.streamlit.io → **New app** → sign in with GitHub.
2. Repo `aitiern/equity-swing-backtest`, branch `main`, main file `streamlit_app.py`.
3. **Advanced → Secrets**, paste (this is Streamlit's secret store, NOT the repo):
   ```toml
   ALPACA_API_KEY = "your_paper_key_id"
   ALPACA_SECRET_KEY = "your_paper_secret"
   ALPACA_PAPER = "true"
   TRADE_SECTOR = "tech"
   TRADE_STRATEGY = "donchian"
   ```
4. **Deploy.** You'll get a public URL — put it in `docs/index.html` (the dashboard button).

## 4. Automate the trader — GitHub Actions (runs daily)
The workflow `.github/workflows/paper-trade.yml` runs the trader after the US close
and commits the equity log the dashboard reads.
1. Repo **Settings → Secrets and variables → Actions → New repository secret**:
   add `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` (paper keys).
2. Make sure the workflow file is present on GitHub (see note below), then check the
   **Actions** tab. Trigger once manually with **Run workflow** to verify.

> **Note:** the connected token can't push workflow files, so add
> `.github/workflows/paper-trade.yml` and `ci.yml` via the GitHub web UI
> (**Add file → Upload files**), or re-scope a token with `workflow` permission.

## 5. Publish the landing page — GitHub Pages
1. Repo **Settings → Pages**.
2. **Source: Deploy from a branch**, branch `main`, folder **/docs**. Save.
3. Your site goes live at `https://aitiern.github.io/equity-swing-backtest/`.
4. Edit the dashboard link in `docs/index.html` to your Streamlit URL.

## Safety recap
- Paper only; the code refuses to run if `ALPACA_PAPER=false`.
- Keys live in `.env` (local), Streamlit Secrets (dashboard), and Actions Secrets
  (scheduler) — never in the repo.
- Always `--dry-run` after changing the strategy or universe.
