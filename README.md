# Bharat Market AI Lab

A runnable Streamlit research MVP for Indian equities with NSE/BSE and sector dropdowns, Yahoo Finance history, technical indicators, latest-news aggregation, headline sentiment, time-aware ML signals, explainability, sector comparison, portfolio and alert scaffolding.

## Important limitations
This is educational software, not investment advice or an execution system. Signals are probabilistic and may be wrong. Yahoo Finance/yfinance is suitable for research/personal use; review provider terms and license a commercial market-data feed before production. News links belong to their publishers. Do not scrape social platforms; use official/licensed APIs.

## Run
### Windows (PowerShell)
```powershell
$Python = "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe"
& $Python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The `$Python` path matches the detected Python installation on this machine. If Python is installed elsewhere, update that path to the full path of its `python.exe`. Calling the executable directly avoids relying on the `python` or `py` commands being available on `PATH`.

PowerShell activation is optional. If script execution is restricted on your system, do not run `Activate.ps1` or change the system execution policy; use the virtual-environment Python commands above directly instead.

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```
Open http://localhost:8501. Optional `NEWS_API_KEY` enables NewsAPI; otherwise Google News RSS is used.

## Docker
```bash
docker compose up --build
```

## What is implemented
- NSE/BSE exchange dropdown, 34 sector groups and curated mapping
- Index dashboard, candlestick chart, snapshot fundamentals
- RSI, MACD, EMA 20/50/200, SMA, Bollinger Bands, VWAP, ATR, ADX and stochastic oscillator
- RSS/NewsAPI headlines and VADER baseline sentiment
- Time-ordered Random Forest + histogram gradient boosting ensemble
- Buy/Hold/Sell-style research signal with confidence, probabilities, risk, expected return caveat and reasons
- PostgreSQL schema, Docker files, deployment and ML architecture notes

## Production hardening checklist
License data/news, obtain exchange permissions, add a symbol master, corporate actions, trading calendar, survivorship-bias-free data, point-in-time fundamentals, walk-forward backtests, transaction costs, probability calibration, drift monitoring, RBAC, audit logs, encrypted secrets, privacy policy, uptime SLOs, security testing and legal/compliance review. Replace VADER with a validated India-finance sentiment model. Social sentiment remains intentionally unimplemented until official API credentials are supplied.

## Intraday note
Yahoo intervals and retention are provider-limited. For reliable intraday production, use a licensed low-latency feed and separate intraday models. The current UI focuses on daily horizons.

## Next-day forecast and automatic retraining
The prediction tab displays the predicted next trading-day close, percentage change, empirical 80% interval, walk-forward MAE, training-row count and timestamp. Every **Run analysis** click refits on all latest completed daily bars. Current headline sentiment and broad-market movement are conservative bounded adjustments because this MVP does not yet have historical point-in-time sentiment.

Self-learning means scheduled retraining and evaluation, not guaranteed accuracy. A production job should retain every forecast, match it to the realized next close, monitor MAE, directional accuracy and drift, and promote a challenger only after robust out-of-sample improvement.
