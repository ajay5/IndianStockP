# Architecture

## Runtime
Browser → Streamlit → service layer → providers → feature engineering → ensemble inference → UI. PostgreSQL stores normalized history, news, features, predictions, portfolios and alerts. Redis is optional for shared caching. A scheduled worker should ingest data and evaluate alerts outside the request path.

## APIs for scale-out FastAPI service
- `GET /v1/markets/overview`
- `GET /v1/stocks?exchange=&sector=&query=`
- `GET /v1/stocks/{symbol}/history?interval=&period=`
- `GET /v1/stocks/{symbol}/fundamentals`
- `GET /v1/stocks/{symbol}/news`
- `POST /v1/predictions` with symbol, horizon and model_version
- `GET/POST /v1/portfolios`
- `GET/POST/DELETE /v1/alerts`

Use OAuth/OIDC, per-user authorization, rate limits, request IDs, audit logs, schema validation and encrypted secrets.

## ML pipeline
1. Ingest corporate-action-adjusted OHLCV using point-in-time timestamps.
2. Deduplicate and validate missing values, splits and stale quotes.
3. Build lagged technical, market, sector, fundamental and sentiment features.
4. Define horizon return labels and neutral band after transaction-cost assumptions.
5. Use expanding-window validation. Never random-split time series.
6. Calibrate class probabilities, compare with a naive benchmark and report coverage.
7. Register model, feature schema, training range and metrics.
8. Monitor drift, calibration, provider failures and prediction decay.

The included MVP uses Random Forest and HistGradientBoosting for fast local inference. Add XGBoost/CatBoost only after this baseline. LSTM/GRU/Transformer models require separate sequence datasets, walk-forward validation, model registry and enough data; they should not be mixed in merely for marketing.

## Deployment
Dev: Docker Compose. Production: container registry → Azure Container Apps/AWS ECS or Kubernetes, managed PostgreSQL, managed Redis, Key Vault/Secrets Manager, job scheduler, observability, WAF and CI/CD. For mobile, build a separate client against FastAPI; Streamlit is the analyst web UI.
