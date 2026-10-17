import numpy as np
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["RSI", "MACD", "MACD_SIGNAL", "EMA20", "EMA50", "EMA200", "ATR", "ADX", "STOCH_RSI", "VOL20", "VOL_Z", "RET1"]

def forecast_next_close(df, news_score=0.0, market_move_pct=0.0):
    x = df.copy()
    x["NEXT_LOG_RETURN"] = np.log(x["Close"].shift(-1) / x["Close"])
    data = x[FEATURES + ["NEXT_LOG_RETURN"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 220:
        raise ValueError("Select at least 2 years of history; about 220 valid training rows are required.")
    X, y = data[FEATURES], data["NEXT_LOG_RETURN"]
    models = [
        RandomForestRegressor(n_estimators=350, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1),
        make_pipeline(StandardScaler(), HistGradientBoostingRegressor(max_iter=220, max_depth=5, learning_rate=0.04, l2_regularization=1.0, random_state=42)),
    ]
    splitter = TimeSeriesSplit(n_splits=5, gap=1)
    fold_mae, residuals = [], []
    for train_idx, test_idx in splitter.split(X):
        fold_predictions = []
        for model in models:
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            fold_predictions.append(model.predict(X.iloc[test_idx]))
        pred = np.mean(fold_predictions, axis=0)
        actual = y.iloc[test_idx].to_numpy()
        fold_mae.append(mean_absolute_error(actual, pred))
        residuals.extend((actual - pred).tolist())
    latest = x[FEATURES].replace([np.inf, -np.inf], np.nan).dropna().iloc[[-1]]
    forecasts = []
    for model in models:
        model.fit(X, y)
        forecasts.append(float(model.predict(latest)[0]))
    technical_return = float(np.mean(forecasts))
    news_adjustment = float(np.clip(news_score, -1, 1)) * 0.0015
    market_adjustment = float(np.clip(market_move_pct / 100.0, -0.03, 0.03)) * 0.15
    predicted_log_return = technical_return + news_adjustment + market_adjustment
    current_close = float(x["Close"].dropna().iloc[-1])
    predicted_close = current_close * float(np.exp(predicted_log_return))
    residuals = np.asarray(residuals)
    q_low, q_high = np.quantile(residuals, [0.10, 0.90])
    return {
        "current_close": round(current_close, 2), "predicted_close": round(predicted_close, 2),
        "predicted_change_pct": round((predicted_close / current_close - 1) * 100, 2),
        "lower": round(current_close * float(np.exp(predicted_log_return + q_low)), 2),
        "upper": round(current_close * float(np.exp(predicted_log_return + q_high)), 2),
        "cv_mae_pct": round(float(np.mean(fold_mae) * 100), 2), "trained_rows": int(len(data)),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "technical_return_pct": round((np.exp(technical_return) - 1) * 100, 2),
        "news_adjustment_pct": round((np.exp(news_adjustment) - 1) * 100, 2),
        "market_adjustment_pct": round((np.exp(market_adjustment) - 1) * 100, 2),
    }
