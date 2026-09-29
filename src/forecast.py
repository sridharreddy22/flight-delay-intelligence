"""Step 5 - Forecast the daily delay rate with Prophet (national + home airport).

How we evaluate honestly: hold out the last 90 days, train on everything before,
and compare Prophet's error with a simple baseline (the average of the last 28
training days). A model is only worth showing if it beats the baseline.
Then we refit on all data and forecast 180 days ahead.

Usage:
    python src/forecast.py
"""
import json

import numpy as np
import pandas as pd
from prophet import Prophet

from config import APP_DATA_DIR, HOME_AIRPORT

HOLDOUT_DAYS = 90
HORIZON_DAYS = 180


def daily_series(daily: pd.DataFrame, origin: str | None) -> pd.DataFrame:
    d = daily if origin is None else daily[daily["origin"] == origin]
    g = d.groupby("flight_date")[["completed", "delayed"]].sum()
    g = g[g["completed"] > 0]
    return pd.DataFrame({"ds": g.index, "y": (g["delayed"] / g["completed"]).values})


def make_model() -> Prophet:
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                daily_seasonality=False, interval_width=0.8)
    m.add_country_holidays(country_name="US")
    return m


def evaluate_and_forecast(series: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    train, test = series.iloc[:-HOLDOUT_DAYS], series.iloc[-HOLDOUT_DAYS:]

    model = make_model().fit(train)
    pred = model.predict(test[["ds"]])
    prophet_mae = np.abs(test["y"].values - pred["yhat"].clip(0, 1).values).mean()
    baseline_mae = (test["y"] - train["y"].tail(28).mean()).abs().mean()

    full = make_model().fit(series)
    fc = full.predict(full.make_future_dataframe(periods=HORIZON_DAYS))
    fc = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    fc[["yhat", "yhat_lower", "yhat_upper"]] = fc[["yhat", "yhat_lower", "yhat_upper"]].clip(0, 1)
    fc = fc.merge(series, on="ds", how="left").rename(columns={"y": "actual"})

    metrics = {
        "holdout_days": HOLDOUT_DAYS,
        "prophet_mae_pts": round(100 * prophet_mae, 2),
        "baseline_mae_pts": round(100 * baseline_mae, 2),
        "improvement_pct": round(100 * (1 - prophet_mae / baseline_mae), 1),
    }
    return fc, metrics


def main() -> None:
    daily = pd.read_parquet(APP_DATA_DIR / "daily.parquet")
    outputs, all_metrics = [], {}

    targets = {"National": None}
    if HOME_AIRPORT in set(daily["origin"]):
        targets[HOME_AIRPORT] = HOME_AIRPORT

    for label, origin in targets.items():
        fc, metrics = evaluate_and_forecast(daily_series(daily, origin))
        fc["series"] = label
        outputs.append(fc)
        all_metrics[label] = metrics
        print(f"{label}: Prophet MAE {metrics['prophet_mae_pts']} pts vs "
              f"baseline {metrics['baseline_mae_pts']} pts")

    pd.concat(outputs).to_parquet(APP_DATA_DIR / "forecast.parquet", index=False)
    (APP_DATA_DIR / "forecast_metrics.json").write_text(json.dumps(all_metrics, indent=2))
    print(f"saved forecast to {APP_DATA_DIR}")


if __name__ == "__main__":
    main()
