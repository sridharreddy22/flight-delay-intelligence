"""Step 6 - Train an XGBoost model: "Will this flight arrive 15+ minutes late?"

Design choices worth explaining in an interview:
  * Only features known at BOOKING time (airline, airports, date, scheduled hour,
    distance). No weather or actual departure delay, which would leak the answer.
    That's why AUC lands around 0.65-0.72, not 0.95. It's the honest number.
  * Time-based split: train on older flights, test on the most recent 3 months,
    the way the model would really be used.
  * No class re-weighting, so predicted probabilities stay calibrated
    (a "30% chance" really means about 30% of such flights are late).
  * Explanations use XGBoost's built-in TreeSHAP (pred_contribs=True).

Usage:
    python src/train_model.py                 # samples 1.5M completed flights
    python src/train_model.py --rows 500000   # smaller/faster on a laptop
"""
import argparse
import json

import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from config import APP_DATA_DIR, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FEATURES = ["month", "day_of_week", "dep_hour", "airline", "origin", "dest", "distance"]
CATEGORICAL = ["airline", "origin", "dest"]
LABELS = {"month": "Month", "day_of_week": "Day of week", "dep_hour": "Departure hour",
          "airline": "Airline", "origin": "Origin airport", "dest": "Destination airport",
          "distance": "Distance"}


def load_sample(total_rows: int) -> pd.DataFrame:
    files = sorted(PROCESSED_DIR.glob("flights_*.parquet"))
    if not files:
        raise SystemExit("No processed files. Run src/clean_data.py first.")
    per_file = max(1, total_rows // len(files))
    frames = []
    for path in files:
        df = pd.read_parquet(path, columns=FEATURES + ["flight_date", "is_delayed"])
        df = df[df["is_delayed"].notna()]  # completed flights only
        frames.append(df.sample(n=min(per_file, len(df)), random_state=42))
    return pd.concat(frames, ignore_index=True)


def prepare(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """Identical preprocessing for training and for the app."""
    X = df[FEATURES].copy()
    for col in ("month", "day_of_week", "dep_hour"):
        X[col] = X[col].astype("int32")
    X["distance"] = X["distance"].astype("float32")
    for col in CATEGORICAL:
        X[col] = pd.Categorical(X[col].astype(str), categories=categories[col])
    return X


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_500_000)
    args = parser.parse_args()

    data = load_sample(args.rows)
    categories = {c: sorted(data[c].astype(str).unique()) for c in CATEGORICAL}
    X, y = prepare(data, categories), data["is_delayed"].astype(int)

    cutoff = data["flight_date"].max() - pd.DateOffset(months=3)
    is_test = (data["flight_date"] > cutoff).values
    X_train, y_train, X_test, y_test = X[~is_test], y[~is_test], X[is_test], y[is_test]
    print(f"train {len(X_train):,} rows | test {len(X_test):,} rows "
          f"(flights after {cutoff.date()}) | delay rate {y.mean():.1%}")

    rng = np.random.default_rng(42)
    is_val = rng.random(len(X_train)) < 0.1  # for early stopping only
    model = xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.08, max_depth=7, min_child_weight=20,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        enable_categorical=True, max_cat_to_onehot=1,
        eval_metric="auc", early_stopping_rounds=30, n_jobs=-1,
    )
    model.fit(X_train[~is_val], y_train[~is_val],
              eval_set=[(X_train[is_val], y_train[is_val])], verbose=50)

    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "test_period_start": str((cutoff + pd.Timedelta(days=1)).date()),
        "test_rows": int(len(y_test)),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 3),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 3),
        "pr_auc_baseline": round(float(y_test.mean()), 3),  # what random guessing scores
        "brier": round(float(brier_score_loss(y_test, proba)), 4),
        "base_delay_rate": round(float(y.mean()), 4),
    }
    print(json.dumps(metrics, indent=2))

    # SHAP values via XGBoost's built-in TreeSHAP; last column is the bias term.
    sample = X_test.sample(n=min(20_000, len(X_test)), random_state=0)
    contribs = model.get_booster()[: model.best_iteration + 1].predict(
        xgb.DMatrix(sample, enable_categorical=True), pred_contribs=True)[:, :-1]
    importance = (pd.DataFrame({"feature": FEATURES,
                                "mean_abs_shap": np.abs(contribs).mean(axis=0)})
                  .sort_values("mean_abs_shap", ascending=False))
    importance["label"] = importance["feature"].map(LABELS)

    fig, ax = plt.subplots(figsize=(7, 4))
    ordered = importance.iloc[::-1]
    ax.barh(ordered["label"], ordered["mean_abs_shap"], color="#2F64A8")
    ax.set_xlabel("Mean |SHAP value| (impact on log-odds of delay)")
    ax.set_title("What drives the delay prediction")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "shap_importance.png", dpi=150)

    # Keep only the trees up to the best early-stopping round, so the app matches these metrics.
    booster = model.get_booster()[: model.best_iteration + 1]
    booster.save_model(MODELS_DIR / "delay_model.json")
    (MODELS_DIR / "categories.json").write_text(json.dumps(categories))
    (APP_DATA_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    importance.to_csv(APP_DATA_DIR / "feature_importance.csv", index=False)
    print(f"saved model to {MODELS_DIR} and chart to {REPORTS_DIR / 'shap_importance.png'}")


if __name__ == "__main__":
    main()
