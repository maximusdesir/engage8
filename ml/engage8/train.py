"""Train the run/pass (and explosive) models with HONEST evaluation.

Key correctness choices:
  * Time-based split (train on older seasons, test on the newest) — never a
    random split, which leaks future plays and inflates accuracy.
  * Probability calibration on a held-out season, so the reported "63% pass"
    actually means ~63% over many such situations.
  * Reports Brier score + log loss + ROC-AUC, not just accuracy.

Run:
    python -m engage8.train
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
)

from . import config
from .features import ALL_FEATURES, CATEGORICAL_FEATURES, TARGET


def _split_by_season(df: pd.DataFrame):
    """Return (train, calib, test) by season. Newest = test."""
    seasons = sorted(df["season"].dropna().unique())
    if len(seasons) < 3:
        raise SystemExit(
            "Need >=3 seasons for a clean train/calib/test split. "
            f"Got: {seasons}. Pull more seasons in `extract`."
        )
    test_season = seasons[-1]
    calib_season = seasons[-2]
    train = df[df["season"] <= seasons[-3]]
    calib = df[df["season"] == calib_season]
    test = df[df["season"] == test_season]
    print(
        f"Split  train: {sorted(train['season'].unique())} ({len(train):,})  "
        f"calib: {calib_season} ({len(calib):,})  "
        f"test: {test_season} ({len(test):,})"
    )
    return train, calib, test


def _fit_lgbm(X_train, y_train, X_calib, y_calib):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=8,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_calib, y_calib)],
        eval_metric="binary_logloss",
        categorical_feature=[c for c in CATEGORICAL_FEATURES if c in X_train.columns],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def _evaluate(name, y_true, p_pred, baseline_rate=None) -> dict:
    y_hat = (p_pred >= 0.5).astype(int)
    metrics = {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_hat)), 4),
        "brier": round(float(brier_score_loss(y_true, p_pred)), 4),
        "log_loss": round(float(log_loss(y_true, p_pred, labels=[0, 1])), 4),
        "roc_auc": round(float(roc_auc_score(y_true, p_pred)), 4),
    }
    if baseline_rate is not None:
        # Accuracy of always guessing the majority class.
        metrics["baseline_accuracy"] = round(float(max(baseline_rate, 1 - baseline_rate)), 4)
    print(f"  [{name}] " + "  ".join(f"{k}={v}" for k, v in metrics.items()))
    return metrics


def train_runpass(df: pd.DataFrame) -> dict:
    train, calib, test = _split_by_season(df)
    feats = [c for c in ALL_FEATURES if c in df.columns]

    X_train, y_train = train[feats], train[TARGET].astype(int)
    X_calib, y_calib = calib[feats], calib[TARGET].astype(int)
    X_test, y_test = test[feats], test[TARGET].astype(int)

    print("Training run/pass model ...")
    model = _fit_lgbm(X_train, y_train, X_calib, y_calib)

    # Calibrate raw probabilities on the held-out calib season (isotonic).
    raw_calib = model.predict_proba(X_calib)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_calib, y_calib)

    # Evaluate on the untouched test season, raw vs calibrated.
    raw_test = model.predict_proba(X_test)[:, 1]
    cal_test = calibrator.predict(raw_test)

    base_rate = float(y_train.mean())
    print("Test-season performance:")
    raw_metrics = _evaluate("raw", y_test, raw_test, base_rate)
    cal_metrics = _evaluate("calibrated", y_test, cal_test, base_rate)

    # Feature importances for the README / "why".
    importances = dict(
        sorted(
            zip(feats, (int(i) for i in model.feature_importances_)),
            key=lambda kv: kv[1], reverse=True,
        )
    )

    import joblib
    artifact = {
        "model": model,
        "calibrator": calibrator,
        "features": feats,
        "categorical_features": [c for c in CATEGORICAL_FEATURES if c in feats],
        "target": TARGET,
        "version": "0.1.0",
    }
    joblib.dump(artifact, config.MODEL_PATH)
    print(f"Saved model -> {config.MODEL_PATH}")

    return {
        "test_season_raw": raw_metrics,
        "test_season_calibrated": cal_metrics,
        "train_pass_rate": round(base_rate, 4),
        "feature_importances": importances,
        "n_estimators_used": int(model.best_iteration_ or model.n_estimators),
    }


def main() -> None:
    if not config.FEATURES_PARQUET.exists():
        raise SystemExit(
            f"{config.FEATURES_PARQUET} not found. Run `python -m engage8.features` first."
        )
    df = pd.read_parquet(config.FEATURES_PARQUET)
    # Restore category dtypes (parquet preserves them, but be safe).
    for c in CATEGORICAL_FEATURES:
        if c in df.columns:
            df[c] = df[c].astype("category")

    report = train_runpass(df)
    with open(config.METRICS_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved metrics -> {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
