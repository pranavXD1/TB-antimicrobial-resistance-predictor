"""
Probability calibration.

The per-drug XGBoost models train with scale_pos_weight to handle class imbalance,
which makes their raw probabilities well-ranked but poorly calibrated
(systematically inflated, worst for the rare drugs). This fits a per-drug
calibrator on out-of-fold predictions so the frontend can show a probability that
means what it says, and recomputes the 90%-sensitivity threshold on the calibrated
scale so both frontend threshold modes stay consistent.

Isotonic where there are enough positives to support it; Platt (sigmoid) for the
rare drugs. Reported Brier scores are honest — the calibrator is fit and evaluated
on disjoint halves of the out-of-fold predictions — while the saved calibrator is
fit on all out-of-fold predictions for deployment.

Honours the same TBAMR_* feature env as training, so calibrate the model you serve.
"""
from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier

from src.features.build_features import load_raw, mutation_matrix, build_dataset, available_drugs

ISOTONIC_MIN_POS = 200


def _xgb(spw: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
        eval_metric="logloss", n_jobs=-1, tree_method="hist",
        scale_pos_weight=spw,
    )


def _oof_probs(X: pd.DataFrame, y: pd.Series, k: int, seed: int) -> np.ndarray:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    yv = y.values
    for tr, te in skf.split(X, y):
        spw = float((yv[tr] == 0).sum() / max((yv[tr] == 1).sum(), 1))
        m = _xgb(spw)
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return oof


def _fit_calibrator(prob: np.ndarray, y: np.ndarray, kind: str):
    if kind == "isotonic":
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(prob, y)
        return ir
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(prob.reshape(-1, 1), y)
    return lr


def apply_calibrator(kind: str, model, prob) -> np.ndarray:
    prob = np.asarray(prob, dtype=float)
    if kind == "isotonic":
        return np.asarray(model.predict(prob), dtype=float)
    return model.predict_proba(prob.reshape(-1, 1))[:, 1]


def _thr_for_90(prob: np.ndarray, y: np.ndarray) -> float:
    """Calibrated-scale threshold catching ~90% of resistant isolates."""
    pos = prob[y == 1]
    return float(np.quantile(pos, 0.10)) if len(pos) else 0.5


def run(data_dir: str, models_dir: str = "models", reports: str = "reports",
        k: int = 5, seed: int = 42):
    raw = load_raw(data_dir)
    X_full = mutation_matrix(raw["variants"])          # honours TBAMR_* env vars
    store, rows = {}, []
    rng = np.random.default_rng(seed)

    for drug in available_drugs(data_dir):
        X, y, _ = build_dataset(data_dir, drug, X_full=X_full)
        if y.nunique() < 2 or int(y.sum()) < 10:
            continue
        oof = _oof_probs(X, y, k, seed)
        yv = y.values
        kind = "isotonic" if int(yv.sum()) >= ISOTONIC_MIN_POS else "sigmoid"

        # honest Brier: fit on one half of the OOF preds, evaluate on the other
        idx = rng.permutation(len(yv))
        a, b = idx[:len(idx) // 2], idx[len(idx) // 2:]
        br_raw = brier_score_loss(yv, oof)

        def _half(tr, te):
            c = _fit_calibrator(oof[tr], yv[tr], kind)
            return brier_score_loss(yv[te], apply_calibrator(kind, c, oof[te]))

        br_cal = 0.5 * (_half(a, b) + _half(b, a))

        # deployable calibrator on all OOF preds + calibrated 90%-sens threshold
        cal = _fit_calibrator(oof, yv, kind)
        thr90 = _thr_for_90(apply_calibrator(kind, cal, oof), yv)
        store[drug.lower()] = {"kind": kind, "model": cal, "thr90": thr90}
        rows.append({"Drug": drug, "n": len(y), "%R": round(100 * yv.mean()),
                     "method": kind, "Brier_raw": round(br_raw, 4),
                     "Brier_cal": round(br_cal, 4),
                     "improve_%": round(100 * (br_raw - br_cal) / br_raw, 1) if br_raw else 0.0,
                     "thr90_cal": round(thr90, 3)})

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(store, os.path.join(models_dir, "calibrators.joblib"))
    df = pd.DataFrame(rows).sort_values("Brier_raw", ascending=False).reset_index(drop=True)
    os.makedirs(reports, exist_ok=True)
    df.to_csv(os.path.join(reports, "calibration.csv"), index=False)
    return df, store


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-drug probability calibration")
    ap.add_argument("--data", default=os.path.join("data", "vcf_indel"))
    ap.add_argument("--models", default="models")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 78)
    print("PROBABILITY CALIBRATION  |  per-drug, out-of-fold, Brier before/after")
    print("=" * 78)
    df, store = run(args.data, args.models, args.reports, args.folds, args.seed)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.to_string(index=False))
    print(f"\nSaved -> {os.path.join(args.models, 'calibrators.joblib')} "
          f"({len(store)} drugs)  +  reports/calibration.csv")
    print("\nReading guide:")
    print("  Brier_raw/cal = mean squared error of the probability (lower = better);")
    print("                  honest split-half estimate")
    print("  improve_%     = Brier reduction from calibration")
    print("  thr90_cal     = 90%-sensitivity threshold on the CALIBRATED scale")
    print("  The app applies these automatically once calibrators.joblib exists.")


if __name__ == "__main__":
    main()
