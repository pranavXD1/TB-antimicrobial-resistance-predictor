"""
Rigorous evaluation: cross-validated, confidence-bounded, threshold-aware.

The single-split metrics in `evaluate.py` are fine for a quick read, but with
20-40 test isolates per drug their AUCs have wide, unstated uncertainty and the
0.5 threshold gives poor sensitivity (the clinically dangerous direction). This
module fixes both, using NO new data:

  * STRATIFIED K-FOLD CV -> every isolate is a test isolate exactly once
    (out-of-fold predictions), so the AUC uses the whole dataset, not a lucky
    25% slice. We also report the mean +/- sd of per-fold AUC as a stability check.
  * BOOTSTRAP 95% CI on the pooled out-of-fold predictions -> turns "AUC 0.93"
    into "AUC 0.93 (95% CI 0.87-0.97)", which is what makes a number defensible.
  * SENSITIVITY-TARGETED THRESHOLD -> instead of 0.5, pick the operating point
    that catches >= TARGET fraction of resistant isolates, and report the
    specificity you pay for it. Missing a resistant call means prescribing a
    drug that won't work, so we tune for sensitivity and are honest about cost.

The XGBoost configuration is identical to src/models/train.py, so these numbers
describe the same model the rest of the pipeline ships.
"""
from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy.sparse import csr_matrix
from xgboost import XGBClassifier

from src.features.build_features import (
    load_raw, mutation_matrix, build_dataset, available_drugs,
)


# ---- model (kept in lock-step with train.py) --------------------------------
def _xgb(scale_pos_weight: float, params: dict | None = None) -> XGBClassifier:
    base = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
        eval_metric="logloss", n_jobs=-1, tree_method="hist",
    )
    if params:
        base.update(params)
    base["scale_pos_weight"] = scale_pos_weight
    return XGBClassifier(**base)


# ---- metric helpers ---------------------------------------------------------
def _oof_predictions(X, y, k: int, seed: int, xgb_params: dict | None):
    """Out-of-fold predicted probabilities for XGBoost and logistic baseline,
    plus the list of per-fold XGBoost AUCs."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    p_xgb = np.full(len(y), np.nan)
    p_base = np.full(len(y), np.nan)
    fold_aucs: list[float] = []

    for tr, te in skf.split(X, y):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        m = _xgb(spw, xgb_params)
        m.fit(X_tr, y_tr)
        pte = m.predict_proba(X_te)[:, 1]
        p_xgb[te] = pte

        b = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        b.fit(csr_matrix(X_tr.values), y_tr)
        p_base[te] = b.predict_proba(csr_matrix(X_te.values))[:, 1]

        if len(np.unique(y_te)) > 1:
            fold_aucs.append(roc_auc_score(y_te, pte))

    return p_xgb, p_base, fold_aucs


def _bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, B: int, seed: int):
    """Percentile 95% CI for AUC by resampling isolates with replacement."""
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yt = y[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, p[idx]))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _at_threshold(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    yp = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    return {"threshold": float(t), "sens": sens, "spec": spec, "ppv": ppv}


def _threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float) -> dict:
    """Highest threshold whose sensitivity >= target (i.e. max specificity while
    still catching `target` of resistant isolates). Sensitivity is monotone in
    the threshold, so the passing set is downward-closed and this is well defined."""
    best = None
    for t in np.unique(p):
        r = _at_threshold(y, p, t)
        if r["sens"] >= target:
            best = r  # keep the largest passing threshold
    if best is None:
        best = _at_threshold(y, p, 0.0)  # predict all resistant -> sens = 1
    best["target_sens"] = target
    return best


# ---- driver -----------------------------------------------------------------
def cross_validate(
    data_dir: str, folds: int = 5, seed: int = 42,
    target_sens: float = 0.90, bootstrap: int = 1000,
    xgb_params: dict | None = None,
) -> pd.DataFrame:
    raw = load_raw(data_dir)
    X_full = mutation_matrix(raw["variants"])

    rows = []
    for drug in available_drugs(data_dir):
        X, y, _ = build_dataset(data_dir, drug, X_full=X_full)
        n_r, n_s = int(y.sum()), int((y == 0).sum())
        min_class = min(n_r, n_s)
        if y.nunique() < 2 or min_class < 2:
            print(f"  {drug:14s} skipped (R={n_r}, S={n_s})")
            continue

        k = min(folds, min_class)  # each fold needs >=1 of each class
        p_xgb, p_base, fold_aucs = _oof_predictions(X, y, k, seed, xgb_params)

        yv = y.values
        auc_xgb = roc_auc_score(yv, p_xgb)
        auc_base = roc_auc_score(yv, p_base)
        lo, hi = _bootstrap_auc_ci(yv, p_xgb, bootstrap, seed)
        at50 = _at_threshold(yv, p_xgb, 0.5)
        tuned = _threshold_for_sensitivity(yv, p_xgb, target_sens)

        rows.append({
            "Drug": drug, "n": len(y), "%R": round(100 * y.mean()),
            "k": k,
            "AUC_xgb": round(auc_xgb, 3),
            "CI_low": round(lo, 3), "CI_high": round(hi, 3),
            "AUC_cv_mean": round(float(np.mean(fold_aucs)), 3) if fold_aucs else float("nan"),
            "AUC_cv_sd": round(float(np.std(fold_aucs)), 3) if fold_aucs else float("nan"),
            "AUC_logit": round(auc_base, 3),
            "Sens@.5": round(at50["sens"], 3), "Spec@.5": round(at50["spec"], 3),
            f"Thr@{int(target_sens*100)}": round(tuned["threshold"], 3),
            "Sens@thr": round(tuned["sens"], 3),
            "Spec@thr": round(tuned["spec"], 3),
            "PPV@thr": round(tuned["ppv"], 3),
        })

    df = pd.DataFrame(rows).sort_values("AUC_xgb", ascending=False).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser(description="Cross-validated evaluation with CIs")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-sens", type=float, default=0.90)
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    print("=" * 78)
    print(f"CROSS-VALIDATED EVALUATION  |  {args.folds}-fold, "
          f"{args.bootstrap} bootstrap resamples, sensitivity target "
          f"{args.target_sens:.0%}")
    print("=" * 78)

    df = cross_validate(
        args.data, folds=args.folds, seed=args.seed,
        target_sens=args.target_sens, bootstrap=args.bootstrap,
    )

    os.makedirs(args.reports, exist_ok=True)
    out = os.path.join(args.reports, "cv_metrics.csv")
    df.to_csv(out, index=False)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\n" + df.to_string(index=False))
    print(f"\nSaved -> {out}")
    print("\nReading guide:")
    print("  AUC_xgb  = pooled out-of-fold AUC (whole dataset, each isolate tested once)")
    print("  CI_low/high = bootstrap 95% CI on that AUC (overlap near 0.5 = weak signal)")
    print("  AUC_cv_mean +/- sd = per-fold AUC stability")
    print(f"  Thr@{int(args.target_sens*100)} / Spec@thr = threshold to catch "
          f"{args.target_sens:.0%} of resistant isolates, and the specificity it costs")


if __name__ == "__main__":
    main()
