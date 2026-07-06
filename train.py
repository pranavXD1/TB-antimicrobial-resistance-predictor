"""
Train per-drug resistance classifiers.

For each drug we fit two models:
  * LogisticRegression  -- the baseline (interpretable, fast, class-weighted).
  * XGBoost             -- the workhorse; handles interactions & imbalance.

Why per-drug models for the MVP: each drug has its own resistance mechanism and
its own class balance, and one-vs-rest is the standard, debuggable starting
point. PHASE 2 swaps in a single multi-task model that predicts all drugs jointly
(sharing structure across co-resistant drugs) -- see README roadmap.

Everything is keyed off a held-out, stratified test split so evaluation is honest.
Models + test predictions are persisted to `models_dir` for the evaluate/explain
stages.
"""
from __future__ import annotations

import os
import json
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.features.build_features import build_dataset, mutation_matrix, load_raw, available_drugs


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


def train_drug(
    data_dir: str, drug: str, X_full: pd.DataFrame, test_size: float,
    seed: int, xgb_params: dict | None,
) -> dict:
    """Train baseline + XGBoost for one drug; return a result bundle."""
    X, y, features = build_dataset(data_dir, drug, X_full=X_full)
    n_r, n_s = int(y.sum()), int((y == 0).sum())
    if y.nunique() < 2 or min(n_r, n_s) < 6:
        return {"drug": drug, "skipped": True,
                "reason": f"too few for training (R={n_r}, S={n_s})"}

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    baseline = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    baseline.fit(csr_matrix(X_tr.values), y_tr)

    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    model = _xgb(spw, xgb_params)
    model.fit(X_tr, y_tr)

    return {
        "drug": drug, "skipped": False, "features": features,
        "baseline": baseline, "model": model,
        "X_test": X_te, "y_test": y_te,
        "p_baseline": baseline.predict_proba(csr_matrix(X_te.values))[:, 1],
        "p_model": model.predict_proba(X_te)[:, 1],
        "n_train": len(y_tr), "n_test": len(y_te), "prevalence": float(y.mean()),
    }


def train_all(
    data_dir: str, models_dir: str, drugs: list[str] | None = None,
    test_size: float = 0.25, seed: int = 42, xgb_params: dict | None = None,
) -> dict[str, dict]:
    os.makedirs(models_dir, exist_ok=True)
    raw = load_raw(data_dir)
    X_full = mutation_matrix(raw["variants"])
    drugs = drugs or available_drugs(data_dir)

    results: dict[str, dict] = {}
    for drug in drugs:
        res = train_drug(data_dir, drug, X_full, test_size, seed, xgb_params)
        results[drug] = res
        if res.get("skipped"):
            print(f"  {drug:<14} skipped ({res['reason']})")
            continue
        # persist models + test bundle
        slug = drug.lower()
        joblib.dump(res["model"], os.path.join(models_dir, f"xgb_{slug}.joblib"))
        joblib.dump(res["baseline"], os.path.join(models_dir, f"logit_{slug}.joblib"))
        bundle = {k: res[k] for k in ("X_test", "y_test", "p_baseline", "p_model", "features")}
        joblib.dump(bundle, os.path.join(models_dir, f"testbundle_{slug}.joblib"))
        print(f"  {drug:<14} trained  (train={res['n_train']:,}  test={res['n_test']:,}"
              f"  prevalence={res['prevalence']:.1%})")

    with open(os.path.join(models_dir, "drugs.json"), "w") as f:
        json.dump([d for d, r in results.items() if not r.get("skipped")], f)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train resistance models")
    parser.add_argument("--data", default="data/sample")
    parser.add_argument("--models", default="models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_all(args.data, args.models, seed=args.seed)
