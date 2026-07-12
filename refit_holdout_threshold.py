#!/usr/bin/env python
"""
Refit the decision thresholds on the external cohort — honestly.

Background: on the Sierra Leone holdout the models' *discrimination* (AUC)
transfers, but the GenTB models' calibrated 90%-sensitivity thresholds do not.
They were quantile-fit on GenTB's own score distribution and sit below almost
every external score, so at that threshold PZA/STR call nearly everything
resistant (specificity ~0). The ranking is fine; only the operating point is off.

This does NOT touch a single model weight — it only relocates the R/S cutoff.
For each drug it re-derives the threshold that hits a target sensitivity on the
external *calibrated* scores. Because choosing a threshold on the evaluation set
is itself a mild form of leakage, the reported operating point is computed by
LEAVE-ONE-OUT: every isolate is classified using a threshold fit on the other
n-1 isolates only. The single deployable threshold (fit on all n) is reported
separately, for use in the app. AUC is threshold-free and unchanged.

Outputs, per drug: AUC, the old (non-transferring) operating point, the refit
deployable threshold, and its honest leave-one-out sensitivity/specificity.

Usage:
    python -m src.models.refit_holdout_threshold \
        --db data/tbamr.duckdb --results results_sl/results \
        --models models --models-pza models_gentb_pza --models-str models_gentb_str \
        --target-sens 0.90 --reports reports
"""
from __future__ import annotations

import os
import argparse

import numpy as np
import pandas as pd
import duckdb
from sklearn.metrics import roc_auc_score

from src.models.evaluate_holdout import (
    DRUG_MODEL, collect_isolates, load_model, build_vector, apply_cal, metrics_at,
)


def threshold_for_sensitivity(scores: np.ndarray, y: np.ndarray, target: float) -> float:
    """Largest threshold (predict R iff score >= t) whose sensitivity >= target.

    Sensitivity is monotone non-increasing in t, so the passing thresholds form a
    prefix and the largest passing one gives the maximum specificity at the target
    recall. Candidate thresholds are the observed score values (ROC operating
    points)."""
    P = int((y == 1).sum())
    if P == 0:
        return float(scores.max()) + 1e-9
    best_t = None
    for t in np.unique(scores):
        sens = int(((scores >= t) & (y == 1)).sum()) / P
        if sens >= target:
            best_t = float(t)          # ascending scan -> last pass = largest passing t
    return best_t if best_t is not None else float(np.unique(scores).min())


def loo_operating_point(scores: np.ndarray, y: np.ndarray, target: float):
    """Leave-one-out sens/spec: classify each isolate with a threshold fit on the rest."""
    n = len(y)
    pred = np.zeros(n, dtype=int)
    for i in range(n):
        m = np.ones(n, dtype=bool); m[i] = False
        t = threshold_for_sensitivity(scores[m], y[m], target)
        pred[i] = 1 if scores[i] >= t else 0
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return sens, spec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/tbamr.duckdb")
    ap.add_argument("--results", default="results_sl/results")
    ap.add_argument("--dataset", default="sl_ext")
    ap.add_argument("--models", default="models")
    ap.add_argument("--models-pza", default="models_gentb_pza")
    ap.add_argument("--models-str", default="models_gentb_str")
    ap.add_argument("--target-sens", type=float, default=0.90)
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()

    dirs = {"cryptic": args.models, "pza": args.models_pza, "str": args.models_str}
    tgt = args.target_sens

    iso_feats = collect_isolates(args.results)
    con = duckdb.connect(args.db)
    ph = con.execute("""
        SELECT p.isolate_id, p.drug, p.phenotype
        FROM phenotypes p JOIN isolates i ON p.isolate_id = i.isolate_id
        WHERE i.dataset_id = ?
    """, [args.dataset]).df()
    con.close()
    pmap = {(r.isolate_id, r.drug): r.phenotype for r in ph.itertuples()}

    rows = []
    for drug, (dkey, stem) in DRUG_MODEL.items():
        model, feats, cal = load_model(dirs[dkey], stem)
        if model is None:
            continue
        ids = [i for i in iso_feats if (i, drug) in pmap]
        y = np.array([1 if pmap[(i, drug)] == "R" else 0 for i in ids])
        if len(np.unique(y)) < 2:
            continue

        X = pd.DataFrame([build_vector(iso_feats[i][0], iso_feats[i][1], feats)
                          for i in ids], columns=feats, index=ids)
        raw = model.predict_proba(X)[:, 1]
        p_cal, old_thr = apply_cal(cal, raw)

        auc = roc_auc_score(y, raw)
        old_sens, old_spec, _ = metrics_at(y, p_cal, old_thr)

        new_thr = threshold_for_sensitivity(p_cal, y, tgt)      # deployable (fit on all)
        in_sens, in_spec, _ = metrics_at(y, p_cal, new_thr)     # in-sample at refit thr
        loo_sens, loo_spec = loo_operating_point(p_cal, y, tgt)  # honest

        rows.append({
            "Drug": drug, "n": len(y), "R": int(y.sum()), "AUC": round(auc, 3),
            "old_thr": round(old_thr, 3),
            "old_Sens": round(old_sens, 3), "old_Spec": round(old_spec, 3),
            "new_thr": round(new_thr, 3),
            "Sens_insample": round(in_sens, 3), "Spec_insample": round(in_spec, 3),
            "Sens_LOO": round(loo_sens, 3), "Spec_LOO": round(loo_spec, 3),
        })

    df = pd.DataFrame(rows)
    os.makedirs(args.reports, exist_ok=True)
    df.to_csv(os.path.join(args.reports, "holdout_refit_thresholds.csv"), index=False)

    print("=" * 100)
    print(f"THRESHOLD REFIT ON EXTERNAL COHORT  |  target sensitivity {tgt:.0%}  "
          "(model weights unchanged; AUC identical)")
    print("=" * 100)
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(df.to_string(index=False))
    print("\nReading guide:")
    print("  AUC              = threshold-free, unchanged — the headline result")
    print("  old_thr/Sens/Spec= the calibrated thr90 as deployed. For the GenTB drugs it")
    print("                     sits below the external scores -> Spec collapses (the bug)")
    print("  new_thr          = refit threshold at the target sensitivity, on the external")
    print("                     calibrated scores — the value to deploy for this cohort")
    print("  Sens/Spec_LOO    = HONEST operating point (each isolate scored by a threshold")
    print("                     fit on the others) — report THESE, not the in-sample ones")
    print("  Sens/Spec_insample = at new_thr fit on all n; mildly optimistic, shown for the gap")
    print(f"\nSaved -> {os.path.join(args.reports, 'holdout_refit_thresholds.csv')}")
    print("To deploy: replace each drug's thr90 in the calibrators with new_thr (the")
    print("calibrator itself is unchanged — only the cutoff moves).")


if __name__ == "__main__":
    main()
