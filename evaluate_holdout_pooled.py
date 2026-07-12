#!/usr/bin/env python
"""
Pooled external validation across multiple holdout cohorts.

Scores each cohort's isolates with the frozen models, then POOLS the per-isolate
calibrated scores + phenotypic labels across cohorts and computes a single AUC
(+ bootstrap CI) and operating point per drug. Both cohorts are holdouts, so
pooling for reporting is legitimate and yields a balanced, better-powered
external result: Sierra Leone contributes the susceptible-heavy first-line side,
Belarus the resistant-heavy side plus a large streptomycin-resistant set.

Model weights untouched. AUC is rank-based and the calibrator is a single
monotone transform applied to every isolate, so pooled AUC is identical whether
computed on raw or calibrated scores. Per-cohort AUCs are reported alongside the
pooled value so any cohort-specific drift is visible.

Usage:
    python -m src.models.evaluate_holdout_pooled \
        --db data/tbamr.duckdb \
        --cohort sl_ext:results_sl/results \
        --cohort bel_ext:results_belarus/results \
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
    DRUG_MODEL, collect_isolates, load_model, build_vector, apply_cal, metrics_at, auc_ci,
)
from src.models.refit_holdout_threshold import threshold_for_sensitivity, loo_operating_point


def score_cohort(dataset: str, results_dir: str, db: str, dirs: dict) -> list:
    """Score one cohort with the frozen models -> per-isolate rows (cohort, isolate, drug, raw, cal, y)."""
    iso = collect_isolates(results_dir)
    con = duckdb.connect(db)
    ph = con.execute("""
        SELECT p.isolate_id, p.drug, p.phenotype
        FROM phenotypes p JOIN isolates i ON p.isolate_id = i.isolate_id
        WHERE i.dataset_id = ?
    """, [dataset]).df()
    con.close()
    pmap = {(r.isolate_id, r.drug): r.phenotype for r in ph.itertuples()}

    recs = []
    for drug, (dkey, stem) in DRUG_MODEL.items():
        model, feats, cal = load_model(dirs[dkey], stem)
        if model is None:
            continue
        ids = [i for i in iso if (i, drug) in pmap]
        if not ids:
            continue
        X = pd.DataFrame([build_vector(iso[i][0], iso[i][1], feats)
                          for i in ids], columns=feats, index=ids)
        raw = model.predict_proba(X)[:, 1]
        p_cal, _ = apply_cal(cal, raw)
        for k, i in enumerate(ids):
            recs.append({"cohort": dataset, "isolate": i, "drug": drug,
                         "raw": float(raw[k]), "cal": float(p_cal[k]),
                         "y": 1 if pmap[(i, drug)] == "R" else 0})
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/tbamr.duckdb")
    ap.add_argument("--cohort", action="append", required=True,
                    help="dataset:results_dir  (repeatable, e.g. sl_ext:results_sl/results)")
    ap.add_argument("--models", default="models")
    ap.add_argument("--models-pza", default="models_gentb_pza")
    ap.add_argument("--models-str", default="models_gentb_str")
    ap.add_argument("--target-sens", type=float, default=0.90)
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()

    dirs = {"cryptic": args.models, "pza": args.models_pza, "str": args.models_str}
    specs = [s.split(":", 1) for s in args.cohort]
    cohorts = [ds for ds, _ in specs]

    all_recs = []
    for ds, rd in specs:
        recs = score_cohort(ds, rd, args.db, dirs)
        n_iso = len({r["isolate"] for r in recs})
        print(f"scored {ds:8s}: {n_iso} isolates, {len(recs)} isolate-drug rows  ({rd})")
        all_recs += recs

    df = pd.DataFrame(all_recs)
    os.makedirs(args.reports, exist_ok=True)
    df.to_csv(os.path.join(args.reports, "pooled_scores.csv"), index=False)

    rows = []
    for drug in DRUG_MODEL:
        d = df[df["drug"] == drug]
        if d.empty or d["y"].nunique() < 2:
            continue
        y = d["y"].to_numpy(); raw = d["raw"].to_numpy(); cal = d["cal"].to_numpy()
        auc, lo, hi = auc_ci(y, raw)
        thr = threshold_for_sensitivity(cal, y, args.target_sens)
        loo_sens, loo_spec = loo_operating_point(cal, y, args.target_sens)
        s05, sp05, _ = metrics_at(y, raw, 0.5)
        row = {"Drug": drug, "n": len(y), "R": int(y.sum()), "S": int((y == 0).sum()),
               "AUC": round(auc, 3), "CI_lo": round(lo, 3), "CI_hi": round(hi, 3)}
        for c in cohorts:
            dc = d[d["cohort"] == c]
            row[f"AUC_{c}"] = round(roc_auc_score(dc["y"], dc["raw"]), 3) \
                if (not dc.empty and dc["y"].nunique() > 1) else np.nan
        row.update({"Sens@.5": round(s05, 3), "Spec@.5": round(sp05, 3),
                    "refit_thr": round(thr, 3),
                    "Sens_LOO": round(loo_sens, 3), "Spec_LOO": round(loo_spec, 3)})
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.reports, "pooled_metrics.csv"), index=False)

    print("\n" + "=" * 100)
    print("POOLED EXTERNAL VALIDATION — frozen models across " + " + ".join(cohorts))
    print("=" * 100)
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(out.to_string(index=False))
    print("\nReading guide:")
    print("  AUC / CI        = pooled ROC-AUC across all cohorts + bootstrap 95% CI (the headline)")
    print(f"  AUC_<cohort>    = per-cohort AUC, so cohort-specific drift is visible")
    print("  Sens/Spec_LOO   = pooled operating point at the target sensitivity, leave-one-out")
    print("                    (honest; refit on the pooled external calibrated scores)")
    print("  Sens/Spec@.5    = pooled operating point at raw 0.5")
    print(f"\nSaved -> {os.path.join(args.reports, 'pooled_metrics.csv')} + pooled_scores.csv")


if __name__ == "__main__":
    main()
