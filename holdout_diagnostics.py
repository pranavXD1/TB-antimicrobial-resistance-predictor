#!/usr/bin/env python
"""
Holdout diagnostics — two honest robustness checks on the external result.

(1) CALIBRATION TRANSFER. You refit the decision *threshold*; this asks whether
    the *probabilities* themselves transfer. Per drug it reports the Brier score
    of the raw and calibrated scores against the true labels, the calibration
    error (ECE), and mean-predicted vs observed resistance rate. Probabilities
    fit on CRyPTIC/GenTB usually don't perfectly transfer to a new cohort — that
    is expected and worth stating; a big Brier gap or ECE means "trust the
    ranking (AUC), not the raw probability" on external data.

(2) RESISTANCE-MISS ERROR ANALYSIS. For every drug it lists the false negatives
    (phenotypically resistant isolates the model scored < 0.5) and, for each,
    which of that drug's known resistance genes actually carry a variant. This is
    the mechanistic read on the STR ceiling: if missed STR-R isolates carry NO
    rpsL/rrs/gid variant, the resistance is invisible to any variant-based model
    (phenotype-only or a mechanism outside the panel) and the miss isn't the
    model's fault; if they DO carry one, the model is underweighting a real
    signal — a concrete, fixable limitation rather than a mysterious low number.

Reuses the scorer's feature reconstruction, so the vectors are identical to
evaluate_holdout. Model weights untouched.

Usage:
    python -m src.models.holdout_diagnostics \
        --db data/tbamr.duckdb --results results_sl/results \
        --models models --models-pza models_gentb_pza --models-str models_gentb_str \
        --reports reports
"""
from __future__ import annotations

import os
import argparse

import numpy as np
import pandas as pd
import duckdb
from sklearn.metrics import brier_score_loss

from src.models.evaluate_holdout import (
    DRUG_MODEL, collect_isolates, load_model, build_vector, apply_cal,
)
from src.features.build_features import GENE_COORDS

# known resistance genes per drug (names match GENE_COORDS)
DRUG_GENES = {
    "INH": {"katG", "fabG1", "inhA", "ahpC", "ndh", "mshA"},
    "RIF": {"rpoB", "rpoC"},
    "EMB": {"embB", "embA", "embC", "embR", "ubiA", "aftA"},
    "PZA": {"pncA"},
    "STR": {"rpsL", "rrs", "gid"},
}


def genes_hit(positions):
    hit = set()
    for pos in positions:
        for name, s, e in GENE_COORDS:
            if s <= pos <= e:
                hit.add(name)
                break
    return hit


def ece(y, p, bins=5):
    """Expected calibration error over equal-width probability bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    n = len(y)
    err = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        err += (m.sum() / n) * abs(float(p[m].mean()) - float(y[m].mean()))
    return err


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/tbamr.duckdb")
    ap.add_argument("--results", default="results_sl/results")
    ap.add_argument("--dataset", default="sl_ext")
    ap.add_argument("--models", default="models")
    ap.add_argument("--models-pza", default="models_gentb_pza")
    ap.add_argument("--models-str", default="models_gentb_str")
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()

    dirs = {"cryptic": args.models, "pza": args.models_pza, "str": args.models_str}
    iso_feats = collect_isolates(args.results)

    con = duckdb.connect(args.db)
    ph = con.execute("""
        SELECT p.isolate_id, p.drug, p.phenotype
        FROM phenotypes p JOIN isolates i ON p.isolate_id = i.isolate_id
        WHERE i.dataset_id = ?
    """, [args.dataset]).df()
    con.close()
    pmap = {(r.isolate_id, r.drug): r.phenotype for r in ph.itertuples()}

    calib_rows, miss_rows = [], []
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
        p_cal, _ = apply_cal(cal, raw)

        calib_rows.append({
            "Drug": drug, "n": len(y), "R": int(y.sum()),
            "Brier_raw": round(brier_score_loss(y, raw), 3),
            "Brier_cal": round(brier_score_loss(y, p_cal), 3),
            "ECE_cal": round(ece(y, p_cal), 3),
            "mean_pred": round(float(p_cal.mean()), 3),
            "obs_rate": round(float(y.mean()), 3),
        })

        pred = (raw >= 0.5).astype(int)
        for k, i in enumerate(ids):
            if y[k] == 1 and pred[k] == 0:                     # missed resistance
                g = sorted(genes_hit(iso_feats[i][1]) & DRUG_GENES.get(drug, set()))
                miss_rows.append({"Drug": drug, "isolate": i, "prob": round(float(raw[k]), 3),
                                  "R_genes_with_variant": ", ".join(g) if g else "— none —"})

    calib = pd.DataFrame(calib_rows)
    miss = pd.DataFrame(miss_rows)
    os.makedirs(args.reports, exist_ok=True)
    calib.to_csv(os.path.join(args.reports, "holdout_calibration.csv"), index=False)
    miss.to_csv(os.path.join(args.reports, "holdout_misses.csv"), index=False)

    print("=" * 88)
    print("(1) CALIBRATION TRANSFER on the external cohort  (lower Brier / ECE = better)")
    print("=" * 88)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(calib.to_string(index=False))
    print("  Brier_raw vs Brier_cal: did the training-set calibrator help or hurt out-of-domain?")
    print("  ECE_cal high or mean_pred >> obs_rate: probabilities are miscalibrated externally")
    print("  -> report AUC as the trustworthy summary; flag that raw probabilities need")
    print("     external recalibration before they'd mean what they say.")

    print("\n" + "=" * 88)
    print("(2) RESISTANCE MISSES (phenotype R, model prob < 0.5) + genes carrying a variant")
    print("=" * 88)
    if miss.empty:
        print("  none — every resistant isolate scored >= 0.5.")
    else:
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(miss.sort_values(["Drug", "prob"]).to_string(index=False))
        print("\n  '— none —' => the isolate carries NO variant in that drug's resistance genes,")
        print("  so a variant-based model cannot call it (mechanism outside the panel or")
        print("  phenotype/borderline) — not a model failure. A named gene => the model saw a")
        print("  variant there and still scored low: a real underweighting to write up.")
    print(f"\nSaved -> {os.path.join(args.reports, 'holdout_calibration.csv')} + holdout_misses.csv")


if __name__ == "__main__":
    main()
