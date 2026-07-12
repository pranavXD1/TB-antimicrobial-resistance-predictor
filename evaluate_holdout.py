#!/usr/bin/env python
"""
External-validation scoring: run the FROZEN CRyPTIC + GenTB models against the
Sierra Leone (PRJEB7727 / Schleusener srep46327) holdout in ``sl_ext``.

For each of the five drugs with phenotypes (INH, RIF, EMB, PZA, STR) this:
  1. loads the trained XGBoost model + probability calibrator from the right dir
     (CRyPTIC first-line in ``models/``; PZA/STR from the GenTB model dirs),
  2. reconstructs each isolate's feature vector *exactly* the way training did —
     SNP presence by exact token match + per-gene ``burden::`` counts from
     GENE_COORDS — reusing the project's own feature definitions,
  3. emits BOTH token forms per variant (``g{pos}_{ref}>{alt}`` for the CRyPTIC
     models and ``gene@{pos}_{ref}>{alt}`` for the GenTB models), so neither
     family silently sees an all-zero SNP block,
  4. prints a per-model FEATURE-OVERLAP check (how many of the model's SNP
     features the cohort actually matched) — the guard against a misalignment
     that would fake a ~0.5 AUC,
  5. reports per-drug AUC + sensitivity/specificity (at the model's calibrated
     90%-sensitivity threshold and at 0.5), an M. africanum-vs-L4 subgroup split,
     and a side-by-side against the paper's own TBProfiler numbers.

Features come from the TB-Profiler JSONs (they carry gene names, needed for the
GenTB token form); labels come from ``sl_ext`` phenotypes in the DB.

Usage:
    python -m src.models.evaluate_holdout \
        --db data/tbamr.duckdb --results results_sl/results \
        --models models --models-pza models_gentb_pza --models-str models_gentb_str \
        --benchmark paper_benchmark.csv --reports reports
"""
from __future__ import annotations

import os
import glob
import json
import argparse

import numpy as np
import pandas as pd
import joblib
import duckdb
from sklearn.metrics import roc_auc_score, confusion_matrix

from src.features.build_features import _extract_pos, GENE_COORDS

# drug code -> (models dir key, model file stem)
DRUG_MODEL = {
    "INH": ("cryptic", "isoniazid"),
    "RIF": ("cryptic", "rifampicin"),
    "EMB": ("cryptic", "ethambutol"),
    "PZA": ("pza", "pyrazinamide"),
    "STR": ("str", "streptomycin"),
}


# ----------------------------------------------------------------------------
# per-isolate features from the TB-Profiler JSONs
# ----------------------------------------------------------------------------
def read_isolate(path):
    """Return (id, dual-format token set, position set, main_lineage) or None for a stub."""
    with open(path) as fh:
        d = json.load(fh)
    iso = d.get("id") or os.path.basename(path).split(".")[0]
    dr = d.get("dr_variants") or []
    other = d.get("other_variants") or []
    if not dr and not other:
        return None
    toks, positions = set(), set()
    for v in dr + other:
        if v.get("filter") not in (None, "pass", "PASS"):
            continue
        pos, ref, alt = v.get("pos"), v.get("ref"), v.get("alt")
        if pos is None or not ref or not alt:
            continue
        positions.add(int(pos))
        toks.add(f"g{pos}_{ref}>{alt}")                       # CRyPTIC form
        for g in (v.get("gene_name"), v.get("gene_id"), v.get("locus_tag")):
            if g:
                toks.add(f"{g}@{pos}_{ref}>{alt}")            # GenTB form
    return iso, toks, positions, str(d.get("main_lineage") or "?")


def collect_isolates(results_dir):
    """{isolate_id: (tokens, positions, lineage)}, unioning a sample's multiple run files."""
    out = {}
    for f in sorted(glob.glob(os.path.join(results_dir, "*.results.json"))):
        r = read_isolate(f)
        if r is None:
            continue
        iso, toks, pos, lin = r
        if iso in out:
            out[iso][0].update(toks)
            out[iso][1].update(pos)
        else:
            out[iso] = [set(toks), set(pos), lin]
    return out


def build_vector(tokens, positions, feats):
    """SNP presence (exact token match) + per-gene burden (distinct positions in
    the gene window, first-match-wins, no pad) — identical to predict.build_vector
    / build_features.gene_burden_matrix."""
    gene_hits = {}
    for pos in positions:
        for name, s, e in GENE_COORDS:
            if s <= pos <= e:
                gene_hits.setdefault(name, set()).add(pos)
                break
    row = []
    for f in feats:
        if f.startswith("burden::"):
            row.append(len(gene_hits.get(f.split("::", 1)[1], set())))
        else:
            row.append(1 if f in tokens else 0)
    return row


# ----------------------------------------------------------------------------
# model loading + calibration
# ----------------------------------------------------------------------------
def load_model(models_dir, stem):
    path = os.path.join(models_dir, f"xgb_{stem}.joblib")
    if not os.path.exists(path):
        return None, None, None
    model = joblib.load(path)
    feats = list(model.get_booster().feature_names)
    cal = None
    cpath = os.path.join(models_dir, "calibrators.joblib")
    if os.path.exists(cpath):
        cal = joblib.load(cpath).get(stem.lower())
    return model, feats, cal


def apply_cal(cal, raw):
    if cal is None:
        return raw, 0.5
    raw = np.asarray(raw, dtype=float)
    if cal["kind"] == "isotonic":
        p = np.asarray(cal["model"].predict(raw), dtype=float)
    else:
        p = cal["model"].predict_proba(raw.reshape(-1, 1))[:, 1]
    return p, float(cal.get("thr90", 0.5))


def metrics_at(y, p, thr):
    yp = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    return sens, spec, ppv


def auc_ci(y, p, B=1000, seed=42):
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    auc = roc_auc_score(y, p)
    rng = np.random.default_rng(seed)
    n = len(y); boots = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], p[idx]))
    lo = float(np.percentile(boots, 2.5)) if boots else float("nan")
    hi = float(np.percentile(boots, 97.5)) if boots else float("nan")
    return float(auc), lo, hi


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/tbamr.duckdb")
    ap.add_argument("--results", default="results_sl/results")
    ap.add_argument("--dataset", default="sl_ext")
    ap.add_argument("--models", default="models")
    ap.add_argument("--models-pza", default="models_gentb_pza")
    ap.add_argument("--models-str", default="models_gentb_str")
    ap.add_argument("--benchmark", default="paper_benchmark.csv")
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()

    dirs = {"cryptic": args.models, "pza": args.models_pza, "str": args.models_str}

    iso_feats = collect_isolates(args.results)
    print(f"loaded features for {len(iso_feats)} isolates from {args.results}")

    con = duckdb.connect(args.db)
    ph = con.execute("""
        SELECT p.isolate_id, p.drug, p.phenotype
        FROM phenotypes p JOIN isolates i ON p.isolate_id = i.isolate_id
        WHERE i.dataset_id = ?
    """, [args.dataset]).df()
    con.close()
    pmap = {(r.isolate_id, r.drug): r.phenotype for r in ph.itertuples()}

    bench = {}
    if os.path.exists(args.benchmark):
        b = pd.read_csv(args.benchmark)
        bench = {r.drug: (r.sens, r.spec)
                 for r in b[b.tool == "TBProfiler"].itertuples()}

    rows, sub_rows = [], []
    for drug, (dkey, stem) in DRUG_MODEL.items():
        model, feats, cal = load_model(dirs[dkey], stem)
        if model is None:
            print(f"  [skip] {drug}: no model at {dirs[dkey]}/xgb_{stem}.joblib")
            continue

        ids = [i for i in iso_feats if (i, drug) in pmap]
        y = np.array([1 if pmap[(i, drug)] == "R" else 0 for i in ids])
        if len(np.unique(y)) < 2:
            print(f"  [skip] {drug}: <2 phenotype classes among {len(ids)} isolates")
            continue

        X = pd.DataFrame([build_vector(iso_feats[i][0], iso_feats[i][1], feats)
                          for i in ids], columns=feats, index=ids)
        raw = model.predict_proba(X)[:, 1]
        p_cal, thr = apply_cal(cal, raw)

        # feature-overlap sanity gate
        snp_feats = [f for f in feats if not f.startswith("burden::")]
        n_burden = len(feats) - len(snp_feats)
        matched = int(sum(1 for f in snp_feats
                          if X[f].sum() > 0)) if snp_feats else 0

        auc, lo, hi = auc_ci(y, raw)
        s5, sp5, _ = metrics_at(y, raw, 0.5)
        st, spt, ppvt = metrics_at(y, p_cal, thr)
        bs, bsp = bench.get(drug, (np.nan, np.nan))
        rows.append({
            "Drug": drug, "n": len(y), "R": int(y.sum()), "S": int((y == 0).sum()),
            "AUC": round(auc, 3), "CI_lo": round(lo, 3), "CI_hi": round(hi, 3),
            "Sens@thr": round(st, 3), "Spec@thr": round(spt, 3), "PPV@thr": round(ppvt, 3),
            "Sens@.5": round(s5, 3), "Spec@.5": round(sp5, 3),
            "SNP_match": f"{matched}/{len(snp_feats)}", "burden": n_burden,
            "TBP_sens": bs, "TBP_spec": bsp,
        })

        # M. africanum (L5/L6) vs L4 subgroup
        for label, keep in [("L4", {"lineage4"}), ("M.afri(L5/6)", {"lineage5", "lineage6"})]:
            gi = [k for k, i in enumerate(ids) if iso_feats[i][2] in keep]
            if len(gi) < 8:
                continue
            yg, pg = y[gi], raw[gi]
            if len(np.unique(yg)) < 2:
                sub_rows.append({"Drug": drug, "group": label, "n": len(gi),
                                 "R": int(yg.sum()), "AUC": "n/a (1 class)"})
                continue
            sub_rows.append({"Drug": drug, "group": label, "n": len(gi),
                             "R": int(yg.sum()), "AUC": round(roc_auc_score(yg, pg), 3)})

    df = pd.DataFrame(rows)
    os.makedirs(args.reports, exist_ok=True)
    df.to_csv(os.path.join(args.reports, "holdout_metrics.csv"), index=False)

    print("\n" + "=" * 92)
    print("EXTERNAL VALIDATION — frozen models on Sierra Leone holdout (vs phenotypic DST)")
    print("=" * 92)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.to_string(index=False))
    if sub_rows:
        print("\nLineage subgroup AUC (M. africanum vs lineage 4):")
        print(pd.DataFrame(sub_rows).to_string(index=False))
    print("\nReading guide:")
    print("  AUC / CI      = pooled ROC-AUC on the holdout + bootstrap 95% CI")
    print("  Sens/Spec@thr = at the model's calibrated 90%-sensitivity threshold (its")
    print("                  deployment operating point; external sens may drift from .90)")
    print("  SNP_match     = model SNP features the cohort actually hit — if this is ~0")
    print("                  while AUC~0.5, it's feature MISALIGNMENT, not a real result")
    print("  TBP_sens/spec = the paper's own TBProfiler vs-DST numbers, same cohort")
    print(f"\nSaved -> {os.path.join(args.reports, 'holdout_metrics.csv')}")


if __name__ == "__main__":
    main()
