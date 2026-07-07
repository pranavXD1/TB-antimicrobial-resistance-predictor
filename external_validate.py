"""
External validation: apply frozen models to an independent holdout cohort and
measure real sensitivity/specificity against the cohort's OWN phenotypes.

No retraining. For each drug with a frozen model, the external isolates are put
into the model's *own* feature vocabulary (reindex the mutation matrix to the
booster's feature columns, fill 0) — variants the model never saw are ignored,
model features the isolate lacks become 0 — then scored. Validation is always
against the cohort's own pDST, never WHO-catalogue grades, so there is no leakage
from the catalogue the models were built alongside.

    python -m src.models.external_validate --models models \
        --db data/tbamr.duckdb --dataset sierraleone \
        --ml-metrics reports/cv_metrics.csv --out reports/external_validation.csv

--ml-metrics (optional) supplies each drug's frozen 90%-sensitivity threshold
(Thr@90 column) so the operating point is the one fixed on CRyPTIC, not retuned on
the external set. Feature mode must match training (export the same TBAMR_* vars).
"""
from __future__ import annotations

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.features.build_features import mutation_matrix, labels_for_drug


def _model_features(model) -> list[str] | None:
    b = model.get_booster()
    if b.feature_names:
        return list(b.feature_names)
    names = getattr(model, "feature_names_in_", None)
    return list(names) if names is not None else None


def _load_thresholds(ml_metrics: str | None) -> dict:
    if not ml_metrics or not os.path.exists(ml_metrics):
        return {}
    m = pd.read_csv(ml_metrics)
    col = "Thr@90" if "Thr@90" in m.columns else None
    if "Drug" not in m.columns or col is None:
        return {}
    return {str(r["Drug"]): float(r[col]) for _, r in m.iterrows() if pd.notna(r[col])}


def _metrics(y: np.ndarray, prob: np.ndarray, thr: float) -> dict:
    pred = (prob >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "Sens": round(sens, 3), "Spec": round(spec, 3),
            "PPV": round(ppv, 3), "NPV": round(npv, 3)}


def evaluate_external(models_dir, variants_df, pheno_df, thresholds=None,
                      calibrators_path=None) -> pd.DataFrame:
    thresholds = thresholds or {}
    cals = {}
    cp = calibrators_path or os.path.join(models_dir, "calibrators.joblib")
    if os.path.exists(cp):
        cals = joblib.load(cp)

    X_full = mutation_matrix(variants_df)
    rows = []

    for drug in sorted(set(pheno_df["drug"])):
        slug = drug.lower()
        mpath = os.path.join(models_dir, f"xgb_{slug}.joblib")
        if not os.path.exists(mpath):
            continue
        y = labels_for_drug(pheno_df, drug)
        if y.empty or y.nunique() < 2:
            continue
        model = joblib.load(mpath)
        feats = _model_features(model)
        if feats is None:
            continue
        # align to the model's own vocabulary AND to the external isolates
        X = X_full.reindex(index=y.index, columns=feats, fill_value=0)
        prob = model.predict_proba(X)[:, 1]
        if slug in {k.lower() for k in cals}:
            key = next(k for k in cals if k.lower() == slug)
            c = cals[key]
            from src.models.calibrate import apply_calibrator
            prob = apply_calibrator(c["kind"], c["model"], prob)

        yv = y.values.astype(int)
        auc = roc_auc_score(yv, prob) if len(set(yv)) > 1 else float("nan")
        thr = thresholds.get(drug, 0.5)
        row = {"Drug": drug, "n": len(yv), "%R": round(100 * yv.mean(), 1),
               "AUC_ext": round(auc, 3), "thr": round(thr, 3)}
        row.update(_metrics(yv, prob, thr))
        # also unseen-variant load: how many external variant tokens the model can even see
        seen = X.loc[:, (X != 0).any()].shape[1]
        row["feat_hit"] = f"{seen}/{len(feats)}"
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="External validation of frozen models")
    ap.add_argument("--models", default="models")
    ap.add_argument("--data", default=None, help="external data dir (variants.csv+phenotypes.csv)")
    ap.add_argument("--db", default=None, help="DuckDB path (use with --dataset)")
    ap.add_argument("--dataset", default=None, help="holdout dataset_id in the DB")
    ap.add_argument("--ml-metrics", default="reports/cv_metrics.csv",
                    help="CRyPTIC CV metrics, for the frozen Thr@90 per drug")
    ap.add_argument("--out", default="reports/external_validation.csv")
    args = ap.parse_args()

    if args.db and args.dataset:
        from src.data.build_db import connect, load_raw_from_db
        con = connect(args.db)
        raw = load_raw_from_db(con, datasets=[args.dataset])
        con.close()
        variants_df, pheno_df = raw["variants"], raw["phenotypes"]
    elif args.data:
        variants_df = pd.read_csv(os.path.join(args.data, "variants.csv"))
        pheno_df = pd.read_csv(os.path.join(args.data, "phenotypes.csv"))
    else:
        raise SystemExit("provide either --data DIR or --db PATH --dataset ID")

    thr = _load_thresholds(args.ml_metrics)
    df = evaluate_external(args.models, variants_df, pheno_df, thresholds=thr)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    print("=" * 78)
    print("EXTERNAL VALIDATION  |  frozen models vs the cohort's own pDST")
    print("=" * 78)
    if df.empty:
        print("No drug had both a frozen model and >=2 phenotype classes in the cohort.")
    else:
        print(df.to_string(index=False))
        print(f"\nSaved -> {args.out}")
    print("Reading guide: AUC_ext = ranking on the external cohort; Sens/Spec at the")
    print("frozen CRyPTIC threshold; feat_hit = model features actually seen in this cohort.")


if __name__ == "__main__":
    main()
