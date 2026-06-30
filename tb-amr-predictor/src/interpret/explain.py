"""
Interpretability layer -- the seed of the clinical decision-support feature.

Two things clinicians and reviewers want from an AMR predictor:
  1. GLOBAL: which mutations is the model actually using for each drug? If the
     answer matches known biology (rpoB for rifampicin, katG for isoniazid),
     the model is trustworthy; if it leans on lineage background SNPs, it's
     confounded.
  2. LOCAL: for *this* isolate, WHY did you predict resistant? A prediction with
     "driven by rpoB_S450L" is actionable; a bare probability is not.

We use TreeSHAP (exact for tree models) via the `shap` library. PHASE 3 turns the
local explanation into a full report: predicted regimen, the mutations behind
each call, and confidence -- but the engine is right here.
"""
from __future__ import annotations

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def global_importance(models_dir: str, drug: str, top_n: int = 12) -> pd.DataFrame:
    """Mean |SHAP| per mutation for one drug's XGBoost model (test set)."""
    model = joblib.load(os.path.join(models_dir, f"xgb_{drug.lower()}.joblib"))
    bundle = joblib.load(os.path.join(models_dir, f"testbundle_{drug.lower()}.joblib"))
    X = bundle["X_test"]

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    mean_abs = np.abs(sv).mean(axis=0)
    imp = (
        pd.DataFrame({"mutation": X.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return imp


def save_summary_plot(models_dir: str, drug: str, out_path: str, top_n: int = 12) -> str:
    model = joblib.load(os.path.join(models_dir, f"xgb_{drug.lower()}.joblib"))
    bundle = joblib.load(os.path.join(models_dir, f"testbundle_{drug.lower()}.joblib"))
    X = bundle["X_test"]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)

    plt.figure()
    shap.summary_plot(sv, X, plot_type="bar", max_display=top_n, show=False)
    plt.title(f"Top mutations driving {drug} resistance prediction")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    return out_path


def explain_isolate(models_dir: str, drug: str, row: int = 0, top_n: int = 6) -> dict:
    """Local 'why this prediction' for a single test isolate."""
    model = joblib.load(os.path.join(models_dir, f"xgb_{drug.lower()}.joblib"))
    bundle = joblib.load(os.path.join(models_dir, f"testbundle_{drug.lower()}.joblib"))
    X = bundle["X_test"]
    explainer = shap.TreeExplainer(model)

    x = X.iloc[[row]]
    sv = explainer.shap_values(x)[0]
    prob = float(model.predict_proba(x)[0, 1])
    contrib = (
        pd.DataFrame({"mutation": X.columns, "shap": sv, "present": x.iloc[0].values})
        .assign(abs=lambda d: d["shap"].abs())
        .sort_values("abs", ascending=False)
        .head(top_n)
    )
    drivers = [
        {"mutation": r.mutation, "present": bool(r.present), "shap": round(float(r.shap), 3)}
        for r in contrib.itertuples()
    ]
    return {
        "isolate": X.index[row], "drug": drug,
        "predicted_prob_resistant": round(prob, 3),
        "call": "RESISTANT" if prob >= 0.5 else "susceptible",
        "top_drivers": drivers,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Explain a model")
    parser.add_argument("--models", default="models")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--drug", default="Rifampicin")
    args = parser.parse_args()

    print(f"Global mutation importance for {args.drug}:")
    print(global_importance(args.models, args.drug).to_string(index=False))
    plot = save_summary_plot(args.models, args.drug,
                             os.path.join(args.reports, f"shap_{args.drug.lower()}.png"))
    print(f"\nSaved SHAP summary plot -> {plot}")
    print("\nExample per-isolate explanation:")
    print(json.dumps(explain_isolate(args.models, args.drug), indent=2))
