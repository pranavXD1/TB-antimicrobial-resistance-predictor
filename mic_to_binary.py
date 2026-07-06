"""
MIC -> binary via ECOFF.

Instead of classifying R/S directly, regress log2(MIC) and derive the binary call
by thresholding the *predicted* MIC at the epidemiological cutoff (ECOFF). This
tests whether the quantitative phenotype carries signal the binary classifier
throws away — a hypothesis raised repeatedly in the CRyPTIC/TB-ML literature
(binary AUC can look high while dilution-level accuracy is modest).

Two comparisons per drug, both on out-of-fold predictions:
  * MIC_AUC  = AUC using the *predicted log2(MIC)* as a resistance score, vs the
               true binary label. Threshold-free; directly comparable to the direct
               binary classifier's AUC (read from reports/cv_metrics.csv).
  * ECOFF operating point = sensitivity/specificity when the predicted MIC is
               thresholded at the ECOFF. The ECOFF is recovered empirically as the
               true-MIC cut that best reproduces CRyPTIC's binary phenotype (the
               binary label is *defined* by MIC vs ECOFF), so no external constant
               is hard-coded.

Honours the same TBAMR_* feature env as every other module, so run it under the
candidate-gene settings to compare like with like.
"""
from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from src.features.build_features import load_raw, mutation_matrix, labels_for_drug
from src.models.mic_regress import (
    parse_mic, _xgbreg, _NAME_TO_CODE, _find_id_column, REUSE_DEFAULT,
)


def _oof_mic(X: pd.DataFrame, y_log2: pd.Series, k: int, seed: int) -> np.ndarray:
    """Out-of-fold predicted log2(MIC), aligned to y_log2.index order."""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    oof = np.full(len(y_log2), np.nan)
    for tr, te in kf.split(X):
        m = _xgbreg()
        m.fit(X.iloc[tr], y_log2.iloc[tr])
        oof[te] = m.predict(X.iloc[te])
    return oof


def _empirical_ecoff(true_log2: np.ndarray, binary: np.ndarray) -> float:
    """Recover the ECOFF as the true-MIC threshold maximising Youden's J against
    the binary phenotype (which CRyPTIC defines by MIC vs ECOFF)."""
    cuts = np.unique(true_log2)
    mids = (cuts[:-1] + cuts[1:]) / 2.0 if len(cuts) > 1 else cuts
    best_t, best_j = float(mids[0]), -1.0
    P, N = binary.sum(), (binary == 0).sum()
    for t in mids:
        pred = true_log2 > t
        sens = (pred & (binary == 1)).sum() / P if P else 0.0
        spec = (~pred & (binary == 0)).sum() / N if N else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def _sens_spec(pred: np.ndarray, binary: np.ndarray) -> tuple[float, float]:
    P, N = binary.sum(), (binary == 0).sum()
    sens = (pred & (binary == 1)).sum() / P if P else float("nan")
    spec = (~pred & (binary == 0)).sum() / N if N else float("nan")
    return float(sens), float(spec)


def _binary_auc_lookup(ml_metrics: str) -> dict[str, float]:
    if not ml_metrics or not os.path.exists(ml_metrics):
        return {}
    df = pd.read_csv(ml_metrics)
    if "Drug" not in df or "AUC_xgb" not in df:
        return {}
    return {str(r["Drug"]): float(r["AUC_xgb"]) for _, r in df.iterrows()}


def run(data_dir: str, reuse_table: str, ml_metrics: str = "reports/cv_metrics.csv",
        quality: str = "HIGH", k: int = 5, seed: int = 42) -> pd.DataFrame:
    raw = load_raw(data_dir)
    X_full = mutation_matrix(raw["variants"])          # honours TBAMR_* env vars
    ref = pd.read_csv(reuse_table, low_memory=False).set_index(_find_id_column(
        pd.read_csv(reuse_table, low_memory=False)))
    bin_auc = _binary_auc_lookup(ml_metrics)

    rows = []
    for drug in sorted(_NAME_TO_CODE):
        code = _NAME_TO_CODE[drug]
        mic_col, q_col = f"{code}_MIC", f"{code}_PHENOTYPE_QUALITY"
        if mic_col not in ref.columns:
            continue
        sub = ref[mic_col]
        if quality and q_col in ref.columns:
            sub = sub[ref[q_col].astype(str).str.upper() == quality.upper()]
        y_log2 = sub.map(parse_mic).dropna()
        binary = labels_for_drug(raw["phenotypes"], drug)

        common = X_full.index.intersection(y_log2.index).intersection(binary.index)
        if len(common) < 50:
            continue
        b = binary.loc[common].astype(int).values
        if b.sum() < 5 or (b == 0).sum() < 5 or y_log2.loc[common].nunique() < 3:
            continue

        Xc, yc = X_full.loc[common], y_log2.loc[common]
        oof = _oof_mic(Xc, yc, k, seed)
        mic_auc = roc_auc_score(b, oof)
        ecoff = _empirical_ecoff(yc.values, b)
        sens, spec = _sens_spec(oof > ecoff, b)

        rows.append({
            "Drug": drug, "n": len(common), "%R": round(100 * b.mean()),
            "MIC_AUC": round(mic_auc, 3),
            "binary_AUC": round(bin_auc[drug], 3) if drug in bin_auc else np.nan,
            "delta": round(mic_auc - bin_auc[drug], 3) if drug in bin_auc else np.nan,
            "ECOFF_mg/L": round(float(2 ** ecoff), 3),
            "Sens@ECOFF": round(sens, 3), "Spec@ECOFF": round(spec, 3),
        })
    return (pd.DataFrame(rows).sort_values("MIC_AUC", ascending=False)
            .reset_index(drop=True))


def main() -> None:
    ap = argparse.ArgumentParser(description="MIC->binary via ECOFF vs direct binary")
    ap.add_argument("--data", default=os.path.join("data", "vcf_indel"))
    ap.add_argument("--reuse-table", default=REUSE_DEFAULT)
    ap.add_argument("--ml-metrics", default="reports/cv_metrics.csv",
                    help="direct-binary AUC_xgb source for comparison")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--quality", default="HIGH")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 78)
    print("MIC -> BINARY via ECOFF  |  predicted-MIC score vs direct binary classifier")
    print("=" * 78)
    df = run(args.data, args.reuse_table, args.ml_metrics, args.quality,
             args.folds, args.seed)
    os.makedirs(args.reports, exist_ok=True)
    df.to_csv(os.path.join(args.reports, "mic_to_binary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.to_string(index=False))
    print("\nSaved -> reports/mic_to_binary.csv")
    print("\nReading guide:")
    print("  MIC_AUC     = AUC using predicted log2(MIC) as the resistance score")
    print("  binary_AUC  = the direct R/S classifier's AUC (reports/cv_metrics.csv)")
    print("  delta > 0   = regressing MIC then thresholding beats direct classification")
    print("  ECOFF_mg/L  = breakpoint recovered from the data (true MIC vs phenotype)")
    print("  Sens/Spec@ECOFF = binary call from thresholding *predicted* MIC at ECOFF")


if __name__ == "__main__":
    main()
