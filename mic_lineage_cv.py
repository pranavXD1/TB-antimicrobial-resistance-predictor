"""
Population-structure check for the MIC regressor.

Finding 5 showed that regressing MIC and using the predicted value as a resistance
score lifts bedaquiline's ranking AUC (0.776 -> 0.870). Every *binary* model in this
project was checked against leave-lineages-out CV; this applies the identical check
to the MIC model, so that gain is verified rather than assumed.

For each drug, out-of-fold predicted log2(MIC) is built two ways -- random 5-fold and
GroupKFold by genetic cluster (test lineages unseen in training) -- and both are
scored by AUC of predicted MIC against the true binary phenotype (the same MIC_AUC as
Finding 5, directly comparable). A small random->grouped drop means the MIC ranking
generalises across genetic backgrounds; a large drop means it was riding population
structure, exactly as the genome-wide binary AUCs did.

Honours the same TBAMR_* feature env as the other modules -- run it under the
candidate-gene settings used for Finding 5.
"""
from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import roc_auc_score

from src.features.build_features import load_raw, mutation_matrix, labels_for_drug
from src.models.mic_regress import (
    parse_mic, _xgbreg, _NAME_TO_CODE, _find_id_column, REUSE_DEFAULT,
)
from src.models.lineage_cv import genetic_clusters


def _oof_mic(X: pd.DataFrame, y_log2: pd.Series, groups: np.ndarray,
             k: int, seed: int, grouped: bool) -> np.ndarray:
    """Out-of-fold predicted log2(MIC), random or grouped-by-cluster."""
    oof = np.full(len(y_log2), np.nan)
    if grouped:
        split = GroupKFold(n_splits=k).split(X, y_log2, groups)
    else:
        split = KFold(n_splits=k, shuffle=True, random_state=seed).split(X)
    for tr, te in split:
        m = _xgbreg()
        m.fit(X.iloc[tr], y_log2.iloc[tr])
        oof[te] = m.predict(X.iloc[te])
    return oof


def run(data_dir: str, reuse_table: str, n_clusters: int = 25, k: int = 5,
        seed: int = 42, quality: str = "HIGH") -> pd.DataFrame:
    raw = load_raw(data_dir)
    X_full = mutation_matrix(raw["variants"])          # honours TBAMR_* env vars
    clusters = genetic_clusters(X_full, n_clusters, seed)
    print(f"  derived {clusters.nunique()} genetic clusters "
          f"(sizes {clusters.value_counts().min()}–{clusters.value_counts().max()})")

    ref = pd.read_csv(reuse_table, low_memory=False)
    ref = ref.set_index(_find_id_column(ref))

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

        common = (X_full.index.intersection(y_log2.index)
                  .intersection(binary.index).intersection(clusters.index))
        if len(common) < 50:
            continue
        b = binary.loc[common].astype(int).values
        if b.sum() < 5 or (b == 0).sum() < 5 or y_log2.loc[common].nunique() < 3:
            continue

        Xc, yc, gc = X_full.loc[common], y_log2.loc[common], clusters.loc[common].values
        oof_r = _oof_mic(Xc, yc, gc, k, seed, grouped=False)
        oof_g = _oof_mic(Xc, yc, gc, k, seed, grouped=True)
        ar, ag = roc_auc_score(b, oof_r), roc_auc_score(b, oof_g)
        rows.append({"Drug": drug, "n": len(common), "%R": round(100 * b.mean()),
                     "MIC_AUC_random": round(ar, 3),
                     "MIC_AUC_grouped": round(ag, 3),
                     "drop": round(ar - ag, 3)})
    return (pd.DataFrame(rows).sort_values("MIC_AUC_random", ascending=False)
            .reset_index(drop=True))


def main() -> None:
    ap = argparse.ArgumentParser(description="Population-structure CV for the MIC regressor")
    ap.add_argument("--data", default=os.path.join("data", "vcf_indel"))
    ap.add_argument("--reuse-table", default=REUSE_DEFAULT)
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--clusters", type=int, default=25)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quality", default="HIGH")
    args = ap.parse_args()

    print("=" * 78)
    print("MIC REGRESSOR — POPULATION-STRUCTURE CV  |  random vs leave-lineages-out")
    print("=" * 78)
    df = run(args.data, args.reuse_table, args.clusters, args.folds, args.seed, args.quality)
    os.makedirs(args.reports, exist_ok=True)
    df.to_csv(os.path.join(args.reports, "mic_lineage_cv.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.to_string(index=False))
    print("\nSaved -> reports/mic_lineage_cv.csv")
    print("\nReading guide:")
    print("  MIC_AUC_random  = predicted-MIC ranking AUC under random 5-fold "
          "(matches Finding 5)")
    print("  MIC_AUC_grouped = same, GroupKFold by genetic cluster "
          "(test lineages unseen)")
    print("  drop            = how much the MIC ranking relied on population "
          "structure; small = generalises")


if __name__ == "__main__":
    main()
