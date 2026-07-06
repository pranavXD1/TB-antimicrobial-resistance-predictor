"""
Population-structure-aware validation -- is the model learning resistance, or
lineage background?

Random cross-validation lets isolates from the same *M. tuberculosis* lineage sit
in both train and test. If a resistance marker is merely correlated with a lineage
(e.g. via co-inherited background SNPs), random CV rewards the model for
recognising the lineage, inflating apparent accuracy. The honest test is to hold
out whole genetic backgrounds: train on some lineages, test on others.

We don't have named lineages for the VCF-only cohort, so we derive genetic
clusters directly from the variant matrix (TruncatedSVD -> MiniBatchKMeans). The
major clusters correspond to lineages/sublineages, so grouping CV folds by cluster
approximates leave-lineages-out validation. For each drug we compare:

  * AUC (random)  -- standard stratified 5-fold (isolates mixed across clusters)
  * AUC (grouped) -- GroupKFold by cluster (test clusters unseen in training)

A large random - grouped gap means the model leaned on population structure; a
small gap means the signal generalises across genetic backgrounds (i.e. it is
learning resistance, not lineage). This is exactly the check that separates a
convincing genomic-prediction result from a confounded one.
"""
from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.features.build_features import (
    load_raw, mutation_matrix, build_dataset, available_drugs,
)


def _xgb(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
        eval_metric="logloss", n_jobs=-1, tree_method="hist",
        scale_pos_weight=scale_pos_weight,
    )


def genetic_clusters(X: pd.DataFrame, n_clusters: int, seed: int) -> pd.Series:
    """Cluster isolates by genetic background: SVD-reduce the sparse variant
    matrix, then k-means. Cluster labels approximate lineage/sublineage."""
    Xs = csr_matrix(X.values.astype(np.float32))
    n_comp = min(50, max(2, min(X.shape) - 1))
    emb = TruncatedSVD(n_components=n_comp, random_state=seed).fit_transform(Xs)
    labels = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed,
                             n_init=3).fit_predict(emb)
    return pd.Series(labels, index=X.index, name="cluster")


def _oof_auc_random(X, y, k, seed):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    p = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y):
        spw = float((y.iloc[tr] == 0).sum() / max((y.iloc[tr] == 1).sum(), 1))
        m = _xgb(spw).fit(X.iloc[tr], y.iloc[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return roc_auc_score(y.values, p)


def _oof_auc_grouped(X, y, groups, k):
    gkf = GroupKFold(n_splits=k)
    p = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        ytr = y.iloc[tr]
        if ytr.nunique() < 2:            # degenerate train fold -> skip
            continue
        spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
        m = _xgb(spw).fit(X.iloc[tr], ytr)
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    ok = ~np.isnan(p)
    if len(np.unique(y.values[ok])) < 2:
        return float("nan")
    return roc_auc_score(y.values[ok], p[ok])


def run(data_dir: str, n_clusters: int = 25, k: int = 5, seed: int = 42) -> pd.DataFrame:
    raw = load_raw(data_dir)
    X_full = mutation_matrix(raw["variants"])
    clusters = genetic_clusters(X_full, n_clusters, seed)
    sizes = clusters.value_counts()
    print(f"  derived {clusters.nunique()} genetic clusters "
          f"(sizes {sizes.min()}–{sizes.max()}, median {int(sizes.median())})")

    rows = []
    for drug in available_drugs(data_dir):
        X, y, _ = build_dataset(data_dir, drug, X_full=X_full)
        if y.nunique() < 2 or min(int(y.sum()), int((y == 0).sum())) < k:
            continue
        g = clusters.loc[X.index]
        n_groups = g.nunique()
        kk = min(k, n_groups)
        if kk < 2:
            continue
        auc_rand = _oof_auc_random(X, y, k, seed)
        auc_grp = _oof_auc_grouped(X, y, g, kk)
        rows.append({
            "Drug": drug, "n": len(y), "%R": round(100 * y.mean()),
            "clusters": n_groups,
            "AUC_random": round(auc_rand, 3),
            "AUC_grouped": round(auc_grp, 3),
            "drop": round(auc_rand - auc_grp, 3),
        })
    return (pd.DataFrame(rows).sort_values("AUC_random", ascending=False)
            .reset_index(drop=True))


def main() -> None:
    ap = argparse.ArgumentParser(description="Population-structure-aware CV")
    ap.add_argument("--data", default=os.path.join("data", "vcf"))
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--clusters", type=int, default=25)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 78)
    print(f"POPULATION-STRUCTURE-AWARE CV  |  {args.clusters} genetic clusters, "
          f"{args.folds}-fold random vs grouped")
    print("=" * 78)
    df = run(args.data, args.clusters, args.folds, args.seed)

    os.makedirs(args.reports, exist_ok=True)
    out = os.path.join(args.reports, "lineage_cv.csv")
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\n" + df.to_string(index=False))
    print(f"\nSaved -> {out}")
    print("\nReading guide:")
    print("  AUC_random  = standard stratified CV (isolates mixed across lineages)")
    print("  AUC_grouped = GroupKFold by genetic cluster (test lineages unseen in training)")
    print("  drop        = how much accuracy relied on population structure;")
    print("                small drop = signal generalises = learning resistance, not lineage")


if __name__ == "__main__":
    main()
