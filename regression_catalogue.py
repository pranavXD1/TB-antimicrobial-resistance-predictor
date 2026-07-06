"""
Regression-based resistance catalogue.

Following Nature Communications 2025 (Hall et al. — multivariate logistic
regression on candidate-gene variants outperforms the WHO SOLO grading method),
this fits a sparse L1-penalised logistic regression per drug on the
candidate-gene SNP features and reads the coefficients as a *data-driven
catalogue*: a variant with a large positive coefficient is resistance-associated,
one with a negative coefficient is susceptibility-associated, and L1 drives the
uninformative majority to exactly zero. Each selected variant is cross-referenced
against the WHO 2023 catalogue to show how many the regression recovers and how
many fall outside it.

L1 logistic is the right tool here: the candidate-gene features are nearly
linearly separable (the logistic baseline already matches XGBoost on them), and
the sparsity yields a compact, interpretable variant list rather than a weight on
every feature.
"""
from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

from src.features.build_features import load_raw, mutation_matrix, build_dataset, available_drugs
from src.models.who_baseline import load_catalogue, _triple, _expand


def _candidate_snp_matrix(data_dir: str, min_count: int = 5) -> tuple[dict, pd.DataFrame]:
    """Build the candidate-gene SNP matrix regardless of ambient env settings —
    coefficients only make sense on per-SNP (not burden) candidate features. A
    prevalence floor is pinned here (not left to shell state) so singleton
    variants can't drive perfect-separation coefficients into the catalogue."""
    saved = {k: os.environ.get(k) for k in
             ("TBAMR_FEATURES", "TBAMR_CANDIDATE_ONLY", "TBAMR_MIN_COUNT")}
    os.environ["TBAMR_FEATURES"] = "snp"
    os.environ["TBAMR_CANDIDATE_ONLY"] = "1"
    os.environ["TBAMR_MIN_COUNT"] = str(min_count)
    try:
        raw = load_raw(data_dir)
        X = mutation_matrix(raw["variants"])
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return raw, X


def _who_known(feat: str, who_set: set) -> bool:
    tr = _triple(feat)
    return bool(tr and who_set and (_expand(*tr) & who_set))


def build_catalogue(data_dir: str, catalogue_path: str | None = None,
                    C: float = 1.0, top_k: int = 12, min_pos: int = 5,
                    min_count: int = 5):
    raw, X_full = _candidate_snp_matrix(data_dir, min_count=min_count)
    who = load_catalogue(catalogue_path) if catalogue_path else {}

    detail, summary = [], []
    for drug in available_drugs(data_dir):
        X, y, _ = build_dataset(data_dir, drug, X_full=X_full)
        if y.nunique() < 2 or int(y.sum()) < min_pos:
            continue
        clf = LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                 class_weight="balanced", max_iter=2000)
        clf.fit(csr_matrix(X.values), y)
        coef = clf.coef_[0]
        who_set = who.get(drug, set())

        pos = int((coef > 0).sum())
        known = novel = 0
        ranked = np.argsort(-coef)
        for rank, i in enumerate(ranked):
            if coef[i] <= 0:
                break
            feat = X.columns[i]
            is_known = _who_known(feat, who_set)
            known += is_known
            novel += (not is_known)
            if rank < top_k:
                detail.append({"drug": drug, "variant": feat,
                               "coef": round(float(coef[i]), 3),
                               "who_graded": is_known})
        summary.append({
            "Drug": drug, "n": len(y), "%R": round(100 * y.mean()),
            "selected_R_variants": pos,
            "WHO_recovered": known if who else np.nan,
            "not_in_WHO": novel if who else np.nan,
        })
    return pd.DataFrame(detail), pd.DataFrame(summary).sort_values(
        "selected_R_variants", ascending=False).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Regression-based resistance catalogue")
    ap.add_argument("--data", default=os.path.join("data", "vcf_indel"))
    ap.add_argument("--catalogue", default=None, help="WHO xlsx for cross-reference")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--C", type=float, default=1.0, help="inverse L1 strength")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--min-count", type=int, default=5,
                    help="prevalence floor for candidate variants")
    args = ap.parse_args()

    print("=" * 78)
    print("REGRESSION-BASED RESISTANCE CATALOGUE  |  L1-logistic on candidate genes")
    print("=" * 78)
    detail, summary = build_catalogue(args.data, args.catalogue, args.C,
                                      args.top_k, min_count=args.min_count)

    os.makedirs(args.reports, exist_ok=True)
    detail.to_csv(os.path.join(args.reports, "regression_catalogue.csv"), index=False)
    summary.to_csv(os.path.join(args.reports, "regression_catalogue_summary.csv"), index=False)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\nPer-drug summary:")
        print(summary.to_string(index=False))
        print("\nTop resistance-associated variants (by L1 coefficient):")
        for drug in summary["Drug"]:
            sub = detail[detail["drug"] == drug].head(6)
            if sub.empty:
                continue
            tags = ", ".join(
                f"{r.variant}{'' if (isinstance(r.who_graded,bool) and r.who_graded) else '*'}"
                f"({r.coef:+.2f})" for r in sub.itertuples())
            print(f"  {drug:14s} {tags}")
    print("\nSaved -> reports/regression_catalogue.csv (+ _summary.csv)")
    print("  * = not graded 'Assoc w R' in the WHO 2023 catalogue "
          "(novel candidate or lineage-correlated — warrants scrutiny)")


if __name__ == "__main__":
    main()
