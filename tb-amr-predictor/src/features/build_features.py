"""
Turn the long-format variant & phenotype tables into model-ready matrices.

This is the data-engineering seam of the project. It is deliberately agnostic to
*where* the CSVs came from (synthetic generator or the real download script), so
the modelling code never knows or cares about the data source.

Pipeline:
    variants.csv (isolate, mutation)  --pivot-->  binary mutation matrix X
    phenotypes.csv (isolate, drug, R/S) --filter per drug--> label vector y
    -> align X and y on the isolates that have a phenotype for that drug.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd


def load_raw(data_dir: str) -> dict[str, pd.DataFrame]:
    """Load variants / phenotypes / lineages from a directory of CSVs."""
    out = {}
    for name in ("variants", "phenotypes", "lineages"):
        path = os.path.join(data_dir, f"{name}.csv")
        if os.path.exists(path):
            out[name] = pd.read_csv(path)
    if "variants" not in out or "phenotypes" not in out:
        raise FileNotFoundError(
            f"Expected variants.csv and phenotypes.csv in {data_dir}. "
            "Run `python -m src.data.synthetic` or `src.data.download` first."
        )
    return out


def mutation_matrix(variants: pd.DataFrame) -> pd.DataFrame:
    """Pivot long (isolate, mutation) -> wide binary matrix indexed by isolate."""
    variants = variants.copy()
    variants["present"] = 1
    X = (
        variants.pivot_table(
            index="isolate_id", columns="mutation", values="present",
            aggfunc="max", fill_value=0,
        )
        .astype(np.int8)
        .sort_index()
    )
    X.columns.name = None
    return X


def labels_for_drug(phenotypes: pd.DataFrame, drug: str) -> pd.Series:
    """R/S phenotype for one drug -> {1,0} Series indexed by isolate (R=1)."""
    sub = phenotypes.loc[phenotypes["drug"] == drug, ["isolate_id", "phenotype"]]
    y = sub.set_index("isolate_id")["phenotype"].map({"R": 1, "S": 0})
    return y.dropna().astype(int)


def build_dataset(
    data_dir: str, drug: str, X_full: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Return (X, y, feature_names) for one drug, aligned on isolates that have a
    phenotype for that drug. Pass a precomputed `X_full` to avoid re-pivoting.
    """
    raw = load_raw(data_dir)
    if X_full is None:
        X_full = mutation_matrix(raw["variants"])
    y = labels_for_drug(raw["phenotypes"], drug)
    common = X_full.index.intersection(y.index)
    X = X_full.loc[common]
    y = y.loc[common]
    return X, y, list(X.columns)


def available_drugs(data_dir: str) -> list[str]:
    raw = load_raw(data_dir)
    return sorted(raw["phenotypes"]["drug"].unique())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect built features")
    parser.add_argument("--data", default="data/sample")
    args = parser.parse_args()

    raw = load_raw(args.data)
    X = mutation_matrix(raw["variants"])
    print(f"Mutation matrix: {X.shape[0]:,} isolates x {X.shape[1]} mutations")
    print(f"Drugs available: {available_drugs(args.data)}")
    print("\nPer-drug class balance (R / total tested):")
    for drug in available_drugs(args.data):
        _, y, _ = build_dataset(args.data, drug, X_full=X)
        print(f"  {drug:<14} {int(y.sum()):>5} / {len(y):>5}  ({y.mean():.1%} resistant)")
