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
import re
import numpy as np
import pandas as pd


# H37Rv (NC_000962.3) coordinates [start, end] of resistance-associated genes,
# used to build gene-BURDEN features (pool every variant in a gene into one
# signal). Covers first/second-line targets plus the newer bedaquiline /
# clofazimine / linezolid / delamanid genes whose resistance is allelically
# heterogeneous -- the case gene burden is meant to rescue. Windows need only be
# approximately right; a boundary mis-bin of a few bp is immaterial to a count.
GENE_COORDS: list[tuple[str, int, int]] = [
    ("gyrB", 5240, 7267), ("gyrA", 7302, 9818),
    ("fgd1", 486624, 487631), ("embR", 1416181, 1417347),
    ("rpoB", 759807, 763325), ("rpoC", 763370, 767320),
    ("mmpS5", 775087, 775590), ("mmpL5", 775586, 778480),
    ("Rv0678", 778990, 779487), ("rpsL", 781560, 781934),
    ("rplC", 800809, 801462), ("fbiC", 1302571, 1304935),
    ("atpE", 1461045, 1461290), ("rrs", 1471846, 1473382),
    ("rrl", 1473658, 1476795), ("fabG1", 1673440, 1674183),
    ("inhA", 1674202, 1675011), ("rpsA", 1833542, 1834987),
    ("tlyA", 1917940, 1918746), ("ndh", 2101651, 2103042),
    ("katG", 2153889, 2156111), ("Rv1979c", 2221073, 2222401),
    ("pncA", 2288681, 2289241), ("eis", 2714124, 2715332),
    ("ahpC", 2726088, 2726780), ("pepQ", 2859300, 2860418),
    ("ribD", 2986839, 2987960), ("thyA", 3073680, 3074471),
    ("whiB7", 3568402, 3568987), ("fbiA", 3640318, 3641349),
    ("fbiB", 3641346, 3642740), ("ddn", 3986844, 3987299),
    ("panD", 4043863, 4044249), ("embC", 4239863, 4243147),
    ("embA", 4243233, 4246517), ("embB", 4246514, 4249810),
    ("aftA", 4267356, 4268516), ("ubiA", 4269503, 4270603),
    ("ethA", 4326004, 4327473), ("ethR", 4327549, 4328199),
    ("gid", 4407528, 4408202), ("folC", 2746234, 2747512),
    ("mshA", 575148, 576704),
]


def _extract_pos(token: str) -> int | None:
    """Pull the genomic position out of a mutation token, whatever its format:
    'rpoB@761155_C>T' / 'Rv0678@779100_A>G' / 'g761155_C>T' / raw
    'NC_000962.3_761155_C_T'."""
    if "@" in token:
        m = re.match(r"(\d+)", token.split("@", 1)[1])
        return int(m.group(1)) if m else None
    if token[:1] == "g" and token[1:2].isdigit():
        m = re.match(r"(\d+)", token[1:])
        return int(m.group(1)) if m else None
    parts = token.rsplit("_", 3)
    if len(parts) == 4 and parts[1].isdigit():
        return int(parts[1])
    return None


def gene_burden_matrix(variants: pd.DataFrame, all_isolates) -> pd.DataFrame:
    """Per-(isolate, gene) burden = number of distinct variant positions the
    isolate carries inside that resistance gene. Pools rare/private alleles into
    a single per-gene signal, which is what individual-SNP features miss for the
    heterogeneous last-line drugs. Uses the FULL (unfiltered) variant table."""
    v = variants[["isolate_id", "mutation"]].copy()
    v["pos"] = v["mutation"].map(_extract_pos)
    v = v.dropna(subset=["pos"])
    v["pos"] = v["pos"].astype(int)

    v["gene"] = None
    for name, start, end in GENE_COORDS:           # first-match wins (handles the
        m = v["gene"].isna() & v["pos"].between(start, end)   # tiny emb overlaps)
        v.loc[m, "gene"] = name
    v = v.dropna(subset=["gene"])

    if v.empty:
        return pd.DataFrame(index=sorted(all_isolates))
    burden = (v.groupby(["isolate_id", "gene"])["pos"].nunique()
              .unstack(fill_value=0))
    burden = burden.reindex(sorted(all_isolates), fill_value=0)
    burden.columns = [f"burden::{c}" for c in burden.columns]
    burden.columns.name = None
    return burden.astype(np.int16)


def candidate_region_filter(variants: pd.DataFrame, pad: int = 200) -> pd.DataFrame:
    """Keep only variants whose position falls within a resistance gene (± pad bp).

    Genome-wide SNP tokens are dominated by lineage/phylogenetic background;
    restricting to curated resistance genes is standard practice and removes the
    population-structure noise that inflates apparent accuracy and confounds
    generalisation. The pad captures promoter/upstream mutations (e.g. eis, inhA,
    ahpC promoters) that sit just outside the coding sequence. Padding is applied
    symmetrically to avoid needing per-gene strand, which over-includes only a
    little harmless downstream sequence."""
    pos = variants["mutation"].map(_extract_pos).fillna(-1).astype(int)
    mask = np.zeros(len(variants), dtype=bool)
    p = pos.to_numpy()
    for _name, start, end in GENE_COORDS:
        mask |= (p >= start - pad) & (p <= end + pad)
    return variants[mask]


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


def mutation_matrix(variants: pd.DataFrame, min_count: int | None = None,
                    max_features: int | None = None) -> pd.DataFrame:
    """Pivot long (isolate, mutation) -> wide binary matrix indexed by isolate.

    Two prevalence controls, applied BEFORE the pivot, keep the matrix tractable
    on large cohorts (raw position tokens give 150k+ SNPs, mostly rare lineage
    background that only add noise, overfitting, and memory blow-ups):
      * min_count      -- drop mutations seen in fewer than this many isolates
                          (a prevalence floor; real signal like rpoB/katG is in
                          hundreds of isolates, so it survives).
      * max_features   -- hard ceiling: after the floor, keep only the top-K
                          most-prevalent mutations. Guarantees a fixed memory
                          footprint regardless of how many isolates you load.

    Both default to env vars (TBAMR_MIN_COUNT=1, TBAMR_MAX_FEATURES=0/off) so they
    can be tuned per-run without touching calling code.

    Feature representation is chosen by the TBAMR_FEATURES env var:
      * "snp"    (default) -- the per-SNP binary matrix described above.
      * "burden"           -- per-gene burden features only (pooled rare alleles).
      * "both"             -- SNP matrix + gene-burden columns concatenated. This
                              keeps the sharp single-SNP signal for first-line
                              drugs while adding pooled signal for the allelically
                              heterogeneous last-line drugs.
    Burden is always computed from the UNFILTERED variants, so the rare alleles
    that min_count removes still contribute to their gene's burden.

    If TBAMR_CANDIDATE_ONLY is set, the per-SNP features are restricted to
    variants inside resistance genes (± TBAMR_GENE_PAD bp, default 200, to catch
    promoters). This drops genome-wide lineage background — standard practice in
    the field — and is the main lever against population-structure confounding.
    """
    if min_count is None:
        min_count = int(os.environ.get("TBAMR_MIN_COUNT", "1"))
    if max_features is None:
        max_features = int(os.environ.get("TBAMR_MAX_FEATURES", "0"))
    mode = os.environ.get("TBAMR_FEATURES", "snp").lower()
    candidate_only = os.environ.get("TBAMR_CANDIDATE_ONLY", "0").lower() in (
        "1", "true", "yes")
    gene_pad = int(os.environ.get("TBAMR_GENE_PAD", "200"))

    variants_full = variants.copy()
    variants = variants.copy()
    all_isolates = variants["isolate_id"].unique()

    if candidate_only:
        before_n = variants["mutation"].nunique()
        variants = candidate_region_filter(variants, gene_pad)
        print(f"  [features] candidate-gene only (±{gene_pad}bp): kept "
              f"{variants['mutation'].nunique():,}/{before_n:,} variants "
              "inside resistance genes")

    def _snp_matrix(vv: pd.DataFrame) -> pd.DataFrame:
        apply_floor = bool(min_count and min_count > 1)
        apply_cap = bool(max_features and max_features > 0)
        if apply_floor or apply_cap:
            counts = (vv.groupby("mutation")["isolate_id"].nunique()
                      .sort_values(ascending=False))
            before = len(counts)
            if apply_floor:
                counts = counts[counts >= min_count]
            if apply_cap and len(counts) > max_features:
                counts = counts.iloc[:max_features]
            keep = set(counts.index)
            vv = vv[vv["mutation"].isin(keep)]
            print(f"  [features] min_count={min_count}, "
                  f"max_features={max_features or 'none'}: kept "
                  f"{len(keep):,}/{before:,} mutations "
                  f"(dropped {before - len(keep):,})")
        vv = vv.copy()
        vv["present"] = 1
        Xs = (vv.pivot_table(index="isolate_id", columns="mutation",
                             values="present", aggfunc="max", fill_value=0)
              .astype(np.int8))
        Xs = Xs.reindex(sorted(all_isolates), fill_value=0).astype(np.int8)
        Xs.columns.name = None
        return Xs

    if mode == "burden":
        burden = gene_burden_matrix(variants_full, all_isolates)
        print(f"  [features] mode=burden: {burden.shape[1]} gene-burden features")
        return burden

    X = _snp_matrix(variants)
    if mode == "both":
        burden = gene_burden_matrix(variants_full, all_isolates)
        X = X.join(burden, how="left").fillna(0)
        X[burden.columns] = X[burden.columns].astype(np.int16)
        print(f"  [features] mode=both: {X.shape[1]:,} total "
              f"({X.shape[1] - burden.shape[1]:,} SNP + {burden.shape[1]} burden)")
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
    # Keep every phenotyped isolate. Isolates with no observed variants become
    # all-zero feature rows (valid data: e.g. wild-type pncA is genuinely
    # susceptible) rather than being silently dropped by an inner join. For
    # genome-wide data, where every isolate carries variants, y.index is already a
    # subset of X_full.index, so this is identical to the previous inner join.
    X = X_full.reindex(y.index, fill_value=0)
    return X, y, list(X.columns)


def available_drugs(data_dir: str) -> list[str]:
    raw = load_raw(data_dir)
    return sorted(raw["phenotypes"]["drug"].unique())


def label_matrix(
    data_dir: str, drugs: list[str] | None = None, X_full: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Build (X, Y, drugs) for multi-task learning.

    X : binary mutation matrix for every isolate that has >=1 phenotype.
    Y : wide label matrix (isolates x drugs), 1=R, 0=S, NaN where untested.
        The NaNs are the point: CRyPTIC (and the synthetic set) don't test every
        isolate for every drug, so the multi-task loss must be *masked*.
    """
    raw = load_raw(data_dir)
    if X_full is None:
        X_full = mutation_matrix(raw["variants"])
    ph = raw["phenotypes"]
    drugs = drugs or sorted(ph["drug"].unique())

    wide = (
        ph[ph["drug"].isin(drugs)]
        .assign(y=lambda d: d["phenotype"].map({"R": 1, "S": 0}))
        .pivot_table(index="isolate_id", columns="drug", values="y", aggfunc="max")
        .reindex(columns=drugs)
    )
    common = X_full.index.intersection(wide.index)
    X = X_full.loc[common]
    Y = wide.loc[common]
    return X, Y, drugs


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
