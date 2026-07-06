"""
WHO 2023 mutation-catalogue baseline -- the "does ML beat the rulebook?" test.

The WHO catalogue is the clinical reference standard: a curated list of mutations
graded by their association with resistance. The obvious question for any ML
resistance predictor is whether it actually adds anything over simply looking each
mutation up in that catalogue. This module builds the rule-based predictor and
runs it head-to-head against the ML models on the same isolates.

Rule: an isolate is called RESISTANT to a drug if it carries ANY variant the WHO
catalogue grades "1) Assoc w R" or "2) Assoc w R - Interim" for that drug.

Matching is done on genomic coordinates. The catalogue's "Genomic_coordinates"
sheet expands every catalogued variant into one or more H37Rv
(position, ref, alt) triples; we reduce each isolate's variants to the same
triples and intersect. This sidesteps gene/amino-acid nomenclature entirely and
matches SNPs and indels alike.

Reported per drug: catalogue Sensitivity / Specificity / PPV, alongside the ML
model's AUC and operating point (read from a cross-validated metrics CSV) so the
two approaches sit side by side.
"""
from __future__ import annotations

import os
import re
import argparse
import numpy as np
import pandas as pd

from src.features.build_features import load_raw

R_GRADES = {"1) Assoc w R", "2) Assoc w R - Interim"}
GRADE_COL = "FINAL CONFIDENCE GRADING"


def _triple(token: str):
    """Mutation token -> (position:int, ref:str, alt:str), or None.
    Handles 'rpoB@761155_C>T', 'Rv0678@779100_G>GA', 'g3844992_T>A' and raw
    'NC_000962.3_761155_C_T'."""
    if "@" in token:
        rest = token.split("@", 1)[1]
    elif token[:1] == "g" and token[1:2].isdigit():
        rest = token[1:]
    else:
        p = token.rsplit("_", 3)
        if len(p) == 4 and p[1].isdigit():
            return (int(p[1]), p[2], p[3])
        return None
    m = re.match(r"(\d+)_([A-Za-z]+)>([A-Za-z]+)$", rest)
    return (int(m.group(1)), m.group(2), m.group(3)) if m else None


def _expand(pos: int, ref: str, alt: str) -> set:
    """Normalise a (pos, ref, alt) call to canonical per-position SNP triples.

    CRyPTIC emits codon-level changes as phased multi-nucleotide blocks (e.g.
    gyrA 7581 GACAG>GGCAC), while the WHO catalogue lists the same change as a
    single-position triple (7582 A>G). Splitting equal-length blocks into their
    differing positions makes the two representations match. Indels (unequal
    lengths) can't be split safely, so they're kept whole."""
    if len(ref) == len(alt):
        return {(pos + i, rb, ab) for i, (rb, ab) in enumerate(zip(ref, alt))
                if rb != ab}
    return {(pos, ref, alt)}


def load_catalogue(path: str) -> dict[str, set]:
    """drug -> set of canonical (pos, ref, alt) triples graded resistance-associated."""
    cat = pd.read_excel(path, sheet_name="Catalogue_master_file", header=2)
    cat = cat[["drug", "variant", GRADE_COL]].dropna(subset=["drug", "variant"])
    R = cat[cat[GRADE_COL].isin(R_GRADES)]

    coords = pd.read_excel(path, sheet_name="Genomic_coordinates", header=0)
    coords = coords.rename(columns={"reference_nucleotide": "ref",
                                    "alternative_nucleotide": "alt"})
    coords = coords.dropna(subset=["position", "ref", "alt", "variant"])
    coords["position"] = coords["position"].astype(int)
    var_to_triples: dict[str, set] = {}
    for v, d in coords.groupby("variant"):
        s: set = set()
        for p, r, a in zip(d["position"], d["ref"].astype(str), d["alt"].astype(str)):
            # Keep clean single-position SNPs and indels. SKIP equal-length
            # multi-nucleotide blocks: the catalogue encodes some codon changes
            # as phased blocks that bundle the causal SNP with linked
            # phylogenetic markers (e.g. gyrA D94 blocks also carry S95T).
            # Decomposing those would inject non-resistance markers and wreck
            # specificity. The causal SNP is always also present as a clean
            # single-position row, so nothing real is lost.
            if len(r) == len(a) == 1 or len(r) != len(a):
                s.add((int(p), r, a))
        var_to_triples[v] = s

    drug_R: dict[str, set] = {}
    for drug, grp in R.groupby("drug"):
        s = set()
        for v in grp["variant"]:
            s |= var_to_triples.get(v, set())
        drug_R[str(drug).strip()] = s
    return drug_R


def run(data_dir: str, catalogue: str, ml_metrics: str | None = None) -> pd.DataFrame:
    print("  loading WHO catalogue (this takes ~1 min)...")
    drug_R = load_catalogue(catalogue)
    print(f"  catalogue: resistance variants for {len(drug_R)} drugs")

    raw = load_raw(data_dir)
    v = raw["variants"].copy()
    v["triple"] = v["mutation"].map(_triple)
    v = v.dropna(subset=["triple"])

    # Decompose each isolate variant to canonical triples, but only keep those
    # that could match some catalogue entry (keeps per-isolate sets tiny and the
    # 14M-row pass fast). Expansion makes CRyPTIC's multi-nucleotide gyrA blocks
    # match the catalogue's per-position codon triples.
    all_cat = set().union(*drug_R.values()) if drug_R else set()
    hit: dict = {}
    for t in v["triple"].drop_duplicates():
        keep = _expand(*t) & all_cat
        if keep:
            hit[t] = keep
    rel = v[v["triple"].isin(hit.keys())]
    iso_triples = rel.groupby("isolate_id")["triple"].apply(
        lambda s: set().union(*(hit[t] for t in s)))
    ph = raw["phenotypes"]

    rows = []
    for drug in sorted(drug_R):
        Rset = drug_R[drug]
        sub = ph[ph["drug"] == drug]
        if not Rset or sub.empty:
            continue
        y = sub.set_index("isolate_id")["phenotype"].map({"R": 1, "S": 0}).dropna()
        if y.nunique() < 2:
            continue
        pred = y.index.map(lambda i: 1 if (iso_triples.get(i, set()) & Rset) else 0)
        yt, yp = y.values, np.array(pred)
        tp = int(((yp == 1) & (yt == 1)).sum()); fp = int(((yp == 1) & (yt == 0)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum()); fn = int(((yp == 0) & (yt == 1)).sum())
        rows.append({
            "Drug": drug, "n": len(yt), "%R": round(100 * yt.mean()),
            "cat_vars": len(Rset),
            "Cat_Sens": round(tp / (tp + fn), 3) if tp + fn else np.nan,
            "Cat_Spec": round(tn / (tn + fp), 3) if tn + fp else np.nan,
            "Cat_PPV": round(tp / (tp + fp), 3) if tp + fp else np.nan,
        })
    df = pd.DataFrame(rows)

    if ml_metrics and os.path.exists(ml_metrics):
        ml = pd.read_csv(ml_metrics).rename(columns={
            "AUC_xgb": "ML_AUC", "Sens@.5": "ML_Sens", "Spec@.5": "ML_Spec"})
        keep = [c for c in ["Drug", "ML_AUC", "ML_Sens", "ML_Spec"] if c in ml.columns]
        df = df.merge(ml[keep], on="Drug", how="left")
        if {"Cat_Sens", "ML_Sens"} <= set(df.columns):
            df["Sens_gain(ML-Cat)"] = (df["ML_Sens"] - df["Cat_Sens"]).round(3)
    return df.sort_values("Cat_Sens", ascending=False).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="WHO catalogue baseline vs ML")
    ap.add_argument("--data", default=os.path.join("data", "vcf_indel"))
    ap.add_argument("--catalogue", required=True, help="WHO-UCN-TB-*.xlsx path")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--ml-metrics", default=os.path.join("reports", "cv_metrics.csv"),
                    help="ML CV metrics CSV to compare against")
    args = ap.parse_args()

    print("=" * 78)
    print("WHO 2023 CATALOGUE BASELINE  |  rule-based vs ML, matched isolates")
    print("=" * 78)
    df = run(args.data, args.catalogue, args.ml_metrics)

    os.makedirs(args.reports, exist_ok=True)
    out = os.path.join(args.reports, "who_baseline.csv")
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\n" + df.to_string(index=False))
    print(f"\nSaved -> {out}")
    print("\nReading guide:")
    print("  Cat_Sens/Spec/PPV = WHO catalogue rule (carries any R-graded variant)")
    print("  ML_* = the ML model (AUC + operating point at 0.5)")
    print("  Sens_gain = extra resistant isolates ML catches over the catalogue at 0.5")


if __name__ == "__main__":
    main()
