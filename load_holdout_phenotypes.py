#!/usr/bin/env python
"""
Load Schleusener et al. (srep46327) Table S2 phenotypic DST labels into the DuckDB
``sl_ext`` holdout dataset.

The paper keys strains by an ENA ``ERS`` secondary-sample accession; your ``sl_ext``
isolates are keyed by the ``SAMEA`` (BioSample) accession. These are two aliases for
the same ENA sample, so we bridge them with the ERS<->SAMEA crosswalk that already
lives in your ENA manifest (``sl_manifest.tsv``). Columns are auto-detected by value
prefix (``ERS...`` / ``SAM...``), so exact header names don't matter.

Loads R/S labels for INH, RIF, EMB, PZA, STR into the ``phenotypes`` table for every
``sl_ext`` isolate we can map, idempotently (existing sl_ext phenotypes are cleared
first). Reports coverage: how many isolates got labels, per-drug R/S counts, and any
isolates that have variants but no phenotype.

Usage:
    python -m src.data.load_holdout_phenotypes \
        --pheno schleusener_pheno.csv \
        --manifest data/processed/sl_manifest.tsv \
        --db data/tbamr.duckdb --dataset sl_ext
"""
from __future__ import annotations

import sys
import argparse

import pandas as pd
import duckdb


def col_by_prefix(df: pd.DataFrame, prefix: str, min_frac: float = 0.5):
    """Return the column whose values most-often start with `prefix` (name-independent)."""
    best, best_frac = None, 0.0
    for c in df.columns:
        vals = df[c].astype(str).str.strip()
        frac = float(vals.str.startswith(prefix).mean())
        if frac > best_frac:
            best, best_frac = c, frac
    return best if best_frac >= min_frac else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pheno", required=True, help="schleusener_pheno.csv (ers,key,drug,phenotype)")
    ap.add_argument("--manifest", default="data/processed/sl_manifest.tsv")
    ap.add_argument("--db", default="data/tbamr.duckdb")
    ap.add_argument("--dataset", default="sl_ext")
    args = ap.parse_args()

    ph = pd.read_csv(args.pheno)
    ph["ers"] = ph["ers"].astype(str).str.strip()

    man = pd.read_csv(args.manifest, sep=None, engine="python")
    print("manifest columns:", list(man.columns))

    ers_col = col_by_prefix(man, "ERS")
    samea_col = col_by_prefix(man, "SAM")
    print(f"detected ERS column: {ers_col!r} ; SAMEA column: {samea_col!r}")
    if ers_col is None or samea_col is None:
        print("\n[!] Could not auto-detect both an ERS (secondary_sample_accession) and a "
              "SAMEA (sample_accession) column in the manifest by value prefix.")
        print("    Paste `head -3 " + args.manifest + "` and I'll wire the columns in directly,")
        print("    or we pull the ERS<->SAMEA crosswalk from ENA another way.")
        sys.exit(2)

    cross = man[[ers_col, samea_col]].copy()
    cross.columns = ["ers", "samea"]
    cross["ers"] = cross["ers"].astype(str).str.strip()
    cross["samea"] = cross["samea"].astype(str).str.strip()
    cross = cross[cross["ers"].str.startswith("ERS")].drop_duplicates()

    merged = ph.merge(cross, on="ers", how="left")
    n_unmapped = int(merged["samea"].isna().sum())
    print(f"\npaper strain-drug labels : {len(ph)}")
    print(f"  mapped ERS->SAMEA      : {int(merged['samea'].notna().sum())}")
    if n_unmapped:
        ex = merged.loc[merged['samea'].isna(), 'ers'].unique()[:6]
        print(f"  UNMAPPED ERS           : {n_unmapped}  e.g. {list(ex)}")

    load = (merged.dropna(subset=["samea"])[["samea", "drug", "phenotype"]]
                  .drop_duplicates())

    con = duckdb.connect(args.db)
    try:
        sl_iso = {r[0] for r in con.execute(
            "SELECT isolate_id FROM isolates WHERE dataset_id = ?", [args.dataset]).fetchall()}
        print(f"\nsl_ext isolates in DB    : {len(sl_iso)}")
        before = len(load)
        load = load[load["samea"].isin(sl_iso)]
        print(f"labels matching a loaded isolate: {len(load)} (dropped {before - len(load)} "
              f"for samples not in your profiled set)")

        # idempotent
        con.execute("DELETE FROM phenotypes WHERE isolate_id IN "
                    "(SELECT isolate_id FROM isolates WHERE dataset_id = ?)", [args.dataset])
        con.register("load_df", load)
        con.execute("INSERT INTO phenotypes (isolate_id, drug, phenotype) "
                    "SELECT samea, drug, phenotype FROM load_df")

        total = con.execute(
            "SELECT COUNT(*) FROM phenotypes p JOIN isolates i ON p.isolate_id=i.isolate_id "
            "WHERE i.dataset_id = ?", [args.dataset]).fetchone()[0]
        print(f"\nloaded {total} phenotype labels into '{args.dataset}'")

        print("\nper-drug labels on isolates that ALSO have variants (i.e. scorable):")
        rows = con.execute("""
            SELECT p.drug,
                   SUM(CASE WHEN p.phenotype='R' THEN 1 ELSE 0 END) AS R,
                   SUM(CASE WHEN p.phenotype='S' THEN 1 ELSE 0 END) AS S
            FROM phenotypes p
            JOIN isolates i ON p.isolate_id = i.isolate_id
            WHERE i.dataset_id = ?
              AND p.isolate_id IN (SELECT DISTINCT isolate_id FROM variants)
            GROUP BY p.drug ORDER BY p.drug
        """, [args.dataset]).fetchall()
        for drug, r, s in rows:
            print(f"  {drug:4s}  R={int(r):3d}  S={int(s):3d}  (n={int(r)+int(s)})")

        gap = con.execute("""
            SELECT COUNT(*) FROM isolates i
            WHERE i.dataset_id = ?
              AND i.isolate_id NOT IN (SELECT DISTINCT isolate_id FROM phenotypes)
        """, [args.dataset]).fetchone()[0]
        print(f"\nsl_ext isolates with variants but NO phenotype: {gap}")
        print("Next: score the frozen CRyPTIC (INH/RIF/EMB) and GenTB (PZA/STR) models on sl_ext.")
    finally:
        con.close()


if __name__ == "__main__":
    main()
