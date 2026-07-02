"""
Pick a balanced isolate subset for ONE drug, and emit their ENA accessions.

The variant step is the heavy part — you don't want to fetch genomes for all
~12k isolates on day one. This selects a manageable, class-balanced subset for a
single drug so you can get the full real-data pipeline working, then scale.

Outputs (into --out):
  subset_<drug>.txt       UNIQUEIDs, one per line
  subset_<drug>_ena.txt   ENA run accessions, one per line  (for FASTQ download
                          / TB-Profiler, or for narrowing a VCF fetch)

Example:
  python -m src.data.make_subset --drug Rifampicin --n 1000 \
      --phenotypes data/processed/phenotypes.csv \
      --reuse-table data/processed/CRyPTIC_reuse_table_20221019.csv
"""
from __future__ import annotations

import os
import argparse
import pandas as pd


def _find(df, *cands):
    for c in cands:
        if c in df.columns:
            return c
    return None


def select(phenotypes_path: str, reuse_table_path: str | None, drug: str,
           n: int = 1000, ratio: float = 1.0, seed: int = 42) -> pd.DataFrame:
    ph = pd.read_csv(phenotypes_path)
    sub = ph[ph["drug"].str.lower() == drug.lower()].copy()
    if sub.empty:
        raise SystemExit(f"No rows for drug '{drug}'. Available: "
                         f"{sorted(ph['drug'].unique())}")

    res = sub[sub["phenotype"] == "R"]
    sus = sub[sub["phenotype"] == "S"]
    # balance: cap susceptibles at ratio * #resistant, then cap total at n
    n_res = min(len(res), max(1, n // 2))
    n_sus = min(len(sus), int(n_res * ratio))
    picked = pd.concat([
        res.sample(n_res, random_state=seed),
        sus.sample(n_sus, random_state=seed),
    ])
    print(f"  {drug}: {len(res):,} R / {len(sus):,} S available "
          f"-> selected {n_res:,} R + {n_sus:,} S")

    # attach ENA accession from the reuse table if available
    if reuse_table_path and os.path.exists(reuse_table_path):
        rt = pd.read_csv(reuse_table_path, low_memory=False)
        id_col = _find(rt, "UNIQUEID")
        ena_col = _find(rt, "ENA_RUN", "ENA_SAMPLE", "ENA_RUN_ACCESSION")
        if id_col and ena_col:
            picked = picked.merge(rt[[id_col, ena_col]],
                                  left_on="isolate_id", right_on=id_col, how="left")
            picked = picked.rename(columns={ena_col: "ena"})
            print(f"  mapped ENA accessions via reuse table column '{ena_col}'")
        else:
            print("  (no ENA column found in reuse table; emitting IDs only)")
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description="Select a balanced isolate subset for one drug")
    ap.add_argument("--drug", required=True)
    ap.add_argument("--phenotypes", default="data/processed/phenotypes.csv")
    ap.add_argument("--reuse-table", default="data/processed/CRyPTIC_reuse_table_20221019.csv")
    ap.add_argument("--n", type=int, default=1000, help="approx. total isolates")
    ap.add_argument("--ratio", type=float, default=1.0, help="susceptible:resistant ratio")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    picked = select(args.phenotypes, args.reuse_table, args.drug, args.n, args.ratio)
    slug = args.drug.lower()

    ids_path = os.path.join(args.out, f"subset_{slug}.txt")
    picked["isolate_id"].drop_duplicates().to_csv(ids_path, index=False, header=False)
    print(f"  wrote {ids_path}  ({picked['isolate_id'].nunique():,} isolates)")

    if "ena" in picked.columns:
        ena_path = os.path.join(args.out, f"subset_{slug}_ena.txt")
        picked["ena"].dropna().drop_duplicates().to_csv(ena_path, index=False, header=False)
        print(f"  wrote {ena_path}  ({picked['ena'].nunique():,} accessions)")
        print("\nNext: fetch these genomes (FASTQ from ENA -> TB-Profiler, or the matching")
        print("VCFs), build variants.csv, then `python run_pipeline.py --data " + args.out + "`")


if __name__ == "__main__":
    main()
