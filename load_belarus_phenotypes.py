#!/usr/bin/env python
"""
Load Belarus (Wollenberg 2017 JCM; Broad umbrella PRJNA200335) per-isolate
phenotypes into the bel_ext holdout.

Source = supplementary Dataset S1 (xlsx). Two quirks the parser handles:
  * the real header is on the 3rd row (a title banner occupies rows 1-2);
  * S1 has TWO column blocks — block 1 is per-drug RESISTANCE (resistant/
    susceptible), block 2 is the TREATMENT regimen (dosages like "every day
    <1.6g>"). We read block 1 only, by taking the FIRST occurrence of each drug
    name (block 1 precedes block 2).
Isolate IDs use the Broad 'XTB##-###' alias; the 97 single-isolate-per-patient
set is flagged with a trailing '*'. We keep the starred 97 by default (using the
serial follow-up isolates too would be pseudo-replication) and map the XTB alias
-> SAMN sample accession via the ENA file report, because the profiled JSONs and
the bel_ext isolates are keyed by SAMN.

Usage:
    python -m src.data.load_belarus_phenotypes \
        --s1 data/belarus_supp/JCM.02116-16_zjm999095347sd2.xlsx \
        --ena data/processed/belarus_ena.tsv \
        --db data/tbamr.duckdb --dataset bel_ext
"""
from __future__ import annotations

import sys
import argparse
import warnings

import pandas as pd
import duckdb

warnings.filterwarnings("ignore")

DRUGS = ["INH", "RIF", "EMB", "PZA", "STR"]   # canonical; S1 already uses these codes


def load_s1(path: str):
    raw = pd.read_excel(path, header=None)
    hrow = None
    for i in range(min(8, len(raw))):
        if str(raw.iloc[i, 0]).strip() == "Isolate ID":
            hrow = i
            break
    if hrow is None:
        sys.exit("could not find an 'Isolate ID' header row in S1 (first column)")
    hdr = [str(x).strip() for x in raw.iloc[hrow].tolist()]
    df = raw.iloc[hrow + 1:].reset_index(drop=True)
    df.columns = range(df.shape[1])
    dcol = {}
    for d in DRUGS:
        idxs = [i for i, h in enumerate(hdr) if h == d]
        if not idxs:
            sys.exit(f"resistance column for {d} not found in S1 header")
        dcol[d] = idxs[0]                      # first occurrence = resistance block
    return df, dcol


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s1", required=True, help="path to Dataset S1 xlsx")
    ap.add_argument("--ena", required=True, help="belarus_ena.tsv (has sample_alias + sample_accession)")
    ap.add_argument("--db", default="data/tbamr.duckdb")
    ap.add_argument("--dataset", default="bel_ext")
    ap.add_argument("--all-isolates", action="store_true",
                    help="include all 138 isolates (default: starred 97 unique-patient set)")
    args = ap.parse_args()

    df, dcol = load_s1(args.s1)
    ids = df[0].astype(str).str.strip()
    starred = ids.str.endswith("*")
    keep = (df if args.all_isolates else df[starred]).reset_index(drop=True)
    n_set = len(keep)
    alias = keep[0].astype(str).str.strip().str.rstrip("*").str.strip()

    ena = pd.read_csv(args.ena, sep="\t", dtype=str)
    if "sample_alias" not in ena.columns or "sample_accession" not in ena.columns:
        sys.exit("belarus_ena.tsv must contain sample_alias and sample_accession columns")
    amap = (ena.dropna(subset=["sample_alias", "sample_accession"])
               .assign(a=lambda x: x["sample_alias"].str.strip())
               .drop_duplicates("a").set_index("a")["sample_accession"].to_dict())

    recs, unmatched = [], set()
    for r in range(len(keep)):
        samn = amap.get(alias[r])
        if samn is None:
            unmatched.add(alias[r])
            continue
        for d in DRUGS:
            v = str(keep.loc[r, dcol[d]]).strip().lower()
            if v == "resistant":
                recs.append((samn, d, "R"))
            elif v == "susceptible":
                recs.append((samn, d, "S"))
            # blank / nan => not tested => skip
    pheno = pd.DataFrame(recs, columns=["isolate_id", "drug", "phenotype"]).drop_duplicates()

    con = duckdb.connect(args.db)
    db_iso = {x[0] for x in con.execute(
        "SELECT isolate_id FROM isolates WHERE dataset_id = ?", [args.dataset]).fetchall()}
    in_db = pheno[pheno["isolate_id"].isin(db_iso)].copy()
    con.register("ph_new", in_db)
    con.execute("""DELETE FROM phenotypes
                   WHERE isolate_id IN (SELECT isolate_id FROM ph_new)
                     AND drug IN (SELECT DISTINCT drug FROM ph_new)""")
    con.execute("INSERT INTO phenotypes SELECT isolate_id, drug, phenotype FROM ph_new")
    con.unregister("ph_new")
    con.close()

    print("=" * 64)
    print(f"BELARUS PHENOTYPES -> {args.dataset}")
    print("=" * 64)
    print(f"isolates read from S1 {'(ALL 138)' if args.all_isolates else '(starred 97)'}: {n_set}")
    print(f"mapped XTB alias -> SAMN via ENA : {n_set - len(unmatched)}/{n_set}")
    if unmatched:
        ex = sorted(unmatched)[:8]
        print(f"  UNMATCHED aliases ({len(unmatched)}): {ex}{' ...' if len(unmatched) > 8 else ''}")
    print(f"of those, present in '{args.dataset}' (profiled & ingested): {in_db['isolate_id'].nunique()}")
    not_prof = set(pheno['isolate_id']) - db_iso
    if not_prof:
        ex = sorted(not_prof)[:6]
        print(f"  mapped but NOT ingested ({len(not_prof)}): {ex}{' ...' if len(not_prof) > 6 else ''}")
    print("\nphenotypes loaded (in-DB isolates), per drug:")
    for d in DRUGS:
        vc = in_db[in_db["drug"] == d]["phenotype"].value_counts()
        print(f"  {d}: {int(vc.get('R', 0))}R / {int(vc.get('S', 0))}S")
    print(f"\nNext: BioSample-level leakage check, then evaluate_holdout on {args.dataset}")


if __name__ == "__main__":
    main()
