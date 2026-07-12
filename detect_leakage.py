#!/usr/bin/env python
"""
Formal leakage check — is any Sierra Leone holdout sample also in CRyPTIC training?

The M. africanum result already implies no overlap (CRyPTIC is almost entirely
L2/L4), but this makes it explicit and airtight. CRyPTIC isolate IDs are internal
UNIQUEIDs, so we bridge through ENA accessions: pull every ENA/SRA accession
(ERR/ERS/SRR/SAMEA/SAMN/...) out of the CRyPTIC reuse table, pull the Sierra
Leone accessions from the ENA crosswalk, and intersect. Run-level (ERR) overlap
is definitive (same sequencing run = same data); sample-level is the backstop.

Zero overlap => sl_ext is genuinely external and the AUCs are real external
validation, not memorised train/test overlap.

Usage:
    python -m src.data.detect_leakage \
        --cryptic-meta data/processed/CRyPTIC_reuse_table_20221019.csv \
        --sl-crosswalk data/processed/ena_crosswalk.tsv \
        --sl-extra data/sl_ext/ena_samples.tsv
"""
from __future__ import annotations

import re
import sys
import argparse

import pandas as pd

ACC = re.compile(r"^(ERR|ERS|ERX|SRR|SRS|SRX|DRR|DRS|DRX|SAMEA|SAMN|SAMD)\d+$")


def load_any(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", dtype=str)


def accession_columns(df: pd.DataFrame) -> dict:
    """{column -> set of ENA/SRA-accession-shaped values} for any column that has them."""
    found = {}
    for c in df.columns:
        vals = df[c].dropna().astype(str).str.strip()
        hits = {v for v in vals if ACC.match(v)}
        if hits:
            found[c] = hits
    return found


def collect(path: str, label: str):
    df = load_any(path)
    cols = accession_columns(df)
    acc = set().union(*cols.values()) if cols else set()
    print(f"{label}: {df.shape[0]} rows, {df.shape[1]} cols")
    for c, s in cols.items():
        ex = ", ".join(sorted(s)[:3])
        print(f"    accession column {c!r}: {len(s)} unique  (e.g. {ex})")
    if not cols:
        print(f"    [!] no accession-shaped values found; columns = {list(df.columns)[:20]}")
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cryptic-meta", default="data/processed/CRyPTIC_reuse_table_20221019.csv")
    ap.add_argument("--sl-crosswalk", default="data/processed/ena_crosswalk.tsv")
    ap.add_argument("--sl-extra", default=None, help="optional extra SL accession file")
    args = ap.parse_args()

    print("=" * 72)
    print("LEAKAGE CHECK — CRyPTIC training  vs  Sierra Leone holdout (ENA accessions)")
    print("=" * 72)

    cry = collect(args.cryptic_meta, "CRyPTIC reuse table")
    if not cry:
        print("\nCould not find ENA accessions in the CRyPTIC table — paste its header "
              "and I'll point at the right column.")
        sys.exit(2)

    print()
    sl = collect(args.sl_crosswalk, "Sierra Leone crosswalk")
    if args.sl_extra:
        try:
            sl |= collect(args.sl_extra, "Sierra Leone extra")
        except Exception as e:
            print(f"    (skipped --sl-extra: {e})")

    overlap = cry & sl
    print("\n" + "=" * 72)
    print(f"CRyPTIC accessions: {len(cry):>6}   |   Sierra Leone accessions: {len(sl):>4}")
    print(f"SHARED ACCESSIONS : {len(overlap)}")
    print("=" * 72)
    if overlap:
        print("  *** LEAKAGE DETECTED *** — present in BOTH datasets:")
        for a in sorted(overlap):
            print("     ", a)
        print("\n  These isolates must be removed from the holdout before the external")
        print("  AUCs can be reported.")
    else:
        print("  PASS — zero shared ENA accessions.")
        print("  sl_ext shares no sequencing run or sample with CRyPTIC training; combined")
        print("  with ~0 M. africanum in CRyPTIC, the external validation is clean.")


if __name__ == "__main__":
    main()
