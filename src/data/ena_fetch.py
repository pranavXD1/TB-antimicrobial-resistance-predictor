"""
Resolve ENA *sample* accessions -> run accessions + FASTQ download URLs.

>>> RUN ON YOUR MACHINE <<< (queries ENA's public API).

`make_subset` emitted sample accessions (SAMEA…), but reads are downloaded per
*run* (ERR…), and one sample can have several runs. This queries ENA's
filereport API to expand each sample into its runs and FASTQ URLs, writing a
manifest you can feed to a downloader and then to TB-Profiler.

    python -m src.data.ena_fetch --accessions data/processed/subset_rifampicin_ena.txt \
        --out data/processed/fastq_manifest.tsv

The TSV-parsing logic was validated on a mock ENA response in the dev sandbox;
the live HTTP call runs from your machine.
"""
from __future__ import annotations

import os
import csv
import time
import argparse

try:
    import requests
except ImportError:
    requests = None

ENA_API = "https://www.ebi.ac.uk/ena/portal/api/filereport"
FIELDS = "run_accession,sample_accession,fastq_ftp,fastq_bytes,read_count"


def parse_filereport_tsv(text: str) -> list[dict]:
    """Parse ENA filereport TSV into rows; split the ;-separated FASTQ URLs."""
    lines = [ln for ln in text.strip().splitlines() if ln]
    if len(lines) < 2:
        return []
    header = lines[0].split("\t")
    out = []
    for ln in lines[1:]:
        rec = dict(zip(header, ln.split("\t")))
        ftp = rec.get("fastq_ftp", "") or ""
        # ENA returns ftp.sra.ebi.ac.uk/... without scheme; prepend https://
        rec["fastq_urls"] = [
            ("https://" + u) if not u.startswith("http") else u
            for u in ftp.split(";") if u
        ]
        out.append(rec)
    return out


def resolve(accession: str, session) -> list[dict]:
    r = session.get(ENA_API, params={
        "accession": accession, "result": "read_run",
        "fields": FIELDS, "format": "tsv",
    }, timeout=60)
    r.raise_for_status()
    return parse_filereport_tsv(r.text)


def main() -> None:
    ap = argparse.ArgumentParser(description="Resolve ENA samples -> runs + FASTQ URLs")
    ap.add_argument("--accessions", required=True, help="file of accessions, one per line")
    ap.add_argument("--out", default="data/processed/fastq_manifest.tsv")
    args = ap.parse_args()
    if requests is None:
        raise SystemExit("pip install requests")

    accs = [a.strip() for a in open(args.accessions) if a.strip()]
    print(f"resolving {len(accs):,} accessions via ENA...")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    session = requests.Session()
    n_runs = 0
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["query_accession", "run_accession", "sample_accession", "fastq_url"])
        for i, acc in enumerate(accs):
            try:
                for rec in resolve(acc, session):
                    for url in rec["fastq_urls"]:
                        w.writerow([acc, rec.get("run_accession", ""),
                                    rec.get("sample_accession", ""), url])
                        n_runs += 1
            except Exception as e:  # keep going; log the failure
                print(f"  [warn] {acc}: {e}")
            if (i + 1) % 100 == 0:
                print(f"\r  {i+1:,}/{len(accs):,}", end="")
            time.sleep(0.05)  # be polite to the API
    print(f"\nwrote {args.out}  ({n_runs:,} FASTQ URLs)")
    print("Next: download those FASTQs, run TB-Profiler, then "
          "`python -m src.data.tbprofiler_aggregate`")


if __name__ == "__main__":
    main()
