"""
Real-data acquisition for TB AMR.

>>> RUN THIS ON YOUR OWN MACHINE <<<  (the dev sandbox can't reach these hosts).

Goal: produce the *same three CSVs* the synthetic generator does --
variants.csv, phenotypes.csv, lineages.csv -- in data/processed/, so
`run_pipeline.py --data data/processed` works with zero code changes.

PRIMARY SOURCE -- CRyPTIC consortium
------------------------------------
The CRyPTIC project released WGS + MIC/binary phenotypes for ~15-20k M. tuberculosis
isolates across 13 drugs. It's the gold-standard public TB-AMR dataset.
  * Data portal / figshare:  https://www.crypticproject.org/  (follow "data")
  * The reuse paper describes the released tables (genotypes, phenotypes, MICs).
Download the genotype (variant) table and the binary phenotype table, then map
them into the long format below.

SECONDARY / ALTERNATIVES
------------------------
  * NCBI Pathogen Detection (AMR metadata + assemblies):
        https://www.ncbi.nlm.nih.gov/pathogens/
  * BV-BRC (formerly PATRIC) AMR phenotypes:
        https://www.bv-brc.org/
  * WHO mutation catalogue (v2, 2023) -- the reference set of resistance-
    associated mutations; use it to *name/validate* features and as a
    rule-based baseline to beat:
        https://www.who.int/publications/i/item/9789240082410
  * TB-Profiler (tool + curated resistance DB, great for calling variants from
    FASTQ/BAM and as a comparison baseline):
        https://github.com/jodyphelan/TBProfiler

TARGET SCHEMA (what build_features expects)
-------------------------------------------
  variants.csv    : isolate_id, mutation        # gene_aaPos notation, e.g. rpoB_S450L
  phenotypes.csv  : isolate_id, drug, phenotype # phenotype in {R, S}
  lineages.csv    : isolate_id, lineage         # optional metadata

IMPLEMENTATION NOTES
--------------------
* Normalise mutation names to a consistent scheme (gene_ref+pos+alt). Use the
  WHO catalogue / TB-Profiler nomenclature so SHAP outputs are human-readable.
* Keep only isolates with both a genotype and >=1 phenotype.
* Map CRyPTIC binary phenotypes (R/S) straight through; if you start from MICs,
  threshold against the published ECOFF/breakpoints.
"""
from __future__ import annotations

import os
import argparse


CRYPTIC_PORTAL = "https://www.crypticproject.org/"
WHO_CATALOGUE = "https://www.who.int/publications/i/item/9789240082410"


def fetch_cryptic(out_dir: str) -> None:
    """
    Placeholder. Fill in once you've chosen the exact CRyPTIC release files.

    Suggested flow:
      1. Download the genotype table + binary-phenotype table from the portal.
      2. Reshape genotypes -> long (isolate_id, mutation).
      3. Reshape phenotypes -> long (isolate_id, drug, phenotype) with R/S.
      4. Save both to `out_dir`.
    """
    raise NotImplementedError(
        "Download CRyPTIC tables from "
        f"{CRYPTIC_PORTAL} and map them to the target schema. "
        "See the module docstring for the exact column layout."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire real TB-AMR data")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--source", default="cryptic", choices=["cryptic"])
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.source == "cryptic":
        print(f"See module docstring. Reference catalogue: {WHO_CATALOGUE}")
        fetch_cryptic(args.out)


if __name__ == "__main__":
    main()
