# Data

This project uses only publicly available data. The repository ships **code and
small derived artifacts** (reports, manifests, ENA crosswalks); the large inputs
and built artifacts are **not** committed — obtain them either by regenerating
from the public accessions below, or by downloading the archived bundle from
Zenodo.

## Quick options

| You want to… | Do this |
|---|---|
| Reproduce everything from scratch | Follow **Regenerating from source** below (needs TB-Profiler + the pipeline) |
| Just run/inspect the trained models and results | Download the **Zenodo bundle** (models + built database) |

**Zenodo bundle:** `models/` (trained XGBoost + calibrators), the built
`data/tbamr.duckdb`, and the CRyPTIC `variants.csv` tables.
DOI: `<add your Zenodo DOI here after upload>`

## Source datasets (all public)

### Training
- **CRyPTIC** — 12,288 isolates with UKMYC broth-microdilution MICs + WGS.
  Reuse table `CRyPTIC_reuse_table_20221019.csv` and variant/VCF release are
  public (CRyPTIC consortium / FigShare + ENA). Used to train the first-line
  models (INH, RIF, EMB).
- **GenTB** — per-drug genotype→phenotype matrices (Farhat lab). Public. Used to
  train the pyrazinamide and streptomycin models (drugs not on the UKMYC plate).

### External validation cohorts (holdouts, never used in training)
| Cohort | Accession | Reads | Phenotypes (per-isolate DST) |
|---|---|---|---|
| Sierra Leone | **PRJEB7727** | ENA FASTQ | Schleusener et al. 2017, *Sci Rep* 7:46327 — Supp. Table S2 |
| Belarus | **PRJNA200335** | NCBI SRA FASTQ (filter to `XTB…` aliases) | Wollenberg et al. 2017, *J Clin Microbiol* 55:457 — Dataset S1 |
| Latvia (staged, unscored) | **PRJEB59824** | ENA FASTQ | per-isolate DST not obtainable; genotype-only |

## Regenerating from source

Prerequisites: the `tb` conda environment (TB-Profiler, XGBoost, DuckDB, pandas).

1. **Training data → database.** Load the CRyPTIC reuse table and variant tables
   and the GenTB matrices into `data/tbamr.duckdb` (DuckDB streams the ~14M-row
   CRyPTIC variant CSV; do not load it into RAM). This regenerates
   `data/vcf_indel/variants.csv`, `data/vcf/variants.csv`, and the database.
2. **Train the models** (produces `models/`, `models_gentb_pza/`,
   `models_gentb_str/`). Weights are frozen after this step — external validation
   must not retrain.
3. **External cohorts.** For each accession above: pull the ENA/SRA file report,
   build a FASTQ manifest, then run the pipeline
   (`run_profiling` → `holdout_ingest` → phenotype loader → `evaluate_holdout`).
   See the External Validation section of `RESULTS.md` for the exact commands.

## What lives where

| Location | Contents |
|---|---|
| **GitHub** | `src/`, `models_gentb_pza/`, `models_gentb_str/`, `reports*/`, `data/processed/*` (manifests, crosswalks), `RESULTS.md`, `DATA.md` |
| **Zenodo** | `models/`, `data/tbamr.duckdb`, `data/vcf*/variants.csv` |
| **Public archives** | raw FASTQ (ENA/SRA), CRyPTIC + GenTB source releases |

## Note on identifiers

CRyPTIC uses ENA accessions (`ERS…`); Belarus uses NCBI accessions
(`SAMN…`/`SRS…`/`SRR…`). These are different namespaces for the same INSDC data,
which is why the Belarus leakage check relies on provenance rather than accession
intersection (see `RESULTS.md` → Data integrity).
