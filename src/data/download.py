"""
Fetch REAL M. tuberculosis AMR data from the CRyPTIC consortium and reshape it
into the three CSVs the pipeline expects (variants / phenotypes / lineages).

>>> RUN THIS ON YOUR OWN MACHINE <<<  EBI's FTP is not reachable from the dev
sandbox, but it is from your laptop. The reshaping logic below was validated
against mock CRyPTIC-format files.

SOURCE  (CRyPTIC June-2022 release, fully open, no login)
---------------------------------------------------------
Root:   https://ftp.ebi.ac.uk/pub/databases/cryptic/release_june2022/
        +-- reuse/            <- general-purpose tables (use this)
        +-- reproducibility/  <- paper-reproduction artefacts

Key file (phenotypes + metadata):
    reuse/CRyPTIC_reuse_table_20221019.csv
    -> binary R/S phenotypes, MICs, phenotype-quality flags, lineage, and
       ENA/UNIQUE ids for ~12,288 isolates across 13 drugs.
    (Note: pyrazinamide is NOT on the CRyPTIC plate, so it won't appear -- the
     13 drugs are INH, RIF, EMB, RFB, LEV, MXF, AMI, KAN, ETH, BDQ, LZD, CFZ, DLM.)

Variants:
    Per-isolate VCFs live under reuse/. They are large; the genome is H37Rv
    (NC_000962.3). Browse the reuse/ directory to find the VCF archive, download
    it, and point `variants` at the unpacked folder.

TWO-STEP USAGE
--------------
    # 1. light & reliable -- real labels in seconds
    python -m src.data.download phenotypes --out data/processed

    # 2. heavier -- the variant matrix (start with a SUBSET, see notes)
    python -m src.data.download variants --vcf-dir /path/to/vcfs --out data/processed

    python run_pipeline.py --data data/processed
"""
from __future__ import annotations

import os
import re
import gzip
import argparse
import pandas as pd

try:
    import requests
except ImportError:  # downloader is optional if you already have the file
    requests = None

CRYPTIC_ROOT = "https://ftp.ebi.ac.uk/pub/databases/cryptic/release_june2022/reuse/"
REUSE_TABLE = "CRyPTIC_reuse_table_20221019.csv"

# CRyPTIC drug abbreviations -> full names
DRUG_NAMES = {
    "INH": "Isoniazid", "RIF": "Rifampicin", "EMB": "Ethambutol", "RFB": "Rifabutin",
    "LEV": "Levofloxacin", "MXF": "Moxifloxacin", "AMI": "Amikacin", "KAN": "Kanamycin",
    "ETH": "Ethionamide", "BDQ": "Bedaquiline", "LZD": "Linezolid", "CFZ": "Clofazimine",
    "DLM": "Delamanid",
}
H37RV = "NC_000962.3"


# ---------------------------------------------------------------------------
# Download helpers (run on your machine)
# ---------------------------------------------------------------------------
def download_file(url: str, dest: str, chunk: int = 1 << 20) -> str:
    if requests is None:
        raise RuntimeError("pip install requests, or pass --reuse-table to a local copy")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"  downloading {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for block in r.iter_content(chunk_size=chunk):
                f.write(block)
                done += len(block)
                if total:
                    print(f"\r    {done/1e6:6.1f} / {total/1e6:.1f} MB", end="")
        print()
    return dest


# ---------------------------------------------------------------------------
# Reshape: reuse table -> phenotypes.csv + lineages.csv   (VALIDATED on mocks)
# ---------------------------------------------------------------------------
def _find_id_column(df: pd.DataFrame) -> str:
    for cand in ("UNIQUEID", "unique_id", "ENA_RUN", "ENA_SAMPLE", "sample_id"):
        if cand in df.columns:
            return cand
    # fallback: first column that looks like an identifier
    for c in df.columns:
        if df[c].dtype == object and df[c].nunique() == len(df):
            return c
    return df.columns[0]


def _find_lineage_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if "lineage" in c.lower():
            return c
    return None


def _phenotype_columns(df: pd.DataFrame) -> dict[str, str]:
    """Return {drug_full_name: column}. Handles *_BINARY_PHENOTYPE and value-sniffing."""
    out: dict[str, str] = {}
    pat = re.compile(r"^([A-Za-z]{3})[_-].*phenotype", re.I)
    for c in df.columns:
        m = pat.match(c)
        if m and "qual" not in c.lower():
            abbr = m.group(1).upper()
            out[DRUG_NAMES.get(abbr, abbr.title())] = c
    if out:
        return out
    # fallback: any column whose non-null values are a subset of {R, S}
    for c in df.columns:
        vals = set(str(v).upper() for v in df[c].dropna().unique())
        if vals and vals <= {"R", "S"}:
            abbr = c.split("_")[0].split("-")[0].upper()
            out[DRUG_NAMES.get(abbr, abbr.title())] = c
    return out


def _quality_column_for(df: pd.DataFrame, pheno_col: str) -> str | None:
    prefix = re.split(r"[_-]", pheno_col)[0]
    for c in df.columns:
        if c.lower().startswith(prefix.lower()) and "qual" in c.lower():
            return c
    return None


def build_phenotypes_and_lineages(
    reuse_table_path: str, out_dir: str, quality: str | None = "HIGH"
) -> dict[str, str]:
    """Reshape the CRyPTIC reuse table into long phenotypes.csv + lineages.csv."""
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(reuse_table_path, low_memory=False)
    print(f"  reuse table: {df.shape[0]:,} rows x {df.shape[1]} cols")

    id_col = _find_id_column(df)
    pheno_cols = _phenotype_columns(df)
    lin_col = _find_lineage_column(df)
    print(f"  id column      : {id_col}")
    print(f"  lineage column : {lin_col}")
    print(f"  drug phenotype columns detected: {len(pheno_cols)}")
    for drug, col in pheno_cols.items():
        print(f"      {drug:<13} <- {col}")

    # melt to long, applying per-drug quality filter if requested
    records = []
    for drug, col in pheno_cols.items():
        sub = df[[id_col, col]].copy()
        sub.columns = ["isolate_id", "phenotype"]
        if quality:
            qcol = _quality_column_for(df, col)
            if qcol:
                keep = df[qcol].astype(str).str.upper() == quality.upper()
                sub = sub[keep.values]
        sub["drug"] = drug
        sub["phenotype"] = sub["phenotype"].astype(str).str.upper()
        sub = sub[sub["phenotype"].isin(["R", "S"])]
        records.append(sub[["isolate_id", "drug", "phenotype"]])

    phenotypes = pd.concat(records, ignore_index=True).dropna()
    p_path = os.path.join(out_dir, "phenotypes.csv")
    phenotypes.to_csv(p_path, index=False)
    print(f"  wrote {p_path}  ({len(phenotypes):,} rows, "
          f"{phenotypes['isolate_id'].nunique():,} isolates)")

    paths = {"phenotypes": p_path}
    if lin_col:
        lineages = df[[id_col, lin_col]].copy()
        lineages.columns = ["isolate_id", "lineage"]
        lineages = lineages.dropna().drop_duplicates("isolate_id")
        l_path = os.path.join(out_dir, "lineages.csv")
        lineages.to_csv(l_path, index=False)
        print(f"  wrote {l_path}  ({len(lineages):,} rows)")
        paths["lineages"] = l_path
    return paths


# ---------------------------------------------------------------------------
# Reshape: VCF directory -> variants.csv             (VALIDATED on mock VCFs)
# ---------------------------------------------------------------------------
def _open_maybe_gzip(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def parse_vcf(path: str, isolate_id: str, snps_only: bool = True,
              pass_only: bool = True) -> list[tuple[str, str]]:
    """Extract (isolate_id, mutation_token) from one VCF.

    Tokens are position-based (e.g. 'NC_000962.3_761155_C_T'). That's a fully
    automatable, catalogue-free representation -- you can later annotate the
    resistance-relevant ones to gene/aa notation (rpoB_S450L) with the WHO
    catalogue or TB-Profiler. For an MVP, position tokens train fine.
    """
    rows: list[tuple[str, str]] = []
    with _open_maybe_gzip(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            chrom, pos, _id, ref, alt = parts[0], parts[1], parts[2], parts[3], parts[4]
            filt = parts[6] if len(parts) > 6 else "."
            if pass_only and filt not in (".", "PASS"):
                continue
            for a in alt.split(","):
                if a in (".", "*"):
                    continue
                if snps_only and (len(ref) != 1 or len(a) != 1):
                    continue
                rows.append((isolate_id, f"{chrom}_{pos}_{ref}_{a}"))
    return rows


def build_variants_from_vcf_dir(
    vcf_dir: str, out_dir: str, id_from: str = "filename", **kw
) -> str:
    """Parse every VCF in a directory into a long variants.csv."""
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isdir(vcf_dir):
        raise SystemExit(
            f"VCF directory not found: '{vcf_dir}'.\n"
            "Download the CRyPTIC VCFs first (browse the reuse/ directory in a "
            "browser), then pass the real unpacked folder path to --vcf-dir."
        )
    vcfs = [f for f in os.listdir(vcf_dir) if f.endswith((".vcf", ".vcf.gz"))]
    if not vcfs:
        raise FileNotFoundError(f"No .vcf/.vcf.gz files in {vcf_dir}")
    print(f"  parsing {len(vcfs):,} VCF files")
    all_rows: list[tuple[str, str]] = []
    for i, fn in enumerate(sorted(vcfs)):
        iid = re.sub(r"\.vcf(\.gz)?$", "", fn)
        all_rows.extend(parse_vcf(os.path.join(vcf_dir, fn), iid, **kw))
        if (i + 1) % 250 == 0:
            print(f"\r    {i+1:,}/{len(vcfs):,}", end="")
    print()
    variants = pd.DataFrame(all_rows, columns=["isolate_id", "mutation"]).drop_duplicates()
    v_path = os.path.join(out_dir, "variants.csv")
    variants.to_csv(v_path, index=False)
    print(f"  wrote {v_path}  ({len(variants):,} rows, "
          f"{variants['isolate_id'].nunique():,} isolates, "
          f"{variants['mutation'].nunique():,} distinct variants)")
    return v_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch & reshape real CRyPTIC TB-AMR data")
    sub = ap.add_subparsers(dest="step", required=True)

    p1 = sub.add_parser("phenotypes", help="download reuse table -> phenotypes + lineages")
    p1.add_argument("--out", default="data/processed")
    p1.add_argument("--reuse-table", default=None,
                    help="path to an already-downloaded reuse table (skips download)")
    p1.add_argument("--quality", default="HIGH",
                    help="keep only this phenotype quality; pass '' to keep all")

    p2 = sub.add_parser("variants", help="parse a directory of VCFs -> variants")
    p2.add_argument("--vcf-dir", required=True)
    p2.add_argument("--out", default="data/processed")
    p2.add_argument("--keep-indels", action="store_true")

    args = ap.parse_args()

    if args.step == "phenotypes":
        table = args.reuse_table
        if table is None:
            table = download_file(CRYPTIC_ROOT + REUSE_TABLE,
                                  os.path.join(args.out, REUSE_TABLE))
        build_phenotypes_and_lineages(table, args.out,
                                      quality=(args.quality or None))
        print("\nNext: get the VCFs, then `download.py variants --vcf-dir ...`")
    elif args.step == "variants":
        build_variants_from_vcf_dir(args.vcf_dir, args.out,
                                    snps_only=not args.keep_indels)
        print("\nNext: python run_pipeline.py --data", args.out)


if __name__ == "__main__":
    main()
