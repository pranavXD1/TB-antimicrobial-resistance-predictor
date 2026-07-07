"""
Scale the dataset WITHOUT TB-Profiler: download CRyPTIC per-sample VCFs directly.

The reuse table already lists a VCF path for every isolate, and a per-sample
"masked" VCF is a few hundred KB of text -- versus the 200-300 MB FASTQs that
made profiling slow and flaky. Parsing is pure Python (no bwa/samtools/Java), so
this scales to thousands of isolates in one modest, resumable download.

Recipe (mirrors arXiv:2509.25216, who did this for isoniazid):
  * keep only HIGH phenotype-quality labels (agreement across >=2 MIC assays)
  * take a stratified, class-balanced subset on an anchor drug
  * drop INDELs and non-PASS/missing loci
  * represent each isolate by the SNPs it carries (position tokens), then
    annotate tokens that fall in known resistance genes so SHAP reads e.g.
    'rpoB@761155_C>T' instead of a bare coordinate.

Output is the SAME variants.csv / phenotypes.csv schema the pipeline already
eats, so run_pipeline.py and evaluate_cv.py run unchanged on the bigger data.

USAGE (run on your machine; EBI FTP is reachable there)
    # start SMALL to confirm the URL/parse before a big pull:
    python -m src.data.vcf_fetch --n 20 --out data/vcf

    # then scale up (resumable -- re-run to continue / top up):
    python -m src.data.vcf_fetch --n 2000 --out data/vcf
    python run_pipeline.py       --data data/vcf
    python -m src.models.evaluate_cv --data data/vcf --folds 5
"""
from __future__ import annotations

import os
import gzip
import time
import argparse
import pandas as pd

try:
    import requests
except ImportError:
    requests = None

from src.data.download import (
    CRYPTIC_ROOT, _find_id_column, _phenotype_columns,
    _quality_column_for, parse_vcf, build_phenotypes_and_lineages,
)

REUSE_TABLE = "CRyPTIC_reuse_table_20221019.csv"

# Known resistance genes on H37Rv (NC_000962.3), [start, end] inclusive.
# Used only to make position tokens human-readable in SHAP; numeric range works
# for both strands. Not exhaustive -- covers the canonical AMR loci.
RESISTANCE_GENES: list[tuple[str, int, int]] = [
    ("gyrB", 5123, 7267), ("gyrA", 7302, 9818),
    ("fabG1", 1673440, 1674183), ("inhA", 1674202, 1675011),
    ("rrs", 1471846, 1473382), ("rrl", 1473658, 1476795),
    ("atpE", 1461045, 1461270), ("tlyA", 1917940, 1918746),
    ("pncA", 2288681, 2289241), ("eis", 2714124, 2715332),
    ("ahpC", 2726088, 2726780), ("katG", 2153889, 2156111),
    ("rpoB", 759807, 763325), ("rpoC", 763370, 767320),
    ("rpsL", 781560, 781934), ("mmpS5", 775087, 775590),
    ("mmpL5", 775586, 778480), ("Rv0678", 778990, 779487),
    ("embC", 4239863, 4243147), ("embA", 4243233, 4246517),
    ("embB", 4246514, 4249810), ("ethA", 4326004, 4327473),
    ("ethR", 4327549, 4328199), ("gid", 4407528, 4408202),
    ("pepQ", 2859300, 2860418), ("whiB7", 3568402, 3568987),
    ("thyA", 3073680, 3074471), ("folC", 2746234, 2747512),
    ("ribD", 2986839, 2987960),
]


def _gene_for(pos: int) -> str | None:
    for name, start, end in RESISTANCE_GENES:
        if start <= pos <= end:
            return name
    return None


def annotate_token(token: str) -> str:
    """'NC_000962.3_761155_C_T' -> 'rpoB@761155_C>T' if in a known gene, else
    'g761155_C>T' (compact, still unique)."""
    parts = token.rsplit("_", 3)  # [chrom, pos, ref, alt]
    if len(parts) != 4:
        return token
    _chrom, pos_s, ref, alt = parts
    try:
        pos = int(pos_s)
    except ValueError:
        return token
    gene = _gene_for(pos)
    prefix = gene if gene else "g"
    sep = "@" if gene else ""
    return f"{prefix}{sep}{pos}_{ref}>{alt}"


# ---- robust, resumable downloader (lessons from run_profiling) ---------------
def _valid_gzip(path: str) -> bool:
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1 << 20):
                pass
        return True
    except Exception:
        return False


def download_vcf(url: str, dest: str, retries: int = 3, timeout: int = 180) -> str:
    """Download to dest with gzip validation, retry, atomic rename. Skips if a
    valid file is already present (that's the resume mechanism)."""
    if os.path.exists(dest) and _valid_gzip(dest):
        return dest
    if requests is None:
        raise RuntimeError("pip install requests")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    last = None
    for attempt in range(1, retries + 1):
        tmp = dest + ".part"
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            f.write(chunk)
            if _valid_gzip(tmp):
                os.replace(tmp, dest)
                return dest
            last = "incomplete gzip"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        time.sleep(2 * attempt)
    raise RuntimeError(f"download failed after {retries} attempts ({last})")


# ---- subset selection --------------------------------------------------------
def select_subset(df: pd.DataFrame, id_col: str, anchor: str,
                  n: int | None, seed: int) -> pd.DataFrame:
    pheno_cols = _phenotype_columns(df)
    if anchor not in pheno_cols:
        raise SystemExit(f"anchor drug '{anchor}' not found. Options: "
                         f"{sorted(pheno_cols)}")
    pcol = pheno_cols[anchor]
    qcol = _quality_column_for(df, pcol)

    # --all (n is None): every isolate that has a VCF, no anchor/quality/balance
    # filter. Per-drug HIGH-quality labelling still happens downstream.
    if n is None:
        d = df[df["VCF"].notna()][[id_col, "VCF"]].copy()
        d["_ph"] = "all"
        return d.sample(frac=1, random_state=seed)

    cols = [id_col, "VCF", pcol] + ([qcol] if qcol else [])
    d = df[cols].copy()
    d = d[d["VCF"].notna()]
    if qcol:
        d = d[d[qcol].astype(str).str.upper() == "HIGH"]
    d["_ph"] = d[pcol].astype(str).str.upper()
    d = d[d["_ph"].isin(["R", "S"])]

    if n >= len(d):
        return d.sample(frac=1, random_state=seed)

    half = n // 2
    R, S = d[d["_ph"] == "R"], d[d["_ph"] == "S"]
    n_r = min(half, len(R))
    n_s = min(n - n_r, len(S))
    if n_r + n_s < n:            # one class short -> top up the other
        n_r = min(len(R), n - n_s)
    pick = pd.concat([R.sample(n_r, random_state=seed),
                      S.sample(n_s, random_state=seed)])
    return pick.sample(frac=1, random_state=seed)


# ---- driver ------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Download & parse CRyPTIC VCFs to variants.csv")
    ap.add_argument("--reuse-table", default=os.path.join("data", "processed", REUSE_TABLE))
    ap.add_argument("--out", default=os.path.join("data", "vcf"))
    ap.add_argument("--cache", default=None, help="VCF download cache (default: <out>/cache)")
    ap.add_argument("--anchor", default="Rifampicin",
                    help="drug used to class-balance the subset")
    ap.add_argument("--n", type=int, default=1000, help="target isolate count")
    ap.add_argument("--all", action="store_true", help="use every isolate (ignores --n)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ftp-base", default=CRYPTIC_ROOT,
                    help="base URL that the reuse-table VCF paths are relative to")
    ap.add_argument("--quality", default="HIGH",
                    help="phenotype-quality filter for labels ('' = keep all)")
    ap.add_argument("--no-annotate", action="store_true",
                    help="keep raw position tokens instead of gene-annotated ones")
    args = ap.parse_args()

    if not os.path.exists(args.reuse_table):
        raise SystemExit(f"reuse table not found: {args.reuse_table}")
    cache = args.cache or os.path.join(args.out, "cache")
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.reuse_table, low_memory=False)
    id_col = _find_id_column(df)
    n = None if args.all else args.n
    pick = select_subset(df, id_col, args.anchor, n, args.seed)
    bal = pick["_ph"].value_counts().to_dict()
    base = args.ftp_base.rstrip("/") + "/"
    print(f"selected {len(pick)} isolates  (anchor={args.anchor}, balance R/S={bal})")
    print(f"cache: {cache}   base URL: {base}")

    # Stream variants to disk one isolate at a time. Cross-isolate rows can't
    # collide (isolate_id differs), so we only dedup WITHIN an isolate and never
    # hold the full (millions-of-rows) table in memory -- this is what lets --all
    # (~12k isolates, ~12M rows) parse without blowing up RAM.
    import csv
    snps_only = os.environ.get("TBAMR_INDELS", "0").lower() not in ("1", "true", "yes")
    if not snps_only:
        print("  including INDELs (TBAMR_INDELS set)")
    v_path = os.path.join(args.out, "variants.csv")
    kept_ids: list[str] = []
    seen_muts: set[str] = set()
    n_rows = 0
    ok = fail = 0
    with open(v_path, "w", newline="") as vf:
        writer = csv.writer(vf)
        writer.writerow(["isolate_id", "mutation"])
        for i, (_, r) in enumerate(pick.iterrows(), 1):
            iid = str(r[id_col])
            rel = str(r["VCF"])
            url = base + rel.lstrip("/")
            dest = os.path.join(cache, rel.replace("/", "_"))
            try:
                download_vcf(url, dest)
                toks = set()
                for _iid, tok in parse_vcf(dest, iid, snps_only=snps_only):
                    toks.add(tok if args.no_annotate else annotate_token(tok))
                for tok in toks:
                    writer.writerow([iid, tok])
                    seen_muts.add(tok)
                n_rows += len(toks)
                kept_ids.append(iid)
                ok += 1
            except Exception as e:  # noqa: BLE001
                fail += 1
                print(f"  [warn] {iid}: {e}")
            if i % 50 == 0 or i == len(pick):
                print(f"  {i}/{len(pick)}  ok={ok} fail={fail}")
                vf.flush()

    print(f"\nwrote {v_path}  ({n_rows:,} rows, {len(kept_ids):,} isolates, "
          f"{len(seen_muts):,} distinct SNPs)")

    # phenotypes for the SELECTED isolates only, HIGH-quality per drug
    build_phenotypes_and_lineages(args.reuse_table, args.out,
                                  quality=(args.quality or None))
    p_path = os.path.join(args.out, "phenotypes.csv")
    pheno = pd.read_csv(p_path)
    pheno = pheno[pheno["isolate_id"].isin(set(kept_ids))]
    pheno.to_csv(p_path, index=False)
    per_drug = pheno.groupby("drug")["isolate_id"].nunique().sort_values(ascending=False)
    print(f"filtered phenotypes.csv -> {pheno['isolate_id'].nunique():,} isolates, "
          f"{len(pheno):,} labels")
    print("  per-drug isolate counts (HIGH-quality labels):")
    for drug, c in per_drug.items():
        print(f"      {drug:<13} {c:,}")
    print(f"\nNext: python run_pipeline.py --data {args.out}")
    print(f"      python -m src.models.evaluate_cv --data {args.out} --folds 5")


if __name__ == "__main__":
    main()
