"""
Ingest GenTB per-drug genotype-phenotype matrices into the project's format.

GenTB (Farhat lab, github.com/farhat-lab/gentb-site) ships ready per-drug binary
matrices under R/Neural_Network/input/ (pza.csv, str.csv, emb.csv, eth.csv).
Column 1 is the phenotype (r/s); the remaining columns are variant
presence/absence with names like:

    SNP_CN_2289180_A62C_V21G_pncA
    <type>_<category>_<H37Rv position>_<nt change>_<aa change>_<gene>

Because the H37Rv genomic position is embedded, we rebuild project-style tokens
(gene@pos_ref>alt) that flow through build_features unchanged. This yields
**pyrazinamide and streptomycin** — the drugs absent from the CRyPTIC UKMYC plate
— as trainable datasets with no sequence download.

Caveats (documented, not bugs): these matrices are candidate-gene by construction
(only each drug's resistance genes), carry no isolate IDs, and are enriched for
resistance. So they support per-drug training / CV / SHAP / calibration, but NOT
the population-structure CV (which needs genome-wide variants to cluster
lineages), and they are not directly poolable with the CRyPTIC isolates.
"""
from __future__ import annotations

import os
import re
import argparse
import pandas as pd

DRUG_FROM_FILE = {
    "pza": "Pyrazinamide", "str": "Streptomycin",
    "emb": "Ethambutol", "eth": "Ethionamide",
}
_SNP_NT = re.compile(r"^([ACGT])\d+([ACGT])$")


def parse_colname(col: str):
    """SNP_CN_2289180_A62C_V21G_pncA -> (type, gpos, nt_change, gene) or None."""
    parts = col.split("_")
    if len(parts) < 4:
        return None
    vtype = parts[0]                      # SNP / INS / DEL
    gpos = gi = None
    for i in range(1, len(parts)):        # first all-digit field = H37Rv position
        if parts[i].isdigit():
            gpos, gi = int(parts[i]), i
            break
    if gpos is None:
        return None
    gene = parts[-1]
    nt_change = parts[gi + 1] if gi + 1 < len(parts) else ""
    return vtype, gpos, nt_change, gene


def make_token(vtype: str, gpos: int, nt_change: str, gene: str) -> str:
    """Rebuild a project-style, position-bearing token."""
    if vtype == "SNP":
        m = _SNP_NT.match(nt_change or "")
        if m:
            return f"{gene}@{gpos}_{m.group(1)}>{m.group(2)}"
        return f"{gene}@{gpos}_snp"
    tag = vtype.lower()                    # ins / del
    return f"{gene}@{gpos}_{tag}_{nt_change}" if nt_change else f"{gene}@{gpos}_{tag}"


def ingest(csv_path: str, drug: str, out_dir: str, prefix: str) -> dict:
    df = pd.read_csv(csv_path)
    label_col = df.columns[0]
    feat_cols = list(df.columns[1:])

    col_token, skipped = {}, 0
    for c in feat_cols:
        p = parse_colname(c)
        if p:
            col_token[c] = make_token(*p)
        else:
            skipped += 1

    var_rows, ph_rows = [], []
    for i, row in df.iterrows():
        iso = f"{prefix}{i:05d}"
        lab = str(row[label_col]).strip().lower().strip('"')
        ph_rows.append((iso, drug, "R" if lab.startswith("r") else "S"))
        for c, tok in col_token.items():
            if row[c] == 1:
                var_rows.append((iso, tok))

    variants = pd.DataFrame(var_rows, columns=["isolate_id", "mutation"]).drop_duplicates()
    phenos = pd.DataFrame(ph_rows, columns=["isolate_id", "drug", "phenotype"])
    os.makedirs(out_dir, exist_ok=True)
    variants.to_csv(os.path.join(out_dir, "variants.csv"), index=False)
    phenos.to_csv(os.path.join(out_dir, "phenotypes.csv"), index=False)
    return {"isolates": len(df), "features_used": len(col_token), "features_skipped": skipped,
            "distinct_tokens": variants["mutation"].nunique(), "variant_rows": len(variants),
            "R": int((phenos["phenotype"] == "R").sum()), "S": int((phenos["phenotype"] == "S").sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest a GenTB per-drug matrix")
    ap.add_argument("--input", required=True, help="path to pza.csv / str.csv / ...")
    ap.add_argument("--drug", default=None, help="drug name (else inferred from filename)")
    ap.add_argument("--out", default=None, help="output data dir (else data/gentb_<drug>)")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.input))[0].lower()
    drug = args.drug or DRUG_FROM_FILE.get(stem, stem.capitalize())
    out = args.out or os.path.join("data", f"gentb_{stem}")
    prefix = f"GENTB_{stem.upper()}_"

    print(f"Ingesting {args.input}  ->  {out}   (drug={drug})")
    s = ingest(args.input, drug, out, prefix)
    print(f"  isolates:          {s['isolates']}  ({s['R']} R / {s['S']} S)")
    print(f"  features used:     {s['features_used']}  (skipped {s['features_skipped']} unparseable)")
    print(f"  distinct tokens:   {s['distinct_tokens']}")
    print(f"  variant rows:      {s['variant_rows']}")
    print(f"  wrote {out}/variants.csv  and  {out}/phenotypes.csv")


if __name__ == "__main__":
    main()
