"""
Aggregate TB-Profiler result JSONs -> variants.csv (+ lineages.csv).

>>> RUN ON YOUR MACHINE <<< (after profiling genomes).

Workflow:
    tb-profiler profile -1 r1.fq.gz -2 r2.fq.gz -p <sample> --dir results/
    -> results/<sample>.results.json   (one per isolate)
    python -m src.data.tbprofiler_aggregate --json-dir results --out data/processed

Each variant becomes a token `gene_change` (e.g. rpoB_S450L) -> interpretable
features, so your SHAP plots read like real biology instead of coordinates.
TB-Profiler also calls lineage, which fills the gap left by the CRyPTIC reuse
table (no lineage column there).

The JSON parsing handles several TB-Profiler schema versions (key names have
changed across releases) and was validated on a mock result file.
"""
from __future__ import annotations

import os
import re
import json
import glob
import argparse
import pandas as pd

THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V", "Ter": "*",
}


def normalize_change(change: str) -> str:
    """p.Ser450Leu -> S450L when possible; otherwise return the change as-is."""
    if not change:
        return change
    m = re.match(r"p\.([A-Za-z]{3})(\d+)([A-Za-z]{3}|\*|Ter)", change)
    if m:
        a1 = THREE_TO_ONE.get(m.group(1), m.group(1))
        a2 = THREE_TO_ONE.get(m.group(3), m.group(3))
        return f"{a1}{m.group(2)}{a2}"
    return change.replace("p.", "").replace("c.", "")


def _first(d: dict, *keys):
    for k in keys:
        if d.get(k):
            return d[k]
    return None


def variants_from_json(obj: dict) -> tuple[str, list[str], str | None]:
    """Return (sample_id, [mutation_tokens], lineage) from one TB-Profiler result."""
    sample = _first(obj, "id", "sample_name", "sample") or "unknown"
    tokens: list[str] = []
    # different TB-Profiler versions store variants under different keys
    for key in ("dr_variants", "other_variants", "variants", "qc_variants"):
        for v in (obj.get(key) or []):
            gene = _first(v, "gene", "gene_name", "locus_tag")
            change = _first(v, "change", "protein_change", "hgvs_p",
                            "nucleotide_change", "hgvs_c")
            if gene and change:
                tokens.append(f"{gene}_{normalize_change(change)}")
    lineage = _first(obj, "main_lineage", "lineage", "sublin", "main_lin")
    return sample, sorted(set(tokens)), lineage


def load_id_map(path: str) -> dict:
    """Map an ENA accession -> CRyPTIC UNIQUEID.

    Accepts the CRyPTIC reuse table or a generic 2-column (accession, uniqueid)
    file. Merges EVERY ENA* column so we match whichever accession form was
    used (the reuse table stores ERS* secondary IDs; ENA's API returns SAMEA*).
    """
    df = pd.read_csv(path, sep=None, engine="python")
    m: dict = {}
    if "UNIQUEID" in df.columns:
        for c in [c for c in df.columns if c.upper().startswith("ENA")]:
            for acc, uid in zip(df[c].astype(str), df["UNIQUEID"].astype(str)):
                if acc and acc != "nan":
                    m[acc] = uid
        if m:
            return m
    if df.shape[1] >= 2:
        return dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))
    return m


def load_manifest_bridge(path: str) -> dict:
    """sample_accession (what TB-Profiler used as the id) -> query_accession
    (what we queried ENA with, which is the form present in the reuse table)."""
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path, sep="\t")
    if {"sample_accession", "query_accession"} <= set(df.columns):
        return dict(zip(df["sample_accession"].astype(str),
                        df["query_accession"].astype(str)))
    return {}


def aggregate(json_dir: str, out_dir: str, id_map: dict[str, str] | None = None) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    if not files:
        raise SystemExit(f"No .json files in {json_dir}. Profile genomes first "
                         "with `tb-profiler profile ... --dir {json_dir}`.")
    print(f"  aggregating {len(files):,} TB-Profiler result files")

    def remap(s):
        return id_map.get(s, s) if id_map else s

    var_rows, lin_rows, unmapped = [], [], 0
    for f in files:
        try:
            obj = json.load(open(f))
        except Exception as e:
            print(f"  [warn] {os.path.basename(f)}: {e}")
            continue
        sample, toks, lineage = variants_from_json(obj)
        if id_map and sample not in id_map:
            unmapped += 1
        sample = remap(sample)
        var_rows += [(sample, t) for t in toks]
        if lineage:
            lin_rows.append((sample, lineage))
    if id_map and unmapped:
        print(f"  [warn] {unmapped} sample id(s) had no UNIQUEID in the map (kept as-is)")

    variants = pd.DataFrame(var_rows, columns=["isolate_id", "mutation"]).drop_duplicates()
    v_path = os.path.join(out_dir, "variants.csv")
    variants.to_csv(v_path, index=False)
    print(f"  wrote {v_path}  ({len(variants):,} rows, "
          f"{variants['isolate_id'].nunique():,} isolates, "
          f"{variants['mutation'].nunique():,} distinct mutations)")
    paths = {"variants": v_path}

    if lin_rows:
        lineages = pd.DataFrame(lin_rows, columns=["isolate_id", "lineage"]).drop_duplicates("isolate_id")
        l_path = os.path.join(out_dir, "lineages.csv")
        lineages.to_csv(l_path, index=False)
        print(f"  wrote {l_path}  ({len(lineages):,} rows)")
        paths["lineages"] = l_path
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate TB-Profiler JSON -> variants.csv")
    ap.add_argument("--json-dir", required=True, help="dir of *.results.json")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--id-map", default="data/processed/CRyPTIC_reuse_table_20221019.csv",
                    help="reuse table or 2-col file to remap accession -> UNIQUEID")
    ap.add_argument("--manifest", default="data/processed/fastq_manifest.tsv",
                    help="ENA manifest, used to bridge SAMEA <-> queried accession")
    args = ap.parse_args()
    id_map = load_id_map(args.id_map) if args.id_map and os.path.exists(args.id_map) else None
    if id_map:
        # extend so the SAMEA* ids TB-Profiler used resolve via the queried
        # accession that actually appears in the reuse table
        bridge = load_manifest_bridge(args.manifest)
        added = 0
        for samp_acc, query_acc in bridge.items():
            if samp_acc not in id_map and query_acc in id_map:
                id_map[samp_acc] = id_map[query_acc]
                added += 1
        print(f"  loaded id map ({len(id_map):,} entries"
              + (f", +{added:,} bridged via manifest" if added else "") + ")")
    aggregate(args.json_dir, args.out, id_map=id_map)
    print("\nNext: python run_pipeline.py --data", args.out,
          "   (and python -m src.models.multitask --data", args.out + ")")


if __name__ == "__main__":
    main()
