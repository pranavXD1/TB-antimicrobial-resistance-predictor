#!/usr/bin/env python
"""
Ingest TB-Profiler results (the Sierra Leone / PRJEB7727 external holdout) into
the DuckDB ``sl_ext`` dataset.

It reads every ``*.results.json`` produced by ``run_profiling``, converts each
PASS variant to the project's CRyPTIC token format ``g{genome_pos}_{ref}>{alt}``
(position-only; the ``gene`` column is left NULL to match how CRyPTIC variants
were loaded), and writes them into the already-registered ``sl_ext`` holdout
dataset.

Isolates whose ``dr_variants`` AND ``other_variants`` are both empty are treated
as failed / stub profiles and skipped (e.g. the skipped-bad-reads stub), so they
don't inject a fake wild-type isolate into the holdout.

After loading, it reports how many distinct ``sl_ext`` tokens also occur in the
CRyPTIC vocabulary. That overlap is the sanity gate: it confirms the token format
matches and that there is transferable signal *before* any model is scored.

Phenotypes (R/S labels) come separately from Schleusener Table S2 and are loaded
by a later step -- this script fills only the variant side of ``sl_ext``.

Usage:
    python -m src.data.holdout_ingest \
        --results results_sl/results \
        --db data/tbamr.duckdb \
        --dataset sl_ext
"""
from __future__ import annotations

import os
import csv
import glob
import json
import argparse

import duckdb


# ---------------------------------------------------------------------------
# TB-Profiler variant -> CRyPTIC token
# ---------------------------------------------------------------------------
def variant_token(v: dict):
    """Return ``g{pos}_{ref}>{alt}`` for one TB-Profiler variant, or None to skip.

    Skips anything that didn't PASS the caller's filter and anything missing a
    concrete ref/alt/pos (e.g. large structural variants with null alleles).
    """
    filt = v.get("filter")
    if filt not in (None, "pass", "PASS"):
        return None
    pos = v.get("pos")
    ref = v.get("ref")
    alt = v.get("alt")
    if pos is None or not ref or not alt:
        return None
    return f"g{pos}_{ref}>{alt}"


def read_isolate(path: str):
    """(isolate_id, [tokens]) for one results JSON; tokens is None for a stub/failed profile."""
    with open(path) as fh:
        d = json.load(fh)
    iso = d.get("id") or os.path.basename(path).split(".")[0]
    dr = d.get("dr_variants") or []
    other = d.get("other_variants") or []
    # both variant lists empty => failed/stub profile; don't inject a fake WT isolate
    if not dr and not other:
        return iso, None
    tokens = []
    for v in dr + other:
        t = variant_token(v)
        if t:
            tokens.append(t)
    return iso, tokens


def collect(results_dir: str):
    files = sorted(glob.glob(os.path.join(results_dir, "*.results.json")))
    pairs = set()                           # (isolate_id, token), de-duped GLOBALLY
    per_iso_files = {}                      # isolate_id -> number of result files it appeared in
    isolates = set()
    skipped, empty_pass = [], []
    for f in files:
        iso, tokens = read_isolate(f)
        if tokens is None:
            skipped.append(iso)
            continue
        per_iso_files[iso] = per_iso_files.get(iso, 0) + 1
        isolates.add(iso)
        uniq = set(tokens)
        if not uniq:
            empty_pass.append(iso)          # had variants, but none survived PASS/allele filter
        for t in uniq:
            pairs.add((iso, t))             # a sample split across >1 run file -> union of its tokens
    rows = sorted(pairs)
    multi = {i: c for i, c in per_iso_files.items() if c > 1}
    return files, sorted(isolates), rows, skipped, empty_pass, multi


def write_variants_csv(rows, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    vpath = os.path.join(out_dir, "variants.csv")
    with open(vpath, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["isolate_id", "mutation"])
        w.writerows(rows)
    return vpath


# ---------------------------------------------------------------------------
# DB load (idempotent) + CRyPTIC-overlap sanity gate
# ---------------------------------------------------------------------------
def load_db(db_path: str, dataset: str, vpath: str):
    con = duckdb.connect(db_path)
    try:
        # idempotent: clear any existing rows for this holdout dataset first
        existing = con.execute(
            "SELECT COUNT(*) FROM isolates WHERE dataset_id = ?", [dataset]
        ).fetchone()[0]
        if existing:
            con.execute(
                "DELETE FROM variants WHERE isolate_id IN "
                "(SELECT isolate_id FROM isolates WHERE dataset_id = ?)", [dataset])
            con.execute(
                "DELETE FROM phenotypes WHERE isolate_id IN "
                "(SELECT isolate_id FROM isolates WHERE dataset_id = ?)", [dataset])
            con.execute("DELETE FROM isolates WHERE dataset_id = ?", [dataset])

        vp = vpath.replace("'", "''")
        # derive isolates from the variant file (every kept isolate has >=1 row)
        con.execute(f"""
            INSERT INTO isolates (isolate_id, dataset_id)
            SELECT DISTINCT isolate_id, '{dataset}'
            FROM read_csv_auto('{vp}', header=true)
        """)
        con.execute(f"""
            INSERT INTO variants (isolate_id, mutation)
            SELECT DISTINCT isolate_id, mutation
            FROM read_csv_auto('{vp}', header=true)
        """)

        n_iso = con.execute(
            "SELECT COUNT(*) FROM isolates WHERE dataset_id = ?", [dataset]).fetchone()[0]
        n_var = con.execute(
            "SELECT COUNT(*) FROM variants v JOIN isolates i ON v.isolate_id=i.isolate_id "
            "WHERE i.dataset_id = ?", [dataset]).fetchone()[0]
        n_distinct = con.execute(
            "SELECT COUNT(DISTINCT v.mutation) FROM variants v JOIN isolates i "
            "ON v.isolate_id=i.isolate_id WHERE i.dataset_id = ?", [dataset]).fetchone()[0]

        row = con.execute("""
            WITH sl AS (
                SELECT DISTINCT v.mutation
                FROM variants v JOIN isolates i ON v.isolate_id=i.isolate_id
                WHERE i.dataset_id = ?
            ),
            cr AS (
                SELECT DISTINCT v.mutation
                FROM variants v JOIN isolates i ON v.isolate_id=i.isolate_id
                WHERE i.dataset_id = 'cryptic'
            )
            SELECT (SELECT COUNT(*) FROM sl),
                   (SELECT COUNT(*) FROM sl WHERE mutation IN (SELECT mutation FROM cr))
        """, [dataset]).fetchone()
        sl_vocab, overlap = row[0], row[1]

        examples = [r[0] for r in con.execute("""
            WITH sl AS (
                SELECT DISTINCT v.mutation
                FROM variants v JOIN isolates i ON v.isolate_id=i.isolate_id
                WHERE i.dataset_id = ?
            ),
            cr AS (
                SELECT DISTINCT v.mutation
                FROM variants v JOIN isolates i ON v.isolate_id=i.isolate_id
                WHERE i.dataset_id = 'cryptic'
            )
            SELECT mutation FROM sl WHERE mutation IN (SELECT mutation FROM cr) LIMIT 8
        """, [dataset]).fetchall()]

        return dict(n_iso=n_iso, n_var=n_var, n_distinct=n_distinct,
                    sl_vocab=sl_vocab, overlap=overlap, examples=examples)
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results_sl/results",
                    help="dir containing *.results.json (default: results_sl/results)")
    ap.add_argument("--db", default="data/tbamr.duckdb",
                    help="DuckDB path (default: data/tbamr.duckdb)")
    ap.add_argument("--dataset", default="sl_ext",
                    help="holdout dataset id to fill (default: sl_ext)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write variants.csv (default: data/<dataset>)")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join("data", args.dataset)

    files, isolates, rows, skipped, empty_pass, multi = collect(args.results)
    if not files:
        raise SystemExit(f"no *.results.json found in {args.results}")

    vpath = write_variants_csv(rows, out_dir)

    print(f"scanned {len(files)} result files")
    print(f"  kept    : {len(isolates)} distinct isolates, {len(rows)} distinct PASS variant rows")
    if multi:
        ex = ", ".join(f"{i}(x{c})" for i, c in list(multi.items())[:5])
        print(f"  note    : {len(multi)} sample(s) appeared in >1 result file; tokens unioned -> {ex}")
    if skipped:
        print(f"  skipped : {len(skipped)} stub/failed profiles (empty variant lists) "
              f"-> {', '.join(skipped[:5])}{' ...' if len(skipped) > 5 else ''}")
    if empty_pass:
        print(f"  warning : {len(empty_pass)} isolates had variants but none PASSed the filter "
              f"-> {', '.join(empty_pass[:5])}")
    print(f"  wrote   : {vpath}")

    stats = load_db(args.db, args.dataset, vpath)
    print("\n" + "=" * 60)
    print(f"LOADED into '{args.dataset}'")
    print("=" * 60)
    print(f"isolates            : {stats['n_iso']}")
    print(f"variant rows        : {stats['n_var']}")
    print(f"distinct tokens     : {stats['n_distinct']}")
    pct = (100.0 * stats['overlap'] / stats['sl_vocab']) if stats['sl_vocab'] else 0.0
    print(f"tokens also in CRyPTIC: {stats['overlap']} / {stats['sl_vocab']}  ({pct:.1f}%)")
    if stats['examples']:
        print(f"  e.g. {', '.join(stats['examples'][:8])}")
    print("\nNext: load Schleusener Table S2 phenotypes into this dataset, then score.")


if __name__ == "__main__":
    main()
