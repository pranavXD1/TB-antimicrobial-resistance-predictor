"""
Local analytical database (DuckDB) unifying every data source behind one schema.

Why a database, and why here: the project now spans three heterogeneous sources
(CRyPTIC, GenTB, and an external validation cohort). A single schema with a
`datasets.is_holdout` flag makes the leakage guardrail *structural* — training
queries exclude the holdout by construction rather than by discipline. DuckDB is
chosen over SQLite/PostgreSQL because the variant table is large (CRyPTIC ~14M rows)
and the workload is analytical (scans/joins/group-bys), which its columnar engine
handles far faster; it is still a single portable file with no server, and its SQL
is close to PostgreSQL.

This is a storage/query/provenance layer, not a modelling change. `load_raw_from_db`
returns the same shape as `build_features.load_raw`, so the existing pipeline
(mutation_matrix -> build_dataset -> train/evaluate) runs on top unchanged and still
pivots to an in-memory sparse matrix.

Schema
    datasets(dataset_id PK, name, source, is_holdout, notes)
    isolates(isolate_id PK, dataset_id, lineage, country)
    variants(isolate_id, mutation, gene, pos, PK(isolate_id,mutation))
    phenotypes(isolate_id, drug, phenotype, PK(isolate_id,drug))
    mic(isolate_id, drug, mic_mgl, PK(isolate_id,drug))
    catalogue(gene, mutation, drug, grade)
"""
from __future__ import annotations

import os
import re
import argparse
import duckdb
import pandas as pd

_POS = re.compile(r"@(\d+)")
_GENE_AT = re.compile(r"^([A-Za-z0-9.\-]+)@")

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS datasets(
         dataset_id TEXT PRIMARY KEY, name TEXT, source TEXT,
         is_holdout BOOLEAN DEFAULT FALSE, notes TEXT)""",
    """CREATE TABLE IF NOT EXISTS isolates(
         isolate_id TEXT PRIMARY KEY, dataset_id TEXT, lineage TEXT, country TEXT)""",
    """CREATE TABLE IF NOT EXISTS variants(
         isolate_id TEXT, mutation TEXT, gene TEXT, pos INTEGER,
         PRIMARY KEY(isolate_id, mutation))""",
    """CREATE TABLE IF NOT EXISTS phenotypes(
         isolate_id TEXT, drug TEXT, phenotype TEXT, PRIMARY KEY(isolate_id, drug))""",
    """CREATE TABLE IF NOT EXISTS mic(
         isolate_id TEXT, drug TEXT, mic_mgl DOUBLE, PRIMARY KEY(isolate_id, drug))""",
    """CREATE TABLE IF NOT EXISTS catalogue(
         gene TEXT, mutation TEXT, drug TEXT, grade TEXT)""",
]


def connect(db_path: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(db_path)
    for stmt in SCHEMA:
        con.execute(stmt)
    return con


def _gene(tok: str):
    m = _GENE_AT.match(str(tok))
    if m:
        return m.group(1)
    if str(tok).startswith("burden::"):
        return str(tok).split("::", 1)[1]
    return None


def _pos(tok: str):
    m = _POS.search(str(tok))
    return int(m.group(1)) if m else None


def register_dataset(con, dataset_id, name, source, is_holdout=False, notes=""):
    con.execute("DELETE FROM datasets WHERE dataset_id=?", [dataset_id])
    con.execute("INSERT INTO datasets VALUES (?,?,?,?,?)",
                [dataset_id, name, source, is_holdout, notes])


def _purge(con, dataset_id):
    for tbl in ("variants", "phenotypes", "mic"):
        con.execute(f"""DELETE FROM {tbl} WHERE isolate_id IN
                        (SELECT isolate_id FROM isolates WHERE dataset_id=?)""", [dataset_id])
    con.execute("DELETE FROM isolates WHERE dataset_id=?", [dataset_id])


def load_long(con, dataset_id, variants_df, phenotypes_df, mic_df=None, isolates_meta=None):
    """Idempotent load of one dataset from long-format frames.

    variants_df: isolate_id,mutation   phenotypes_df: isolate_id,drug,phenotype
    """
    _purge(con, dataset_id)
    isos = pd.unique(pd.concat([variants_df["isolate_id"], phenotypes_df["isolate_id"]]))
    iso_df = pd.DataFrame({"isolate_id": isos, "dataset_id": dataset_id})
    if isolates_meta is not None:
        iso_df = iso_df.merge(isolates_meta, on="isolate_id", how="left")
    for c in ("lineage", "country"):
        if c not in iso_df:
            iso_df[c] = None
    iso_df = iso_df[["isolate_id", "dataset_id", "lineage", "country"]]

    v = variants_df.drop_duplicates(["isolate_id", "mutation"]).copy()
    mut = v["mutation"].astype(str)
    v["gene"] = mut.str.extract(r"^([A-Za-z0-9.\-]+)@", expand=False)   # vectorized (fast on ~14M rows)
    burden = mut.str.startswith("burden::")
    if burden.any():
        v.loc[burden, "gene"] = mut[burden].str.split("::", n=1).str[1]
    v["pos"] = pd.to_numeric(mut.str.extract(r"@(\d+)", expand=False), errors="coerce").astype("Int64")
    v = v[["isolate_id", "mutation", "gene", "pos"]]
    p = phenotypes_df.drop_duplicates(["isolate_id", "drug"])[["isolate_id", "drug", "phenotype"]]

    con.register("iso_df", iso_df); con.execute("INSERT INTO isolates SELECT * FROM iso_df")
    con.register("v_df", v);        con.execute("INSERT INTO variants SELECT * FROM v_df")
    con.register("p_df", p);        con.execute("INSERT INTO phenotypes SELECT * FROM p_df")
    if mic_df is not None and len(mic_df):
        m = mic_df.drop_duplicates(["isolate_id", "drug"])[["isolate_id", "drug", "mic_mgl"]]
        con.register("m_df", m);    con.execute("INSERT INTO mic SELECT * FROM m_df")
    for v_ in ("iso_df", "v_df", "p_df"):
        con.unregister(v_)


def load_dataset_dir(con, data_dir, dataset_id, name, source, is_holdout=False, notes=""):
    """Load a project data dir (variants.csv + phenotypes.csv) as one dataset.

    Uses DuckDB's native streaming CSV reader and does de-duplication + gene/pos
    extraction in-engine, so CRyPTIC's ~14M-row variants.csv loads in a couple of
    minutes without ever building a large in-memory DataFrame (which was swapping).
    """
    vpath = os.path.join(data_dir, "variants.csv").replace("'", "''")
    ppath = os.path.join(data_dir, "phenotypes.csv").replace("'", "''")
    register_dataset(con, dataset_id, name, source, is_holdout, notes)
    _purge(con, dataset_id)
    con.execute(f"""
        INSERT INTO variants
        SELECT DISTINCT isolate_id, mutation,
               NULLIF(regexp_extract(mutation, '^([A-Za-z0-9.-]+)@', 1), '') AS gene,
               TRY_CAST(regexp_extract(mutation, '@([0-9]+)', 1) AS INTEGER)   AS pos
        FROM read_csv_auto('{vpath}', header=true)
    """)
    con.execute(f"""
        INSERT INTO isolates
        SELECT DISTINCT isolate_id, '{dataset_id}', NULL, NULL FROM (
            SELECT isolate_id FROM read_csv_auto('{vpath}', header=true)
            UNION
            SELECT isolate_id FROM read_csv_auto('{ppath}', header=true)
        )
    """)
    con.execute(f"""
        INSERT INTO phenotypes
        SELECT DISTINCT isolate_id, drug, phenotype
        FROM read_csv_auto('{ppath}', header=true)
    """)
    n_iso = con.execute("SELECT COUNT(*) FROM isolates WHERE dataset_id=?", [dataset_id]).fetchone()[0]
    n_var = con.execute("""SELECT COUNT(*) FROM variants WHERE isolate_id IN
                           (SELECT isolate_id FROM isolates WHERE dataset_id=?)""", [dataset_id]).fetchone()[0]
    return {"isolates": n_iso, "variants": n_var}


def load_raw_from_db(con, datasets=None, exclude_holdout=False) -> dict:
    """Return {'variants','phenotypes'} like build_features.load_raw, filtered by dataset.

    exclude_holdout=True drops every dataset flagged is_holdout — the training filter.
    """
    cond, params = "1=1", []
    if datasets:
        cond += f" AND i.dataset_id IN ({','.join(['?'] * len(datasets))})"
        params += list(datasets)
    if exclude_holdout:
        cond += " AND COALESCE(d.is_holdout, FALSE)=FALSE"
    v = con.execute(f"""SELECT v.isolate_id, v.mutation FROM variants v
        JOIN isolates i ON v.isolate_id=i.isolate_id
        JOIN datasets d ON i.dataset_id=d.dataset_id WHERE {cond}""", params).df()
    p = con.execute(f"""SELECT p.isolate_id, p.drug, p.phenotype FROM phenotypes p
        JOIN isolates i ON p.isolate_id=i.isolate_id
        JOIN datasets d ON i.dataset_id=d.dataset_id WHERE {cond}""", params).df()
    return {"variants": v, "phenotypes": p}


def summary(con) -> pd.DataFrame:
    return con.execute("""
        SELECT d.dataset_id, d.name, d.is_holdout,
               COUNT(DISTINCT i.isolate_id)                       AS isolates,
               (SELECT COUNT(*) FROM variants v JOIN isolates i2 ON v.isolate_id=i2.isolate_id
                 WHERE i2.dataset_id=d.dataset_id)                AS variant_rows,
               (SELECT COUNT(*) FROM phenotypes p JOIN isolates i3 ON p.isolate_id=i3.isolate_id
                 WHERE i3.dataset_id=d.dataset_id)                AS pheno_rows
        FROM datasets d LEFT JOIN isolates i ON i.dataset_id=d.dataset_id
        GROUP BY d.dataset_id, d.name, d.is_holdout ORDER BY isolates DESC""").df()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the unified TB-AMR DuckDB")
    ap.add_argument("--db", default="data/tbamr.duckdb")
    ap.add_argument("--cryptic", default=None, help="CRyPTIC data dir (variants.csv+phenotypes.csv)")
    ap.add_argument("--gentb", action="append", default=[], help="GenTB data dir; repeatable")
    ap.add_argument("--holdout", action="append", default=[],
                    help="external holdout data dir (loaded is_holdout=TRUE); repeatable")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    con = connect(args.db)
    if args.cryptic:
        s = load_dataset_dir(con, args.cryptic, "cryptic", "CRyPTIC compendium",
                             "CRyPTIC 2022", is_holdout=False)
        print(f"  cryptic       : {s['isolates']} isolates, {s['variants']} variant rows")
    for d in args.gentb:
        did = f"gentb_{os.path.basename(d.rstrip('/')).replace('gentb_', '')}"
        s = load_dataset_dir(con, d, did, f"GenTB ({did})", "GenTB / Farhat lab", is_holdout=False)
        print(f"  {did:14s}: {s['isolates']} isolates, {s['variants']} variant rows")
    for d in args.holdout:
        did = os.path.basename(d.rstrip("/"))
        s = load_dataset_dir(con, d, did, f"External holdout ({did})", "external cohort",
                             is_holdout=True, notes="external validation only — never trained on")
        print(f"  {did:14s}: {s['isolates']} isolates, {s['variants']} variant rows  [HOLDOUT]")

    if args.summary or True:
        print("\n" + "=" * 60 + "\nDATABASE SUMMARY\n" + "=" * 60)
        print(summary(con).to_string(index=False))
    con.close()
    print(f"\nWrote {args.db}")


if __name__ == "__main__":
    main()
