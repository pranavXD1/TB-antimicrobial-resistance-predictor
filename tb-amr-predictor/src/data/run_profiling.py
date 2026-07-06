"""
Download FASTQs from the ENA manifest and run TB-Profiler, one sample at a time.

>>> RUN IN WSL2 OR DOCKER <<<  (TB-Profiler + bwa/samtools are Unix-only.)

Streams per sample: download -> profile -> (optionally) delete reads, so disk
stays bounded even for hundreds of genomes. Resumable: samples already in
--results-dir are skipped. Profiles by SAMPLE accession; the aggregator's
--id-map then maps those back to CRyPTIC UNIQUEIDs.

Start small to validate the real pipeline end to end:
    python -m src.data.run_profiling --manifest data/processed/fastq_manifest.tsv \
        --max 50 --threads 4 --clean
Then scale up by raising --max (it skips what's already done).

Add --dry-run to print the commands without downloading or profiling.
"""
from __future__ import annotations

import os
import csv
import gzip
import shutil
import subprocess
import argparse
from collections import OrderedDict

try:
    import requests
except ImportError:
    requests = None


def read_manifest(path: str) -> "OrderedDict[str, dict]":
    """Group manifest rows into one entry per sample: {sample: {run, urls[]}}."""
    samples: "OrderedDict[str, dict]" = OrderedDict()
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            samp = row.get("sample_accession") or row.get("query_accession")
            run = row.get("run_accession") or samp
            url = row.get("fastq_url")
            if not samp or not url:
                continue
            entry = samples.setdefault(samp, {"run": run, "urls": []})
            # keep only the first run's files for a clean 1 genome : 1 sample pass
            if run == entry["run"]:
                entry["urls"].append(url)
    return samples


def _valid_gzip(path: str) -> bool:
    """True if path is a complete, readable gzip (catches truncated downloads)."""
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1 << 20):
                pass
        return True
    except Exception:
        return False


def download(url: str, dest_dir: str, retries: int = 3) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, url.split("/")[-1])
    is_gz = dest.endswith(".gz")
    # reuse an existing file only if it's actually complete
    if os.path.exists(dest) and (not is_gz or _valid_gzip(dest)):
        return dest
    last = None
    for _ in range(retries):
        tmp = dest + ".part"
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if is_gz and not _valid_gzip(tmp):
                raise ValueError("incomplete/corrupt gzip after download")
            os.replace(tmp, dest)  # atomic: a partial never lands at the real name
            return dest
        except Exception as e:
            last = e
            if os.path.exists(tmp):
                os.remove(tmp)
    raise RuntimeError(f"download failed after {retries} attempts: {url} ({last})")


def profile_cmd(sample: str, reads: list[str], results_dir: str, threads: int) -> list[str]:
    cmd = ["tb-profiler", "profile", "-p", sample, "--dir", results_dir,
           "--threads", str(threads)]
    if len(reads) >= 2:
        cmd += ["-1", reads[0], "-2", reads[1]]
    else:
        cmd += ["-1", reads[0]]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description="Download + TB-Profiler over an ENA manifest")
    ap.add_argument("--manifest", default="data/processed/fastq_manifest.tsv")
    ap.add_argument("--reads-dir", default="reads")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max", type=int, default=None, help="profile at most N samples (first pass)")
    ap.add_argument("--shuffle", action="store_true",
                    help="deterministically shuffle sample order so --max gives a "
                         "class-balanced subset (the raw subset is R-block then S-block)")
    ap.add_argument("--clean", action="store_true", help="delete reads after each sample")
    ap.add_argument("--dry-run", action="store_true", help="print commands only")
    args = ap.parse_args()

    samples = read_manifest(args.manifest)
    items = list(samples.items())
    if args.shuffle:
        import random
        random.Random(0).shuffle(items)
    if args.max:
        items = items[:args.max]
    # tb-profiler writes JSONs into a `results/` subfolder of --dir, so that's
    # where we check resumability and where the aggregator should look.
    json_dir = os.path.join(args.results_dir, "results")
    os.makedirs(json_dir, exist_ok=True)

    # Fail fast: actually try to run tb-profiler, so a missing/broken/not-yet-
    # activated install stops us BEFORE any downloads.
    if not args.dry_run:
        try:
            r = subprocess.run(["tb-profiler", "version"],
                               capture_output=True, text=True)
            tbp_ok = (r.returncode == 0)
        except (FileNotFoundError, PermissionError, OSError):
            tbp_ok = False
        if not tbp_ok:
            raise SystemExit(
                "tb-profiler isn't runnable here.\n"
                "TB-Profiler needs the Unix toolchain and a conda env where it's "
                "installed AND activated.\n"
                "In WSL2/Docker:\n"
                "    conda create -n tb -c conda-forge -c bioconda tb-profiler -y\n"
                "    conda activate tb        # prompt should now show (tb)\n"
                "    tb-profiler version      # must print a version\n"
                "(or add --dry-run to preview the commands without it.)"
            )

    already = sum(1 for s, _ in items
                  if os.path.exists(os.path.join(json_dir, f"{s}.results.json")))
    todo = len(items) - already
    print(f"{len(items)} sample(s) selected: {already} already done, "
          f"{todo} to profile (resumable)")

    done = ok = 0
    for i, (samp, info) in enumerate(items, 1):
        out_json = os.path.join(json_dir, f"{samp}.results.json")
        if os.path.exists(out_json):
            done += 1
            continue
        cmd = profile_cmd(samp, [os.path.join(args.reads_dir, u.split("/")[-1])
                                 for u in info["urls"]], args.results_dir, args.threads)
        if args.dry_run:
            print("  would download:", *info["urls"])
            print("  would run:", " ".join(cmd))
            continue
        if requests is None:
            raise SystemExit("pip install requests")
        try:
            print(f"  [{i}/{len(items)}] {samp}: downloading reads...", flush=True)
            reads = [download(u, args.reads_dir) for u in info["urls"]]
            print(f"  [{i}/{len(items)}] {samp}: profiling...", flush=True)
            cmd = profile_cmd(samp, reads, args.results_dir, args.threads)
            subprocess.run(cmd, check=True)
            ok += 1
            if args.clean:
                for r in reads:
                    os.remove(r)
        except Exception as e:
            print(f"  [warn] {samp}: {e}")
    print(f"\ndone. {ok} newly profiled, {done} already present -> {json_dir}/")
    if not args.dry_run:
        print("Next: python -m src.data.tbprofiler_aggregate --json-dir",
              json_dir, "--out data/processed")


if __name__ == "__main__":
    main()
