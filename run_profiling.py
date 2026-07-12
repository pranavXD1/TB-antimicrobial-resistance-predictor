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
import time
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
    """True if `gzip -t` accepts the file. This is the integrity test real
    bioinformatics tools rely on, so it tolerates BGZF / multi-member gzip that
    Python's gzip.read can wrongly reject, while still catching truncation."""
    try:
        return subprocess.run(["gzip", "-t", path], capture_output=True).returncode == 0
    except Exception:
        return False


class _SourceCorrupt(Exception):
    """A full-size file that still fails integrity — re-downloading won't help."""


def _remote_size(url: str, timeout=30):
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        if r.ok and "content-length" in r.headers:
            return int(r.headers["content-length"])
    except Exception:
        pass
    return None


def download(url: str, dest_dir: str, retries: int = 8, base_sleep: int = 5) -> str:
    """Download with resume (HTTP Range) + exponential backoff.

    A stalled transfer resumes from the bytes already on disk instead of
    restarting, so an overnight run survives flaky connections. The finished file
    is gzip-validated; only a complete file is atomically moved to its real name.
    """
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, url.split("/")[-1])
    is_gz = dest.endswith(".gz")
    if os.path.exists(dest) and (not is_gz or _valid_gzip(dest)):
        return dest
    tmp = dest + ".part"
    total = _remote_size(url)
    last = None
    for attempt in range(1, retries + 1):
        try:
            have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            if total is not None and have >= total:
                have = 0  # oversized/corrupt partial -> restart clean
            headers = {"Range": f"bytes={have}-"} if have > 0 else {}
            mode = "ab" if have > 0 else "wb"
            with requests.get(url, stream=True, timeout=(30, 120), headers=headers) as r:
                if headers and r.status_code == 200:   # server ignored Range -> full restart
                    mode = "wb"
                r.raise_for_status()
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            f.write(chunk)
            if is_gz and not _valid_gzip(tmp):
                sz = os.path.getsize(tmp)
                os.remove(tmp)
                if total is not None and sz >= total:      # got the whole file, still bad -> source problem
                    raise _SourceCorrupt(f"downloaded full size ({sz} B) but fails gzip integrity")
                raise ValueError("incomplete/corrupt gzip after download")
            os.replace(tmp, dest)                       # atomic: a partial never lands at real name
            return dest
        except _SourceCorrupt as e:
            last = e
            break                                       # re-fetching identical bytes is futile -> skip sample
        except Exception as e:
            last = e
            if attempt < retries:
                sleep = min(base_sleep * (2 ** (attempt - 1)), 120)
                kept = os.path.getsize(tmp) // (1 << 20) if os.path.exists(tmp) else 0
                print(f"      attempt {attempt}/{retries} failed ({type(e).__name__}); "
                      f"{kept} MB kept, resuming in {sleep}s", flush=True)
                time.sleep(sleep)
    raise RuntimeError(f"download failed: {url} ({last})")


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
    ap.add_argument("--retries", type=int, default=8,
                    help="download attempts per file, with exponential backoff (overnight-safe)")
    ap.add_argument("--sleep", type=float, default=0,
                    help="seconds to pause between samples")
    ap.add_argument("--profile-timeout", type=int, default=1800,
                    help="max seconds for one TB-Profiler run before it is killed and the sample skipped")
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
    failed = []
    try:
        for i, (samp, info) in enumerate(items, 1):
            out_json = os.path.join(json_dir, f"{samp}.results.json")
            if os.path.exists(out_json):
                done += 1
                continue
            if args.dry_run:
                cmd = profile_cmd(samp, [os.path.join(args.reads_dir, u.split("/")[-1])
                                         for u in info["urls"]], args.results_dir, args.threads)
                print("  would download:", *info["urls"])
                print("  would run:", " ".join(cmd))
                continue
            if requests is None:
                raise SystemExit("pip install requests")
            try:
                print(f"  [{i}/{len(items)}] {samp}: downloading reads...", flush=True)
                reads = [download(u, args.reads_dir, retries=args.retries) for u in info["urls"]]
                print(f"  [{i}/{len(items)}] {samp}: profiling...", flush=True)
                subprocess.run(profile_cmd(samp, reads, args.results_dir, args.threads),
                               check=True, timeout=args.profile_timeout)
                ok += 1
                if args.clean:
                    for r in reads:
                        if os.path.exists(r):
                            os.remove(r)
            except Exception as e:
                print(f"  [warn] {samp} failed, skipping (re-run to retry): {e}", flush=True)
                failed.append(samp)
            if args.sleep:
                time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("\n[interrupted] progress saved; re-run the same command to resume "
              "(finished samples and partial downloads are kept).")

    print(f"\ndone. {ok} newly profiled, {done} already present -> {json_dir}/")
    if failed:
        print(f"{len(failed)} sample(s) failed this pass (re-run to retry): "
              f"{', '.join(failed[:10])}{' ...' if len(failed) > 10 else ''}")
    if not args.dry_run:
        print("Next: python -m src.data.tbprofiler_aggregate --json-dir",
              json_dir, "--out data/processed")


if __name__ == "__main__":
    main()
