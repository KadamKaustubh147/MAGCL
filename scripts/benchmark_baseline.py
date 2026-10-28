"""Shared driver behind scripts/benchmark_{adamcl,sgl,simgcl}.py: run one model over several seeds on
MovieLens-1M, THEN Alibaba-iFashion, via scripts/benchmark_seeds.py (same resumable per-seed runs, outputs
and summary.csv as benchmark_all.py / benchmark_lightgcn.py). Not meant to be run directly.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from benchmark_all import DATASETS, recap

ROOT = Path(__file__).resolve().parent.parent


def main(doc: str, label: str, prefix: str, script: str, configs: dict) -> None:
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="ml1m,ifashion", help="comma list, run in this order")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=2020)
    ap.add_argument("--prefix", default=prefix, help="result folders are <prefix>-<dataset>")
    ap.add_argument("--extra", default="{}", help="extra overrides (inline JSON or file) for both datasets")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "benchmark"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in datasets if d not in configs]
    if unknown:
        sys.exit(f"unknown dataset(s) {unknown}; choose from {list(configs)}")

    failed = []
    for ds in datasets:
        print("\n" + "#" * 78 + f"\n# {label} on {DATASETS[ds]['label']}  ({args.seeds} seeds)\n" + "#" * 78, flush=True)
        cmd = [sys.executable, str(ROOT / "scripts" / "benchmark_seeds.py"),
               "--seeds", str(args.seeds), "--first-seed", str(args.first_seed),
               "--tag", f"{args.prefix}-{ds}", "--experiment", f"{label}-benchmark-{ds}",
               "--config", configs[ds], "--script", script,
               "--hparams", "", "--extra", args.extra, "--out", args.out]
        cmd += ["--force"] * args.force + ["--dry-run"] * args.dry_run
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:   # keep going: iFashion should still run if ML-1M had a failed seed
            failed.append(ds)

    if not args.dry_run:
        recap(args.prefix, datasets, Path(args.out))
    if failed:
        print(f"\nsome seeds failed on: {failed} (rerun the same command to retry only those)")
        sys.exit(1)
