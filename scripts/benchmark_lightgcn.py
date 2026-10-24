"""Benchmark stock LightGCN over several seeds on MovieLens-1M, THEN Alibaba-iFashion.

    python scripts/benchmark_lightgcn.py                  # 10 seeds each (2020..2029), same seeds as benchmark_all.py
    python scripts/benchmark_lightgcn.py --seeds 5
    python scripts/benchmark_lightgcn.py --datasets ifashion
    python scripts/benchmark_lightgcn.py --dry-run

Same driver/outputs as scripts/benchmark_all.py (resumable; results in outputs/benchmark/lightgcn-<dataset>/:
seed_<N>.json/.log, results.csv, summary.csv; every run also in MLflow). Use the SAME seeds as the MAGCL
benchmark -- the seed fixes the random 8:1:1 split, so equal seeds = equal splits = a valid paired test:

    python scripts/compare_benchmarks.py magcl-baseline-ml1m lightgcn-ml1m
"""

import argparse
import subprocess
import sys
from pathlib import Path

from benchmark_all import DATASETS, recap

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = {"ml1m": "configs/lightgcn.yaml", "ifashion": "configs/lightgcn_ifashion.yaml"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="ml1m,ifashion", help="comma list, run in this order")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=2020)
    ap.add_argument("--prefix", default="lightgcn", help="result folders are <prefix>-<dataset>")
    ap.add_argument("--extra", default="{}", help="extra overrides (inline JSON or file) for both datasets")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "benchmark"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in datasets if d not in CONFIGS]
    if unknown:
        sys.exit(f"unknown dataset(s) {unknown}; choose from {list(CONFIGS)}")

    failed = []
    for ds in datasets:
        print("\n" + "#" * 78 + f"\n# LightGCN on {DATASETS[ds]['label']}  ({args.seeds} seeds)\n" + "#" * 78, flush=True)
        cmd = [sys.executable, str(ROOT / "scripts" / "benchmark_seeds.py"),
               "--seeds", str(args.seeds), "--first-seed", str(args.first_seed),
               "--tag", f"{args.prefix}-{ds}", "--experiment", f"LightGCN-benchmark-{ds}",
               "--config", CONFIGS[ds], "--script", "scripts/run_lightgcn.py",
               "--hparams", "", "--extra", args.extra, "--out", args.out]
        cmd += ["--force"] * args.force + ["--dry-run"] * args.dry_run
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            failed.append(ds)

    if not args.dry_run:
        recap(args.prefix, datasets, Path(args.out))
    if failed:
        print(f"\nsome seeds failed on: {failed} (rerun the same command to retry only those)")
        sys.exit(1)


if __name__ == "__main__":
    main()
