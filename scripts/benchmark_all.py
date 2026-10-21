"""Benchmark MAGCL over several seeds on MovieLens-1M, THEN Alibaba-iFashion.

    python scripts/benchmark_all.py                    # baseline MAGCL, 10 seeds each, ML-1M first
    python scripts/benchmark_all.py --seeds 5
    python scripts/benchmark_all.py --datasets ifashion,ml1m          # other order / only one
    python scripts/benchmark_all.py --dry-run                         # show what would run

This is a thin driver: for each dataset it calls scripts/benchmark_seeds.py (see its docstring for the
resume behaviour, outputs and the seed/split note), so everything is resumable -- rerun the same command
and finished seeds are skipped. Results land in outputs/benchmark/<prefix>-<dataset>/.

Both datasets use the tuned lam1-4/theta from configs/best_ml1m_hparams.json (configs/magcl_ifashion.yaml
already carries the same values as its defaults) and the dataset's own tau/gamma/alpha from its YAML.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "ml1m": {"config": "configs/magcl.yaml", "label": "MovieLens-1M"},
    "ifashion": {"config": "configs/magcl_ifashion.yaml", "label": "Alibaba-iFashion"},
}


def recap(prefix: str, datasets: list, out: Path) -> None:
    """One compact line per dataset from each summary.csv."""
    print("\n" + "=" * 78 + "\nFINAL RECAP (test, mean +- std over seeds)\n" + "=" * 78)
    for ds in datasets:
        path = out / f"{prefix}-{ds}" / "summary.csv"
        if not path.exists():
            print(f"{DATASETS[ds]['label']:<18} no results yet")
            continue
        rows = {(r["split"], r["metric"]): r for r in csv.DictReader(open(path))}
        parts = [f"{m} {float(rows[('test', m)]['mean']):.4f}+-{float(rows[('test', m)]['std']):.4f}"
                 for m in ("recall@20", "ndcg@20", "recall@50") if ("test", m) in rows]
        print(f"{DATASETS[ds]['label']:<18} n={rows[('test', 'recall@20')]['n_seeds']}  " + "   ".join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="ml1m,ifashion", help="comma list, run in this order")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=2020)
    ap.add_argument("--prefix", default="magcl-baseline", help="result folders are <prefix>-<dataset>")
    ap.add_argument("--script", default="scripts/run.py")
    ap.add_argument("--hparams", default="configs/best_ml1m_hparams.json")
    ap.add_argument("--extra", default="{}", help="extra overrides (inline JSON or file) for both datasets")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "benchmark"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        sys.exit(f"unknown dataset(s) {unknown}; choose from {list(DATASETS)}")

    failed = []
    for ds in datasets:
        print("\n" + "#" * 78 + f"\n# {DATASETS[ds]['label']}  ({args.seeds} seeds)\n" + "#" * 78, flush=True)
        cmd = [sys.executable, str(ROOT / "scripts" / "benchmark_seeds.py"),
               "--seeds", str(args.seeds), "--first-seed", str(args.first_seed),
               "--tag", f"{args.prefix}-{ds}", "--experiment", f"MAGCL-benchmark-{ds}",
               "--config", DATASETS[ds]["config"], "--script", args.script,
               "--hparams", args.hparams, "--extra", args.extra, "--out", args.out]
        cmd += ["--force"] * args.force + ["--dry-run"] * args.dry_run
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:   # keep going: iFashion should still run if ML-1M had a failed seed
            failed.append(ds)

    if not args.dry_run:
        recap(args.prefix, datasets, Path(args.out))
    if failed:
        print(f"\nsome seeds failed on: {failed} (rerun the same command to retry only those)")
        sys.exit(1)


if __name__ == "__main__":
    main()
