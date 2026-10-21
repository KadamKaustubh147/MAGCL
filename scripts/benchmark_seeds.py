"""Run a MAGCL variant over several seeds and report mean +- std of every metric.

    python scripts/benchmark_seeds.py                       # baseline MAGCL, ML-1M, seeds 2020..2029
    python scripts/benchmark_seeds.py --seeds 5 --tag quick
    python scripts/benchmark_seeds.py --aggregate-only      # just re-print the summary from saved results
    python scripts/benchmark_seeds.py --dry-run             # show the commands, run nothing

Each seed is a separate `scripts/run.py` process (clean GPU/RAM each time), run one after another,
logged to MLflow as "<tag>-seed<N>". Results are saved per seed, so a crash or Ctrl-C loses nothing:
rerun the same command and finished seeds are skipped (--force redoes them).

NOTE the seed changes BOTH the weight initialisation AND RecBole's random 8:1:1 split (the split is
drawn after init_seed), so the std includes split-to-split variation. That is the usual reporting
protocol, but it means differences between two models are only comparable if you give both the SAME
seeds -- which this script does by default.

Other variants: --script scripts/run_graclus.py --extra configs/graclus_ml1m.json --tag graclus-replace
(--hparams is merged first, --extra second, the seed last).

Outputs in outputs/benchmark/<tag>/: seed_<N>.json, seed_<N>.log, override_<N>.json,
results.csv (one row per seed), summary.csv (mean/std/min/max per metric).
"""

import argparse
import ast
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_json(arg: str) -> dict:
    """Inline JSON or a path to a .json file (same convention as scripts/run.py --override)."""
    p = Path(arg)
    return json.loads(p.read_text(encoding="utf-8") if p.is_file() else arg)


def parse_log(text: str) -> dict:
    """Pull the two result dicts scripts/run.py prints at the end: 'best valid: {...}' and 'test: {...}'."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"^(best valid|test): (\{.*\})\s*$", line.strip())
        if m:
            out["valid" if m.group(1) == "best valid" else "test"] = ast.literal_eval(m.group(2))
    if set(out) != {"valid", "test"}:
        raise ValueError("no 'best valid:' / 'test:' result lines found (did the run finish?)")
    return out


def run_seed(seed: int, args, hparams: dict, extra: dict, outdir: Path) -> dict:
    override = outdir / f"override_{seed}.json"
    override.write_text(json.dumps({**hparams, **extra, "seed": seed}, indent=2), encoding="utf-8")
    cmd = [sys.executable, str(ROOT / args.script), "--config", args.config, "--override", str(override),
           "--experiment", args.experiment, "--run-name", f"{args.tag}-seed{seed}", "--quiet"]
    if args.dry_run:
        print(" ".join(cmd))
        return {}
    log = outdir / f"seed_{seed}.log"
    start = time.time()
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"}).returncode
    if rc != 0:
        raise RuntimeError(f"exit code {rc}, see {log}")
    result = {"seed": seed, "minutes": round((time.time() - start) / 60, 1), **parse_log(log.read_text(encoding="utf-8", errors="replace"))}
    (outdir / f"seed_{seed}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def summarize(rows: list, outdir: Path) -> None:
    """Mean / sample-std (ddof=1) / min / max of every metric across seeds; prints and saves summary.csv."""
    summary = []
    for split in ("valid", "test"):
        for metric in rows[0][split]:
            v = np.array([r[split][metric] for r in rows], dtype=float)
            summary.append([split, metric, v.mean(), v.std(ddof=1) if len(v) > 1 else float("nan"), v.min(), v.max(), len(v)])
    with open(outdir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "metric", "mean", "std", "min", "max", "n_seeds"])
        w.writerows(summary)
    with open(outdir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        metrics = list(rows[0]["test"])
        w.writerow(["seed"] + [f"valid_{m}" for m in metrics] + [f"test_{m}" for m in metrics])
        for r in rows:
            w.writerow([r["seed"]] + [r["valid"][m] for m in metrics] + [r["test"][m] for m in metrics])

    print(f"\n{len(rows)} seeds: {[r['seed'] for r in rows]}")
    print(f"{'metric':<12}{'valid mean +- std':<24}{'test mean +- std':<24}{'test min / max'}")
    by = {(s, m): (mu, sd, lo, hi) for s, m, mu, sd, lo, hi, _ in summary}
    for metric in rows[0]["test"]:
        vm, vs, _, _ = by[("valid", metric)]
        tm, ts, tl, th = by[("test", metric)]
        print(f"{metric:<12}{f'{vm:.4f} +- {vs:.4f}':<24}{f'{tm:.4f} +- {ts:.4f}':<24}{tl:.4f} / {th:.4f}")
    print(f"\nsaved: {outdir / 'summary.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10, help="number of seeds")
    ap.add_argument("--first-seed", type=int, default=2020, help="seeds are first-seed, first-seed+1, ... (2020 is the repo default)")
    ap.add_argument("--tag", default="magcl-baseline")
    ap.add_argument("--script", default="scripts/run.py", help="training entry point (run.py, run_graclus.py, run_projection.py)")
    ap.add_argument("--config", default="configs/magcl.yaml")
    ap.add_argument("--hparams", default="configs/best_ml1m_hparams.json", help="tuned lam1-4/theta, merged first")
    ap.add_argument("--extra", default="{}", help="extra overrides (inline JSON or file), merged after --hparams")
    ap.add_argument("--experiment", default="MAGCL-benchmark", help="MLflow experiment name")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "benchmark"))
    ap.add_argument("--force", action="store_true", help="redo seeds that already have a saved result")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.out) / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    hparams, extra = load_json(str(ROOT / args.hparams)), load_json(args.extra)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))

    rows, failed = [], []
    for i, seed in enumerate(seeds, 1):
        saved = outdir / f"seed_{seed}.json"
        if saved.exists() and not args.force:
            rows.append(json.loads(saved.read_text(encoding="utf-8")))
            print(f"[{i}/{len(seeds)}] seed {seed}: already done (test recall@20 {rows[-1]['test'].get('recall@20')}), skipping")
            continue
        if args.aggregate_only:
            continue
        print(f"[{i}/{len(seeds)}] seed {seed}: training ... (log: {outdir / f'seed_{seed}.log'})", flush=True)
        try:
            result = run_seed(seed, args, hparams, extra, outdir)
        except Exception as e:  # keep going: one bad seed should not waste the others
            print(f"    FAILED: {e}")
            failed.append(seed)
            continue
        if result:
            rows.append(result)
            print(f"    done in {result['minutes']} min: valid recall@20 {result['valid']['recall@20']:.4f}, "
                  f"test recall@20 {result['test']['recall@20']:.4f}")

    if args.dry_run:
        return
    if rows:
        summarize(rows, outdir)
    if failed:
        print(f"\nFAILED seeds: {failed} (rerun the same command to retry only those)")
        sys.exit(1)


if __name__ == "__main__":
    main()
