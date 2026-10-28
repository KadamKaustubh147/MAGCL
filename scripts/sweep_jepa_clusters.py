"""Sweep the item-clustering method behind MAGCL + JEPA (not a benchmark: ONE seed by default).

    python scripts/sweep_jepa_clusters.py                        # iFashion; graclus, metis, spectral, popularity, random + no-JEPA reference
    python scripts/sweep_jepa_clusters.py --dataset ml1m
    python scripts/sweep_jepa_clusters.py --methods graclus,metis --n-clusters 64,256
    python scripts/sweep_jepa_clusters.py --seeds 3              # repeat each config on seeds 2020..2022 (same seed = same split for every method)
    python scripts/sweep_jepa_clusters.py --aggregate-only       # re-print the table from saved results
    python scripts/sweep_jepa_clusters.py --dry-run

Methods (src/jepa_cf/item_clusters.py): graclus (the original), metis, spectral, plus two controls with no graph
structure, popularity and random. `no-jepa` is the reference row: the same MAGCL run with jepa_weight 0, so every
method's gain is read against it. MAGCL's own hyperparameters are the dataset's tuned ones (same as benchmark_all.py).

Each run is a separate scripts/run_magcl_jepa.py process, logged to MLflow as "<tag>-seed<N>" in experiment
MAGCL-JEPA-clusters-<dataset>. Resumable: finished runs are skipped (--force redoes them). A method whose library is
missing (metis needs `pymetis`: Linux/macOS only) is skipped with a note, not an error.

CAVEAT when reading the table: graclus lands BELOW n_clusters (it halves until <= p), the others hit p exactly, so
the 'p' column differs between rows. Compare methods at similar p, or sweep --n-clusters so each has a neighbour.

Outputs in outputs/jepa_clusters/<dataset>/: <tag>/seed_<N>.json|.log|override_<N>.json, summary.csv, summary.md.
"""

import argparse
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from benchmark_all import DATASETS
from benchmark_seeds import load_json, parse_log

ROOT = Path(__file__).resolve().parent.parent
METHODS = ["graclus", "metis", "spectral", "popularity", "random"]
REQUIRES = {"graclus": "torch_cluster", "metis": "pymetis"}   # import name each method needs beyond the base install
HEADLINE = ["recall@20", "ndcg@20", "recall@50"]


def parse_clusters(text: str) -> dict:
    m = re.search(r"item clusters: method=(\w+) p=(\d+) size min/median/max=(\d+)/(\d+)/(\d+)", text)
    return {"p": int(m.group(2)), "size_min": int(m.group(3)), "size_median": int(m.group(4)), "size_max": int(m.group(5))} if m else {}


def run_one(tag, label, seed, overrides, args, ds, outdir):
    d = outdir / tag
    d.mkdir(parents=True, exist_ok=True)
    override = d / f"override_{seed}.json"
    override.write_text(json.dumps({**load_json(str(ROOT / ds["hparams"])), **load_json(args.extra), **overrides, "seed": seed}, indent=2),
                        encoding="utf-8")
    cmd = [sys.executable, str(ROOT / "scripts" / "run_magcl_jepa.py"),
           "--config", ds["config"], "configs/magcl_jepa.yaml", "--override", str(override),
           "--experiment", f"MAGCL-JEPA-clusters-{args.dataset}", "--run-name", f"{tag}-seed{seed}", "--quiet"]
    if args.dry_run:
        print("   ", " ".join(cmd))
        return None
    log = d / f"seed_{seed}.log"
    start = time.time()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"exit code {rc}, see {log}")
    text = log.read_text(encoding="utf-8", errors="replace")
    result = {"seed": seed, "tag": tag, "label": label, "minutes": round((time.time() - start) / 60, 1), **parse_log(text), **parse_clusters(text)}
    (d / f"seed_{seed}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def mean_std(vals):
    v = np.array(vals, dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else float("nan"))


def summarize(results: dict, outdir: Path, order: list) -> None:
    ref = results.get("no-jepa")
    ref_r20 = np.mean([r["test"]["recall@20"] for r in ref]) if ref else None
    rows = []
    for tag in order:
        rs = results.get(tag)
        if not rs:
            continue
        row = {"tag": tag, "n_seeds": len(rs), "p": rs[0].get("p", "") if tag != "no-jepa" else "", "size_min/med/max": "/".join(str(rs[0].get(k, "")) for k in ("size_min", "size_median", "size_max")) if "p" in rs[0] and tag != "no-jepa" else "",
               "minutes": round(float(np.mean([r["minutes"] for r in rs])), 1)}
        for m in HEADLINE:
            row[f"test_{m}"], row[f"test_{m}_std"] = mean_std([r["test"][m] for r in rs])
        row["valid_recall@20"], _ = mean_std([r["valid"]["recall@20"] for r in rs])
        row["d_recall@20_vs_no-jepa_%"] = 100 * (row["test_recall@20"] / ref_r20 - 1) if ref_r20 and tag != "no-jepa" else ""
        rows.append(row)
    if not rows:
        return
    rows.sort(key=lambda r: -r["test_recall@20"])
    with open(outdir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    pm = lambda r, m: f"{r[f'test_{m}']:.4f}" + (f" +- {r[f'test_{m}_std']:.4f}" if r["n_seeds"] > 1 else "")
    head = ["run", "seeds", "p (size min/med/max)", "test recall@20", "test ndcg@20", "test recall@50", "valid recall@20", "vs no-jepa", "min"]
    lines = [head] + [[r["tag"], str(r["n_seeds"]), f"{r['p']} ({r['size_min/med/max']})" if r["p"] != "" else "-", pm(r, "recall@20"), pm(r, "ndcg@20"),
                       pm(r, "recall@50"), f"{r['valid_recall@20']:.4f}",
                       f"{r['d_recall@20_vs_no-jepa_%']:+.2f}%" if r["d_recall@20_vs_no-jepa_%"] != "" else "-", str(r["minutes"])] for r in rows]
    widths = [max(len(l[i]) for l in lines) for i in range(len(head))]
    fmt = lambda l: "  ".join(c.ljust(w) for c, w in zip(l, widths))
    print("\n" + fmt(lines[0]) + "\n" + "  ".join("-" * w for w in widths))
    print("\n".join(fmt(l) for l in lines[1:]))
    md = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"] + ["| " + " | ".join(l) + " |" for l in lines[1:]]
    (outdir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nsaved: {outdir / 'summary.csv'} and summary.md  (sorted by test recall@20; 'vs no-jepa' = relative change of the mean)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ifashion", choices=list(DATASETS))
    ap.add_argument("--methods", default=",".join(METHODS), help=f"comma list from {METHODS}")
    ap.add_argument("--n-clusters", default="64", help="comma list; every method runs at every value")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--first-seed", type=int, default=2020)
    ap.add_argument("--no-reference", action="store_true", help="skip the no-JEPA (jepa_weight 0) reference run")
    ap.add_argument("--extra", default="{}", help="extra config overrides for every run (inline JSON or file), e.g. '{\"epochs\": 1}'")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "jepa_clusters"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        sys.exit(f"unknown method(s) {unknown}; choose from {METHODS}")
    ps = [int(x) for x in args.n_clusters.split(",")]
    ds = DATASETS[args.dataset]
    outdir = Path(args.out) / args.dataset
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))

    skipped = {m: REQUIRES[m] for m in methods if m in REQUIRES and importlib.util.find_spec(REQUIRES[m]) is None and not args.aggregate_only}
    for m, lib in skipped.items():
        print(f"SKIPPING {m}: `{lib}` is not installed in this environment")
    # (tag, label, overrides). Reference first, so a crash later still leaves the baseline.
    plan = [] if args.no_reference else [("no-jepa", "no-jepa", {"jepa_weight": 0, "jepa_z": "none", "jepa_cluster": "random"})]   # plain MAGCL; no clustering/positions needed
    plan += [(f"{m}-p{p}", m, {"jepa_cluster": m, "n_clusters": p}) for m in methods if m not in skipped for p in ps]

    results, failed = {}, []
    for tag, label, overrides in plan:
        for seed in seeds:
            saved = outdir / tag / f"seed_{seed}.json"
            if saved.exists() and not args.force:
                results.setdefault(tag, []).append(json.loads(saved.read_text(encoding="utf-8")))
                print(f"[{tag} seed {seed}] already done, skipping")
                continue
            if args.aggregate_only:
                continue
            print(f"[{tag} seed {seed}] {'would run' if args.dry_run else 'training'} ... (log: {outdir / tag / f'seed_{seed}.log'})", flush=True)
            try:
                r = run_one(tag, label, seed, overrides, args, ds, outdir)
            except Exception as e:   # keep going: one bad method should not waste the others
                print(f"    FAILED: {e}")
                failed.append(f"{tag}/seed{seed}")
                continue
            if r:
                results.setdefault(tag, []).append(r)
                print(f"    done in {r['minutes']} min: p={r.get('p', '-')}, valid recall@20 {r['valid']['recall@20']:.4f}, test recall@20 {r['test']['recall@20']:.4f}")

    if args.dry_run:
        return
    summarize(results, outdir, [t for t, _, _ in plan])
    if skipped:
        print(f"\nskipped (library missing): {skipped}")
    if failed:
        print(f"\nFAILED: {failed} (rerun the same command to retry only those)")
        sys.exit(1)


if __name__ == "__main__":
    main()
