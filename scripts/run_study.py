"""One command for the whole MAGCL-vs-LightGCN study: benchmark both models, run the paired significance
tests, and write everything to one report file.

    python scripts/run_study.py                       # 10 seeds, ML-1M + iFashion, both models, then the report
    python scripts/run_study.py --seeds 5
    python scripts/run_study.py --datasets ml1m
    python scripts/run_study.py --report-only         # skip training, rebuild the report from saved results
    python scripts/run_study.py --dry-run             # show what would run

Training is resumable (finished seeds are skipped), so after a crash/Ctrl-C just rerun the same command.
Outputs, all under outputs/benchmark/:
    magcl-baseline-<ds>/, lightgcn-<ds>/     per-seed json + logs, results.csv, summary.csv
    compare_<magcl>_vs_<lightgcn>.csv        paired-test numbers (one per dataset)
    study_report.md                          the combined report: mean +- std per model + significance tables
"""

import argparse
import csv
import datetime
import subprocess
import sys
from pathlib import Path

import numpy as np

from benchmark_all import DATASETS
from compare_benchmarks import HEADLINE, compare, load

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
MODELS = {"magcl": "magcl-baseline", "lightgcn": "lightgcn"}   # model -> result-folder prefix (the scripts' defaults)


def train(args) -> list:
    """Run both benchmark drivers; returns the names of any that reported failed seeds."""
    common = ["--datasets", args.datasets, "--seeds", str(args.seeds), "--first-seed", str(args.first_seed),
              "--out", args.out] + ["--dry-run"] * args.dry_run
    failed = []
    for name, script in (("MAGCL", "benchmark_all.py"), ("LightGCN", "benchmark_lightgcn.py")):
        print("\n" + "=" * 78 + f"\n{name}\n" + "=" * 78, flush=True)
        if subprocess.run([sys.executable, str(SCRIPTS / script)] + common, cwd=ROOT).returncode != 0:
            failed.append(name)   # keep going: the other model should still run
    return failed


def model_table(ds: str, out: Path) -> list:
    """Markdown rows: test mean +- std (ddof=1) per model, plus the seeds each ran."""
    runs = {m: load(f"{p}-{ds}", "test", out) for m, p in MODELS.items() if (out / f"{p}-{ds}" / "results.csv").exists()}
    if not runs:
        return ["_no results yet_"]
    metrics = list(next(iter(next(iter(runs.values())).values())))   # metric names of the first seed of the first model
    lines = ["| model | n | " + " | ".join(metrics) + " |", "|---|---|" + "---|" * len(metrics)]
    for m, rows in runs.items():
        cells = []
        for met in metrics:
            v = np.array([r[met] for r in rows.values()])
            cells.append(f"{v.mean():.4f} ± {v.std(ddof=1) if len(v) > 1 else float('nan'):.4f}")
        lines.append(f"| {m} | {len(rows)} | " + " | ".join(cells) + " |")
    return lines


def test_table(rows: list) -> list:
    head = ("| metric | MAGCL | LightGCN | diff | rel % | 95% CI of diff | t-test p (Holm) | Wilcoxon p (Holm) | d_z | MAGCL wins |\n"
            "|---|---|---|---|---|---|---|---|---|---|")
    out = [head]
    for r in rows:
        name = f"**{r['metric']}**" if r["metric"] in HEADLINE else r["metric"]
        out.append(f"| {name} | {r['mean_a']:.4f} | {r['mean_b']:.4f} | {r['diff']:+.4f} | {r['rel_pct']:+.1f}% | "
                   f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['t_p']:.4g} ({r['t_p_holm']:.4g}) | "
                   f"{r['wilcoxon_p']:.4g} ({r['wilcoxon_p_holm']:.4g}) | {r['dz']:.2f} | {r['a_wins']}/{r['n']} |")
    return out


def verdict(rows: list) -> str:
    """One plain sentence per dataset on the two headline metrics, using the Holm-adjusted t-test and Wilcoxon."""
    parts = []
    for r in rows:
        if r["metric"] not in HEADLINE:
            continue
        sig = r["t_p_holm"] < 0.05 and r["wilcoxon_p_holm"] < 0.05
        direction = "MAGCL > LightGCN" if r["diff"] > 0 else "MAGCL < LightGCN"
        parts.append(f"{r['metric']}: {direction if sig else 'no significant difference'} "
                     f"({r['rel_pct']:+.1f}%, Holm t p={r['t_p_holm']:.3g}, Wilcoxon p={r['wilcoxon_p_holm']:.3g})")
    return "; ".join(parts)


def write_report(args, out: Path) -> Path:
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    md = ["# MAGCL vs LightGCN study", "",
          f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M}. Test split, seeds {args.first_seed}..{args.first_seed + args.seeds - 1} "
          f"(each seed fixes the init AND the random 8:1:1 split, identical for both models).", "",
          "- MAGCL: ML-1M raw CL with `configs/best_ml1m_hparams.json`; iFashion normalized CL with `configs/best_ifashion_normcl_hparams.json`. "
          "Lambdas/theta were tuned on the **validation** split.",
          "- LightGCN: stock RecBole, embedding 64, L=3, default reg, same protocol (`configs/lightgcn*.yaml`).", ""]
    for ds in datasets:
        md += [f"## {DATASETS[ds]['label']}", "", "### Mean ± std over seeds (test)", ""] + model_table(ds, out) + [""]
        a, b = f"{MODELS['magcl']}-{ds}", f"{MODELS['lightgcn']}-{ds}"
        try:
            seeds, rows = compare(a, b, "test", out)
        except (ValueError, FileNotFoundError) as e:
            md += [f"### Paired significance test", "", f"_skipped: {e}_", ""]
            continue
        path = out / f"compare_{a}_vs_{b}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        md += [f"### Paired significance test: MAGCL − LightGCN ({len(seeds)} paired seeds)", "",
               f"**{verdict(rows)}**", ""] + test_table(rows) + ["", f"Raw numbers: `{path.name}`", ""]
    md += ["## Method", "",
           "- **Pairing**: seed N draws the same random split for both models, so each metric is compared per seed and the split-to-split "
           "variance cancels out of the difference (paired t-test and Wilcoxon signed-rank on those differences).",
           "- **Multiple comparisons**: six correlated metrics are tested per dataset; p-values in parentheses are Holm-adjusted over those six. "
           "A result counts as significant only if the Holm-adjusted p is < 0.05 for BOTH tests. Headline metrics (bold): recall@20, ndcg@20.",
           "- **Effect size**: d_z = mean diff / std of the per-seed differences; the 95% CI is a t-interval on the mean difference.",
           "- **Limits**: with n=10 the smallest two-sided Wilcoxon p is 2/1024 ≈ 0.002 (only if MAGCL wins every seed). Seeds sample "
           "split/init noise, not users or datasets: the claim is 'better on this dataset', not 'better in general'. LightGCN is not "
           "hyperparameter-tuned beyond defaults, which favours MAGCL, whose lambdas/theta were tuned."]
    path = out / "study_report.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="ml1m,ifashion", help="comma list")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=2020)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "benchmark"))
    ap.add_argument("--report-only", action="store_true", help="skip training, just rebuild the report")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    unknown = [d for d in args.datasets.split(",") if d.strip() not in DATASETS]
    if unknown:
        sys.exit(f"unknown dataset(s) {unknown}; choose from {list(DATASETS)}")

    failed = [] if args.report_only else train(args)
    if args.dry_run:
        return
    path = write_report(args, Path(args.out))
    print(f"\nreport written: {path}")
    if failed:
        print(f"some seeds failed in: {failed} (rerun the same command to retry only those; the report uses whatever finished)")
        sys.exit(1)


if __name__ == "__main__":
    main()
