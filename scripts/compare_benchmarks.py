"""Paired significance test between two benchmark runs (outputs/benchmark/<tag>/results.csv).

    python scripts/compare_benchmarks.py magcl-baseline-ml1m lightgcn-ml1m
    python scripts/compare_benchmarks.py magcl-baseline-ifashion lightgcn-ifashion --split valid

Reports A - B per metric over the seeds both runs share: mean difference, 95% CI, paired t-test, Wilcoxon
signed-rank, Cohen's d_z, and seeds-won. Raw p-values AND Holm-adjusted p-values (family = the metrics
tested together, 6 by default), because testing 6 correlated metrics at alpha=0.05 inflates false positives.

Why paired: seed N draws the same random 8:1:1 split for every model, so A and B are measured on identical
data per seed and the split-to-split variance cancels out of the difference. This needs the two runs to use
the SAME seeds (the benchmark scripts default to 2020..2029).

Reading the output: 10 seeds is small. The smallest two-sided Wilcoxon p at n=10 is 2/1024 = 0.002, and
the t-test assumes the per-seed differences are roughly normal. Seeds are a sample of splits/inits, NOT of
users or datasets, so a significant result says "better on this dataset under split/init noise", not "better
in general". Saved to outputs/benchmark/compare_<A>_vs_<B>.csv.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
HEADLINE = ("recall@20", "ndcg@20")


def load(tag: str, split: str, out: Path) -> dict:
    """{seed: {metric: value}} for one run."""
    path = out / tag / "results.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run that benchmark first")
    rows = {}
    for r in csv.DictReader(open(path)):
        rows[int(r["seed"])] = {k[len(split) + 1:]: float(v) for k, v in r.items() if k.startswith(split + "_")}
    return rows


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (controls family-wise error; never weaker than Bonferroni)."""
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def compare(a_tag: str, b_tag: str, split: str, out: Path) -> tuple:
    """Paired tests of run A vs run B over their shared seeds -> (seeds, one dict of stats per metric)."""
    A, B = load(a_tag, split, out), load(b_tag, split, out)
    seeds = sorted(set(A) & set(B))
    if len(seeds) < 3:
        raise ValueError(f"only {len(seeds)} shared seed(s) between {a_tag} and {b_tag} -- need at least 3 for a paired test")

    metrics = list(A[seeds[0]])
    rows = []
    for m in metrics:
        a = np.array([A[s][m] for s in seeds])
        b = np.array([B[s][m] for s in seeds])
        d = a - b
        n, sd = len(d), d.std(ddof=1)
        se = sd / np.sqrt(n)
        ci = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=se) if se > 0 else (d.mean(), d.mean())
        t_p = stats.ttest_rel(a, b).pvalue
        w_p = stats.wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
        rows.append(dict(metric=m, mean_a=a.mean(), mean_b=b.mean(), diff=d.mean(), rel_pct=100 * d.mean() / b.mean(),
                         ci_lo=ci[0], ci_hi=ci[1], t_p=t_p, wilcoxon_p=w_p,
                         dz=d.mean() / sd if sd > 0 else float("inf"), a_wins=int((d > 0).sum()), n=n))
    for key in ("t_p", "wilcoxon_p"):
        adj = holm(np.array([r[key] for r in rows]))
        for r, v in zip(rows, adj):
            r[key + "_holm"] = v
    return seeds, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="tag of run A, e.g. magcl-baseline-ml1m")
    ap.add_argument("b", help="tag of run B, e.g. lightgcn-ml1m")
    ap.add_argument("--split", default="test", choices=["test", "valid"])
    ap.add_argument("--out", default=str(ROOT / "outputs" / "benchmark"))
    args = ap.parse_args()

    out = Path(args.out)
    try:
        seeds, rows = compare(args.a, args.b, args.split, out)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(e)

    print(f"\nA = {args.a}\nB = {args.b}\n{args.split} split, {len(seeds)} paired seeds {seeds[0]}..{seeds[-1]}, difference = A - B\n")
    print(f"{'metric':<11}{'A':>8}{'B':>8}{'A-B':>9}{'rel%':>8}   {'95% CI of A-B':<20}{'t p':>8}{'t p(Holm)':>11}{'W p':>8}{'W p(Holm)':>11}{'d_z':>7}{'A wins':>8}")
    for r in rows:
        star = "*" if r["metric"] in HEADLINE else " "
        print(f"{r['metric']:<10}{star}{r['mean_a']:>8.4f}{r['mean_b']:>8.4f}{r['diff']:>+9.4f}{r['rel_pct']:>+7.1f}%   "
              f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  {r['t_p']:>8.4f}{r['t_p_holm']:>11.4f}{r['wilcoxon_p']:>8.4f}{r['wilcoxon_p_holm']:>11.4f}"
              f"{r['dz']:>7.2f}{r['a_wins']:>5}/{r['n']}")
    print("\n* headline metrics. Significant at 0.05 only if the Holm column is < 0.05 (the CI excluding 0 agrees with the raw t p).")

    path = out / f"compare_{args.a}_vs_{args.b}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
