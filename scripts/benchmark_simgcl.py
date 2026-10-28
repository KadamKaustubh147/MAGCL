"""Benchmark SimGCL (RecBole-GNN) over several seeds on MovieLens-1M, THEN Alibaba-iFashion.

    python scripts/benchmark_simgcl.py                    # 10 seeds each (2020..2029), same seeds as benchmark_all.py
    python scripts/benchmark_simgcl.py --seeds 5
    python scripts/benchmark_simgcl.py --datasets ifashion
    python scripts/benchmark_simgcl.py --dry-run

Results in outputs/benchmark/simgcl-<dataset>/ (resumable; also in MLflow). Same seeds as the other benchmarks =
same splits = a valid paired test:

    python scripts/compare_benchmarks.py magcl-baseline-ml1m simgcl-ml1m
"""

from benchmark_baseline import main

if __name__ == "__main__":
    main(__doc__, label="SimGCL", prefix="simgcl", script="scripts/run_simgcl.py",
         configs={"ml1m": "configs/simgcl.yaml", "ifashion": "configs/simgcl_ifashion.yaml"})
