"""Benchmark AdaMCL over several seeds on MovieLens-1M, THEN Alibaba-iFashion.

    python scripts/benchmark_adamcl.py                    # 10 seeds each (2020..2029), same seeds as benchmark_all.py
    python scripts/benchmark_adamcl.py --seeds 5
    python scripts/benchmark_adamcl.py --datasets ml1m    # iFashion builds a ~300k x 300k user-user Jaccard graph per seed: slow
    python scripts/benchmark_adamcl.py --dry-run

Results in outputs/benchmark/adamcl-<dataset>/ (resumable; also in MLflow). Use the SAME seeds as the MAGCL /
LightGCN benchmarks -- the seed fixes the random 8:1:1 split, so equal seeds = equal splits = a valid paired test:

    python scripts/compare_benchmarks.py magcl-baseline-ml1m adamcl-ml1m
"""

from benchmark_baseline import main

if __name__ == "__main__":
    main(__doc__, label="AdaMCL", prefix="adamcl", script="scripts/run_adamcl.py",
         configs={"ml1m": "configs/adamcl.yaml", "ifashion": "configs/adamcl_ifashion.yaml"})
