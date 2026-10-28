"""Benchmark stock RecBole SGL over several seeds on MovieLens-1M, THEN Alibaba-iFashion.

    python scripts/benchmark_sgl.py                       # 10 seeds each (2020..2029), same seeds as benchmark_all.py
    python scripts/benchmark_sgl.py --seeds 5
    python scripts/benchmark_sgl.py --datasets ifashion
    python scripts/benchmark_sgl.py --dry-run

Results in outputs/benchmark/sgl-<dataset>/ (resumable; also in MLflow). Same seeds as the other benchmarks =
same splits = a valid paired test:

    python scripts/compare_benchmarks.py magcl-baseline-ml1m sgl-ml1m
"""

from benchmark_baseline import main

if __name__ == "__main__":
    main(__doc__, label="SGL", prefix="sgl", script="scripts/run_sgl.py",
         configs={"ml1m": "configs/sgl.yaml", "ifashion": "configs/sgl_ifashion.yaml"})
