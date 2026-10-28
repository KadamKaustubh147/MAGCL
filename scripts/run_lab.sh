#!/usr/bin/env bash
# One entry point for the lab PC.
#
#   bash scripts/run_lab.sh                  # everything, in this order: jepa, simgcl, sgl, adamcl
#   bash scripts/run_lab.sh jepa             # only MAGCL + JEPA on iFashion (a single run, NOT a benchmark)
#   bash scripts/run_lab.sh sgl simgcl       # any subset of: jepa simgcl sgl adamcl
#   bash scripts/run_lab.sh clusters         # OPT-IN (not in the default run): MAGCL + JEPA on iFashion with each item-clustering
#                                            # method (graclus, metis, spectral, popularity, random) vs a no-JEPA reference
#
# Environment knobs:
#   SEEDS=10                 seeds per benchmark (2020..2029, same as the MAGCL / LightGCN benchmarks)
#   DATASETS=ml1m,ifashion   which datasets the three benchmarks cover (adamcl on ifashion is the slow one)
#   PYTHON="..."             interpreter (default: "uv run python" if uv exists, else "python")
#   CLUSTER_ARGS=""          extra flags for the clusters stage, e.g. CLUSTER_ARGS="--n-clusters 64,256 --seeds 3"
#   DRY_RUN=1                print the commands, run nothing
#
# Unattended:  nohup bash scripts/run_lab.sh > outputs/lab/all.log 2>&1 &
#
# Every benchmark is resumable (finished seeds are skipped), so after a crash or Ctrl-C just rerun the same
# command. One stage failing does not stop the next one; failures are listed at the end (exit code 1).

cd "$(dirname "$0")/.." || exit 1
mkdir -p outputs/lab

# src/ holds magcl, magcl_graclus, jepa_cf, recbole_gnn. Don't rely on the editable install's .pth (it can point
# at a stale checkout; seen locally), so import them straight from this checkout.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

SEEDS=${SEEDS:-10}
DATASETS=${DATASETS:-ml1m,ifashion}
ALL_STAGES="jepa simgcl sgl adamcl"      # cheap-to-expensive for the benchmarks; AdaMCL's iFashion graph build is last
STAGES=${*:-$ALL_STAGES}
OPT_STAGES="clusters"                    # valid by name only

PY=${PYTHON:-}
if [ -z "$PY" ]; then
    if command -v uv >/dev/null 2>&1; then PY="uv run python"; else PY=python; fi
fi

run() {   # run <logfile> <cmd...>: echo, run, tee to the log, return the command's own exit code
    local log=$1; shift
    echo; echo ">>> $*"
    [ -n "$DRY_RUN" ] && return 0
    "$@" 2>&1 | tee "$log"
    return "${PIPESTATUS[0]}"
}

# JEPA's item clustering imports torch_cluster.graclus_cluster, which is on no PyPI index and is NOT in
# pyproject.toml. PyG hosts wheels tagged per torch + CUDA version, so build that URL from the installed torch.
ensure_torch_cluster() {
    [ -n "$DRY_RUN" ] && return 0
    $PY -c "import torch_cluster" 2>/dev/null && return 0
    local url
    url=$($PY -c "import torch; c = torch.version.cuda; print('https://data.pyg.org/whl/torch-%s+%s.html' % (torch.__version__.split('+')[0], 'cu' + c.replace('.', '') if c else 'cpu'))") || return 1
    echo ">>> torch_cluster missing, installing from $url"
    # --no-deps: its only dependency is scipy, and an unpinned resolve could move it off the 1.13.1 RecBole needs
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$($PY -c 'import sys; print(sys.executable)')" --no-deps torch-cluster --find-links "$url"
    else
        $PY -m pip install --no-deps torch-cluster -f "$url"
    fi
}

stage_jepa() {
    ensure_torch_cluster || return 1
    # MAGCL's iFashion setting = benchmark_all.py's: configs/magcl_ifashion.yaml + the tuned normalised-CL
    # hparams (NOT the lam1-4 / theta inside the yaml, those are the ML-1M values), plus the JEPA overlay.
    run outputs/lab/jepa_ifashion.log $PY scripts/run_magcl_jepa.py \
        --config configs/magcl_ifashion.yaml configs/magcl_jepa.yaml \
        --override configs/best_ifashion_normcl_hparams.json \
        --experiment MAGCL-JEPA-ifashion --run-name magcl-jepa-ifashion --quiet
}

stage_clusters() {
    ensure_torch_cluster || return 1     # the graclus method needs it; metis needs pymetis (Linux wheels, declared in pyproject)
    run outputs/lab/jepa_clusters.log $PY scripts/sweep_jepa_clusters.py --dataset ifashion $CLUSTER_ARGS
}

stage_bench() {   # stage_bench <adamcl|sgl|simgcl>
    run "outputs/lab/benchmark_$1.log" $PY "scripts/benchmark_$1.py" --seeds "$SEEDS" --datasets "$DATASETS"
}

for s in $STAGES; do
    case " $ALL_STAGES $OPT_STAGES " in *" $s "*) ;; *) echo "unknown stage '$s'; choose from: $ALL_STAGES $OPT_STAGES" >&2; exit 2 ;; esac
done

failed=""
for s in $STAGES; do
    echo; echo "################ $s  ($(date '+%F %T'))"
    case "$s" in
        jepa) stage_jepa ;;
        clusters) stage_clusters ;;
        *) stage_bench "$s" ;;
    esac || failed="$failed $s"
done

echo; echo "################ finished  ($(date '+%F %T'))"
if [ -n "$failed" ]; then
    echo "FAILED stages:$failed  (benchmarks are resumable: rerun the same command to retry only what's missing)"
    exit 1
fi
echo "all stages OK. Benchmarks: outputs/benchmark/<model>-<dataset>/summary.csv; logs: outputs/lab/"
