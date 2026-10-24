"""Sweep the number of Graclus clusters K for MAGCL-Graclus (src/magcl_graclus), one dataset at a time.

    python scripts/sweep_graclus_k.py --dataset ml1m --dry-run             # make the .part.K files, print the plan
    python scripts/sweep_graclus_k.py --dataset ml1m                       # union mode, K = 4 8 16 32 64
    python scripts/sweep_graclus_k.py --dataset ifashion --modes union replace --ks 32 64 128
    python scripts/sweep_graclus_k.py --dataset ml1m --ks 8 --epochs 1     # wiring check (own MLflow experiment)

Per (mode, K): make graclus1.2/<graph>.part.K with the Graclus binary through WSL if missing, write an
override JSON (the dataset's tuned base override + mode + part file) and call scripts/run_graclus.py,
so every run is a normal run_graclus.py run. Run name = graclus-<mode>-k<K>; a (mode, K) already FINISHED
in the MLflow experiment is skipped, so the sweep can be stopped and restarted. mode "off" is plain MAGCL
(no clusters) and runs once, as the reference.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent.parent

# dataset -> (RecBole config, tuned base override with graclus_map, MLflow experiment run_graclus.py uses, default Ks)
PRESETS = {
    "ml1m": ("configs/magcl.yaml", "configs/graclus_ml1m_union.json", "MAGCL-graclus-ml-1m_preprocessed_byrating", [4, 8, 16, 32, 64]),
    "ifashion": ("configs/magcl_ifashion.yaml", "configs/graclus_ifashion_union.json", "MAGCL-graclus-Alibaba-iFashion", [16, 32, 64, 128, 256]),
}


def ensure_part(graph: Path, k: int) -> str:
    part = ROOT / "graclus1.2" / f"{graph.name}.part.{k}"
    if not part.exists():
        wsl_root = "/mnt/" + ROOT.drive[0].lower() + ROOT.as_posix()[2:]
        subprocess.run(["wsl.exe", "--exec", "bash", "-c",
                        f"cd {wsl_root}/graclus1.2 && ./graclus ../outputs/graclus/{graph.name} {k}"],
                       check=True, capture_output=True)
    return part.relative_to(ROOT).as_posix()


def already_done(experiment: str, run_name: str) -> bool:
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        return False
    runs = mlflow.search_runs([exp.experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}' and status = 'FINISHED'")
    return len(runs) > 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=PRESETS, required=True)
    ap.add_argument("--ks", type=int, nargs="+", help="cluster counts (default: the dataset's preset list)")
    ap.add_argument("--modes", nargs="+", default=["union"], choices=["union", "replace", "off"])
    ap.add_argument("--epochs", type=int, help="override epochs (wiring check); uses a separate MLflow experiment")
    ap.add_argument("--dry-run", action="store_true", help="make the part files and print the plan, train nothing")
    args = ap.parse_args()

    config, base_path, experiment, default_ks = PRESETS[args.dataset]
    base = json.loads((ROOT / base_path).read_text())
    graph = ROOT / "outputs" / "graclus" / (Path(base["graclus_map"]).name.removesuffix(".map.npz") + ".graph")
    if args.epochs:
        experiment += "-smoke"

    plan = []  # (run name, override dict)
    for mode in args.modes:
        for k in ([None] if mode == "off" else (args.ks or default_ks)):
            name = f"graclus-{mode}-k{k}" if k else "magcl-off"
            override = {key: v for key, v in base.items() if key != "graclus_part"}
            override["graclus_mode"] = mode
            if k:
                override["graclus_part"] = ensure_part(graph, k)
            if args.epochs:
                override["epochs"] = args.epochs
            plan.append((name, override))

    out = ROOT / "outputs" / "sweep"
    out.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    for i, (name, override) in enumerate(plan, 1):
        skip = not args.epochs and already_done(experiment, name)
        print(f"[{i}/{len(plan)}] {name}: {'already FINISHED, skipping' if skip else 'to run'}", flush=True)
        if skip or args.dry_run:
            continue
        path = out / f"{args.dataset}_{name}.json"
        path.write_text(json.dumps(override, indent=4))
        rc = subprocess.run([sys.executable, "scripts/run_graclus.py", "--config", config, "--override", str(path),
                             "--experiment", experiment, "--run-name", name, "--quiet"], cwd=ROOT).returncode
        if rc:
            print(f"  {name} failed (exit {rc}); continuing with the next one", flush=True)


if __name__ == "__main__":
    main()
