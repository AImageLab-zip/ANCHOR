"""Run one federated experiment (FedAvg, NVIDIA FLARE simulator) on every cross-validation fold.

    MANIFEST_ROOT=<manifests> python -m anchor.fl.job --config configs/par/fl_anchor_full.yaml \\
        --features-root <features> --output-dir runs/par/anchor_full_s7 --seed 7 \\
        --set manifest_set=own_anchor34_u6 --set phase1_dir=runs/par/phase1_N34_s7 \\
        --set ai_offsets_dir=runs/par/ai_offsets_N34_s7

The YAML holds the server options (`clients`, `num_rounds`) and the client options of `anchor.fl.client` (underscored
names); `--set key=value` overrides any of them. Each fold runs as its own job `<output-dir>/<fold>`; the final global
model is `<output-dir>/<fold>/server/simulate_job/app_server/FL_global_model.pt` and each client's last local model
(with its private offset) `<output-dir>/<fold>/<client>/last_model.pt`.
"""
from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path

import yaml

GLOBAL_MODEL = Path("server/simulate_job/app_server/FL_global_model.pt")
SERVER_KEYS = {"clients", "num_rounds"}
MODEL_KEYS = ("num_classes", "input_dim", "attention_dim", "hidden_dim", "dropout")
MODEL_DEFAULTS = {"num_classes": 6, "input_dim": 1536, "attention_dim": 128, "hidden_dim": 64, "dropout": 0.5}


def parse_value(v: str):
    return yaml.safe_load(v)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--features-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--manifest-root", type=Path, default=None, help="Default: $MANIFEST_ROOT.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--folds", nargs="+", default=[f"fold_{i:02d}" for i in range(1, 6)])
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override a config key.")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def main() -> None:
    from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
    from nvflare.recipe import SimEnv

    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text()) or {}
    for item in args.set:
        key, value = item.split("=", 1)
        cfg[key] = parse_value(value)
    manifest_root = (args.manifest_root or Path(os.environ["MANIFEST_ROOT"])).resolve()
    clients, rounds = cfg.pop("clients"), cfg.pop("num_rounds")
    manifest_set = cfg.pop("manifest_set")
    model_args = {k: cfg.get(k, MODEL_DEFAULTS[k]) for k in MODEL_KEYS}
    for fold in args.folds:
        job_dir = args.output_dir.resolve() / fold
        if (job_dir / GLOBAL_MODEL).exists():
            if args.skip_existing:
                print(f"{fold}: done, skipped")
                continue
            raise FileExistsError(job_dir)
        client_args = {**cfg, "features_root": args.features_root.resolve(), "output_dir": job_dir,
                       "manifest_root": manifest_root, "manifest_set": f"{manifest_set}/{fold}", "seed": args.seed}
        for key in ("phase1_dir", "ai_offsets_dir"):
            if client_args.get(key):
                client_args[key] = Path(client_args[key]).resolve()
        argv = []
        for k, v in client_args.items():
            if v is None:
                continue
            argv += [f"--{k.replace('_', '-')}", str(v)]
        recipe = FedAvgRecipe(
            name=fold, min_clients=len(clients), num_rounds=rounds,
            model={"class_path": "anchor.models.GlobalOrdinalMIL", "args": model_args},
            train_script="anchor/fl/client.py", train_args=shlex.join(argv))  # run from the repository root
        run = recipe.execute(SimEnv(clients=clients, workspace_root=str(args.output_dir.resolve())))
        print(f"{fold}: {run.get_status()} -> {run.get_result()}")
        if not (job_dir / GLOBAL_MODEL).exists():
            raise RuntimeError(f"{fold}: no global model; check the client logs under {job_dir}")


if __name__ == "__main__":
    main()
