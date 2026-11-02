"""Centralised training of the multi-reader ordinal MIL, fold by fold.

One case score, one head per label source (see `anchor.models.MultiReaderOrdinalMIL`); the loss of a bag is the sum
of the cumulative-link NLL of the grades it has (-1 = missing). Used for

  * ANCHOR phase 1     sources = the centres' columns on the anchor set (every anchor graded by every centre),
                       sum-zero offsets, epoch selection on the validation grades;
  * Crossed            sources = every centre's grade of every biopsy, sum-zero offsets, last epoch;
  * Crossed + AI       Crossed + the AI reader as a fourth source with its own learned offset, outside the sum-zero
                       centring (`--free-sources ai`), started at its marginal shift;
  * MedianLabels       one source, the median of the centres' grades, scored by theta (`--consensus-sources`).

    python -m anchor.train_central --config configs/par/central.yaml --folds-dir <manifests>/merged/anchor34_u6 \\
        --sources r1 r2 vir --epochs 105 --patience 35 --select-epoch --seed 7 --output-dir runs/par/phase1_N34_s7

Outputs per fold (`<output-dir>/fold_XX/`): model.pt, config.yaml, metrics.json, {val,test}_predictions.csv with
`prob_<head>_<k>` for head = consensus and every source.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from anchor.data import fold_names, make_loader, read_fold, seed_everything
from anchor.models import GatedAttentionMIL, MultiReaderOrdinalMIL, cumulative_logits, median_grade


def parse_args(argv=None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    known, _ = pre.parse_known_args(argv)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], parents=[pre])
    p.add_argument("--folds-dir", type=Path, required=False)
    p.add_argument("--features-root", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--sources", nargs="+", help="Label sources: manifest columns label_<source>.")
    p.add_argument("--consensus-sources", nargs="*", default=[], help="Sources scored by theta itself (offset 0).")
    p.add_argument("--free-sources", nargs="*", default=[],
                   help="External sources (the AI reader) with a learned offset outside the sum-zero centring; they do "
                        "not enter epoch selection or the initial cut-points.")
    p.add_argument("--external-weight", type=float, default=1.0, help="Loss weight of the free sources.")
    p.add_argument("--no-sum-zero", action="store_true", help="Do not centre the offsets (needs a consensus source).")
    p.add_argument("--no-offsets", action="store_true", help="One theta for every source (pooled reads).")
    p.add_argument("--init-checkpoint-dir", type=Path, help="Start each fold from <dir>/<fold>/model.pt.")
    p.add_argument("--freeze-offsets", action="store_true")
    p.add_argument("--select-epoch", action="store_true",
                   help="Keep the epoch with the best mean validation QWK of the sources (ties: detection AUC), with "
                        "early stopping after --patience epochs; default: the model after the last epoch.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--offset-lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-bag-size", type=int, default=512)
    p.add_argument("--num-classes", type=int, default=6)
    p.add_argument("--input-dim", type=int, default=1536)
    p.add_argument("--attention-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--folds", nargs="*", help="Subset of folds (default: all).")
    p.add_argument("--skip-existing", action="store_true")
    if known.config is not None:
        cfg = yaml.safe_load(known.config.read_text()) or {}
        unknown = set(cfg) - {a.dest for a in p._actions}
        if unknown:
            p.error(f"unknown config keys {sorted(unknown)}")
        p.set_defaults(**cfg)
    args = p.parse_args(argv)
    for key in ("folds_dir", "features_root", "output_dir", "sources"):
        if getattr(args, key) is None:
            p.error(f"--{key.replace('_', '-')} is required")
    return args


def build_model(args: argparse.Namespace, fixed_offsets: dict | None = None) -> MultiReaderOrdinalMIL:
    mil = GatedAttentionMIL(args.input_dim, args.attention_dim, args.hidden_dim, args.dropout)
    return MultiReaderOrdinalMIL(mil, args.num_classes, args.sources, args.consensus_sources,
                                 sum_zero=not args.no_sum_zero, learn_offsets=not args.no_offsets,
                                 fixed_offsets=fixed_offsets, free_sources=args.free_sources)


def multi_read_nll(log_probs: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """log_probs (B, 1 + S, K), labels (B, S) with -1 = missing -> mean over bags of the summed NLL of the grades."""
    present = labels >= 0
    picked = torch.gather(log_probs[:, 1:, :], 2, labels.clamp(min=0).long().unsqueeze(-1)).squeeze(-1)
    terms = picked * present
    if weights is not None:
        terms = terms * weights
    return (-terms.sum(dim=1)).mean()


def head_qwk_auc(y: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """QWK of the median-rule prediction and detection AUC (P(y >= 1)) on bags with a grade."""
    keep = y >= 0
    y, probs = y[keep], probs[keep]
    q = cohen_kappa_score(y, median_grade(probs), labels=list(range(probs.shape[1])), weights="quadratic")
    try:
        auc = roc_auc_score(y >= 1, 1 - probs[:, 0])
    except ValueError:
        auc = np.nan
    return (float(q) if np.isfinite(q) else np.nan), float(auc)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device) -> tuple[list[str], np.ndarray]:
    model.eval()
    ids, probs = [], []
    for bags, _, batch_ids in loader:
        log_probs, _ = model([b.to(device) for b in bags])
        probs.append(torch.exp(log_probs).cpu().numpy())
        ids += batch_ids
    return ids, np.concatenate(probs)


def save_predictions(ids: list[str], probs: np.ndarray, heads: list[str], path: Path) -> None:
    data = {"bag_id": ids}
    for h, head in enumerate(heads):
        data[f"pred_{head}"] = median_grade(probs[:, h])
        for k in range(probs.shape[2]):
            data[f"prob_{head}_{k}"] = probs[:, h, k]
    pd.DataFrame(data).to_csv(path, index=False)


def train_fold(args: argparse.Namespace, fold: str, device: torch.device) -> None:
    out = args.output_dir / fold
    if args.skip_existing and (out / "test_predictions.csv").exists():
        print(f"{fold}: exists, skipped")
        return
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    man = read_fold(args.folds_dir, fold)
    cols = [f"label_{s}" for s in args.sources]
    own = [s for s in args.sources if s not in args.free_sources]
    own_idx = [args.sources.index(s) for s in own]
    weights = torch.tensor([args.external_weight if s in args.free_sources else 1.0 for s in args.sources],
                           device=device) if args.free_sources else None

    model = build_model(args)
    if args.init_checkpoint_dir is not None:
        model.load_state_dict(torch.load(args.init_checkpoint_dir / fold / "model.pt", map_location="cpu"))
    else:
        reads = man["train"][[f"label_{s}" for s in own]].to_numpy().ravel()
        model.cutpoints.init_from_labels(reads[reads >= 0])
        theta0 = model.cutpoints.thresholds().detach()
        for s in args.free_sources:  # external source: start at its marginal shift from the initial cut-points
            y = man["train"][f"label_{s}"].to_numpy()
            logits = torch.as_tensor(cumulative_logits(y[y >= 0], args.num_classes), dtype=theta0.dtype)
            model.delta.data[model.offset_sources.index(s)] = logits - theta0
    if args.freeze_offsets:
        model.delta.requires_grad_(False)
    model = model.to(device)
    other = [p for n, p in model.named_parameters() if n != "delta"]
    groups = [{"params": other}] + ([] if args.freeze_offsets or args.no_offsets else
                                    [{"params": [model.delta], "lr": args.offset_lr, "weight_decay": 0.0}])
    opt = torch.optim.Adam(groups, lr=args.lr, weight_decay=args.weight_decay)

    loaders = {s: make_loader(man[s], args.features_root, cols, args.batch_size, train=(s == "train"),
                              max_bag_size=args.max_bag_size, num_workers=args.num_workers) for s in man}
    val_y = man["val"][cols].to_numpy().astype(int)
    best = (-np.inf, -np.inf, None, 0)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for bags, labels, _ in loaders["train"]:
            log_probs, _ = model([b.to(device) for b in bags])
            loss = multi_read_nll(log_probs, labels.to(device), weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(bags)
        _, val_p = predict(model, loaders["val"], device)
        scores = [head_qwk_auc(val_y[:, i], val_p[:, 1 + i]) for i in own_idx]
        val_qwk, val_auc = float(np.mean([q for q, _ in scores])), float(np.mean([a for _, a in scores]))
        history.append({"epoch": epoch, "loss": total / len(man["train"]), "val_qwk": val_qwk, "val_auc": val_auc})
        print(f"{fold} epoch {epoch}: loss {history[-1]['loss']:.3f} val QWK {val_qwk:.3f} AUC {val_auc:.3f}",
              flush=True)
        if not args.select_epoch:
            continue
        if val_qwk >= best[0]:
            if val_qwk > best[0] or val_auc > best[1]:
                best = (val_qwk, val_auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch)
        elif epoch - best[3] >= args.patience:
            print(f"{fold}: early stop at epoch {epoch} (best {best[3]})")
            break
    if best[2] is not None:
        model.load_state_dict(best[2])

    heads = ["consensus"] + list(args.sources)
    for split in ("val", "test"):
        ids, probs = predict(model, loaders[split], device)
        save_predictions(ids, probs, heads, out / f"{split}_predictions.csv")
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / "model.pt")
    (out / "config.yaml").write_text(yaml.safe_dump({k: str(v) if isinstance(v, Path) else v
                                                     for k, v in vars(args).items()}))
    (out / "metrics.json").write_text(json.dumps({
        "best_epoch": best[3] if args.select_epoch else args.epochs, "history": history,
        "theta": model.cutpoints.thresholds().detach().cpu().tolist(),
        "offset_sources": model.offset_sources, "delta": model.effective_delta().detach().cpu().tolist()}, indent=1))
    print(f"{fold}: theta {np.round(model.cutpoints.thresholds().detach().cpu().numpy(), 3).tolist()} "
          f"delta {np.round(model.effective_delta().detach().cpu().numpy(), 3).tolist()}")


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.no_sum_zero and not args.consensus_sources and not args.no_offsets:
        raise SystemExit("Offsets need sum-zero centring or a consensus source to be identifiable.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fold in args.folds or fold_names(args.folds_dir):
        train_fold(args, fold, device)


if __name__ == "__main__":
    main()
