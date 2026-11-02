"""TraceReg competitor (crowd layer + trace regulariser, Tanno et al., CVPR 2019), centralised, fold by fold.

The loss is that of `RegCrowdNetClassifier` in the maml library (github.com/ies-research/multi-annotator-machine-learning,
BSD-3): a categorical MIL head p(z | x) over benign, GG1-GG5, one confusion matrix per annotator (the centres, optionally
the AI reader), read y ~ sum_z p(z | x) C_a[z, y], plus lambda * mean trace of the confusions; confusions initialised at
logits 6 I - 5, lambda = 1e-2, their learning rate 1e-2 (the offsets' learning rate). Same MIL, optimiser, batch and
30 epochs (last epoch) as the central references.

  N    every centre's grades of its own biopsies       --folds-dir <manifests>/merged/nested
  N+A  + every centre's grades of the anchors           --folds-dir <manifests>/merged/nested_anchors34_d0

Training columns: label_<centre> (-1 = not graded). Predictions: head `consensus` = p(z | x), head `<centre>`
= p(z | x) C_centre, in the layout of `anchor.train_central`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from anchor.data import fold_names, make_loader, read_fold, seed_everything
from anchor.models import GatedAttentionMIL
from anchor.train_central import save_predictions


def crowd_loss(z: torch.Tensor, logits: torch.Tensor, confusion_logits: torch.Tensor, lmbda: float) -> torch.Tensor:
    """z (B, A) reads (-1 missing), logits (B, K), confusion_logits (A, K, K)."""
    B, A = z.shape
    pairs = torch.cartesian_prod(torch.arange(B, device=z.device), torch.arange(A, device=z.device))
    z = z.ravel()
    keep = z != -1
    pairs, z = pairs[keep], z[keep]
    log_p = F.log_softmax(logits, dim=-1)
    log_conf = torch.log_softmax(confusion_logits, dim=-1)
    log_read = torch.logsumexp(log_p[pairs[:, 0]][:, :, None] + log_conf[pairs[:, 1]], dim=1)
    reg = F.softmax(confusion_logits, dim=-1).diagonal(dim1=-2, dim2=-1).sum(-1).mean()
    return F.nll_loss(log_read, z) + lmbda * reg


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, help="YAML with the shared hyper-parameters (configs/par/central.yaml).")
    p.add_argument("--folds-dir", type=Path, required=True)
    p.add_argument("--features-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--centres", nargs="+", required=True)
    p.add_argument("--ai-column", default=None, help="Add the AI reader (e.g. label_ai) as one more annotator.")
    p.add_argument("--lmbda", type=float, default=1e-2)
    p.add_argument("--confusion-lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--folds", nargs="*")
    p.add_argument("--epochs", type=int)
    args = p.parse_args(argv)
    cfg = {"epochs": 30, "batch_size": 32, "lr": 5e-4, "weight_decay": 1e-4, "max_bag_size": 512, "num_classes": 6,
           "input_dim": 1536, "attention_dim": 128, "hidden_dim": 64, "dropout": 0.5, "num_workers": 4}
    if args.config:
        cfg.update({k: v for k, v in (yaml.safe_load(args.config.read_text()) or {}).items() if k in cfg})
    if args.epochs:
        cfg["epochs"] = args.epochs
    K = cfg["num_classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cols = [f"label_{c}" for c in args.centres]
    annot = cols + ([args.ai_column] if args.ai_column else [])
    for fold in args.folds or fold_names(args.folds_dir):
        out = args.output_dir / fold
        if (out / "test_predictions.csv").exists():
            continue
        seed_everything(args.seed)
        man = read_fold(args.folds_dir, fold)
        train = man["train"][(man["train"][cols] >= 0).any(axis=1)].reset_index(drop=True)
        model = GatedAttentionMIL(cfg["input_dim"], cfg["attention_dim"], cfg["hidden_dim"], cfg["dropout"],
                                  output_dim=K).to(device)
        conf = nn.Parameter(torch.stack([6.0 * torch.eye(K) - 5.0] * len(annot)).to(device))
        opt = torch.optim.Adam([{"params": list(model.parameters())}, {"params": [conf], "lr": args.confusion_lr}],
                               lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        loader = make_loader(train, args.features_root, annot, cfg["batch_size"], train=True,
                             max_bag_size=cfg["max_bag_size"], num_workers=cfg["num_workers"])
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            total = 0.0
            for bags, labels, _ in loader:
                logits, _ = model([b.to(device) for b in bags])
                loss = crowd_loss(labels.to(device).long(), logits, conf, args.lmbda)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(bags)
            print(f"{fold} epoch {epoch}: loss {total / len(train):.4f}", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        C = torch.softmax(conf.detach(), dim=-1)[:len(cols)]
        model.eval()
        for split in ("val", "test"):
            ids, probs = [], []
            with torch.no_grad():
                for bags, _, batch_ids in make_loader(man[split], args.features_root, [], cfg["batch_size"],
                                                      train=False, num_workers=cfg["num_workers"]):
                    pz = torch.softmax(model([b.to(device) for b in bags])[0], dim=-1)
                    probs.append(torch.cat([pz.unsqueeze(1), torch.einsum("bz,szj->bsj", pz, C)], 1).cpu().numpy())
                    ids += batch_ids
            save_predictions(ids, np.concatenate(probs), ["consensus"] + list(args.centres),
                             out / f"{split}_predictions.csv")
        (out / "confusions.json").write_text(json.dumps({"annotators": annot,
                                                         "C": torch.softmax(conf.detach(), -1).cpu().tolist()}))
        torch.save({"model": model.state_dict(), "confusion_logits": conf.detach().cpu()}, out / "model.pt")


if __name__ == "__main__":
    main()
