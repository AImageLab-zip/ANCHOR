"""NVFlare client: local epochs of one centre in FedAvg (or alone, for ANCHOR-Loc).

The centre's model is shared MIL + shared cut-points theta (averaged by the server) + a private offset delta_c that never
leaves the centre (`anchor.models.CentreOrdinalMIL`). What is trained is set by the options:

  --phase1-dir DIR      start round 1 from the phase-1 model DIR/<fold>/model.pt (MIL + theta) and set delta_c to this
                        centre's phase-1 offset (ANCHOR); without it theta starts at the logits of the centre's grade
                        distribution and delta_c at 0
  --offsets MODE        none: no private offset (BaseFed); fixed: delta_c fixed (ANCHOR); trainable: delta_c learned
                        locally (FedRep-type private head); zero: delta_c fixed at 0
  --second-column COL   second head on the same score for a second grade per bag (-1 = none), loss = own NLL +
                        second_weight * NLL of COL; cut-points theta + omega, omega = --ai-offsets-dir/<fold>/ai_offsets.json
                        (fixed) or 0 without it. COL = label_ai: the AI reader (ANCHOR-Full, BaseFed + AI with omega 0);
                        label_second: AI grade on own biopsies, anchors' median consensus on the anchors (ANCHOR-Loc);
                        label_median: the median on the anchors only (ANCHOR-Loc without AI reader)
  --consensus-flag COL  rows with COL > 0 use offset 0 on the second head (their second grade is the median consensus)
  --glc                 GLC: rows with `trusted` = 1 (anchors, median grade) use the plain NLL, the others the
                        forward-corrected NLL -log sum_z p(z | x) C[z, y] with C = <fold dir>/transition.json

Manifests: $MANIFEST_ROOT/<site>_client/<manifest set>/<fold>/{train,val,test}_manifest.csv with `label` = the
centre's own grade (and `label_ai`, the flag column, `trusted` where used). The Adam optimiser is created anew in every
round, as in the experiments of the paper.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nvflare.client as flare
import pandas as pd
import torch
import yaml
from torch.nn import functional as F

from anchor.data import make_loader, seed_everything
from anchor.models import CentreOrdinalMIL, GatedAttentionMIL


def parse_bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "yes")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--features-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--manifest-root", type=Path, default=None, help="Default: $MANIFEST_ROOT.")
    p.add_argument("--manifest-set", required=True, help="<set>/<fold>")
    p.add_argument("--phase1-dir", type=Path, default=None)
    p.add_argument("--offsets", choices=["none", "fixed", "trainable", "zero"], default="none")
    p.add_argument("--second-column", default=None)
    p.add_argument("--ai-offsets-dir", type=Path, default=None)
    p.add_argument("--second-weight", type=float, default=1.0)
    p.add_argument("--consensus-flag", default=None)
    p.add_argument("--glc", type=parse_bool, default=False)
    p.add_argument("--epochs", type=int, default=1, help="Local epochs per round.")
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
    p.add_argument("--aggregation-weight", type=float, default=1.0,
                   help="Multiply this centre's FedAvg weight (number of local steps).")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def phase1_offset(phase1_dir: Path, fold: str, site: str) -> tuple[dict, torch.Tensor]:
    """Phase-1 state (MIL + theta) and this centre's sum-zero offset."""
    state = torch.load(phase1_dir / fold / "model.pt", map_location="cpu")
    sources = yaml.safe_load((phase1_dir / fold / "config.yaml").read_text())["sources"]
    delta = state["delta"] - state["delta"].mean(dim=0, keepdim=True)
    return state, delta[sources.index(site)]


def train_round(model, loader, args, device, ai_offset, transition) -> float:
    """Local epochs of one round; columns of the batch labels: own grade [, second grade [, flag]] or [own, trusted]."""
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                           weight_decay=args.weight_decay)
    log_c = None if transition is None else torch.log(transition)
    total, n = 0.0, 0
    for _ in range(args.epochs):
        model.train()
        for bags, labels, _ in loader:
            bags = [b.to(device) for b in bags]
            labels = labels.to(device).long()
            scores, _ = model.mil(bags)
            own = model.cutpoints(scores, model.delta().expand(len(scores), -1))
            y = labels[:, 0]
            if transition is not None:  # GLC: the consensus head is p(z | x), centre grades are noisy reads of it
                log_read = torch.logsumexp(own.unsqueeze(-1) + log_c.unsqueeze(0), dim=1)
                trusted = labels[:, 1] == 1
                loss = torch.where(trusted, -own.gather(1, y[:, None])[:, 0],
                                   -log_read.gather(1, y[:, None])[:, 0]).mean()
            else:
                loss = F.nll_loss(own, y)
                if ai_offset is not None:
                    offset = ai_offset.expand(len(scores), -1)
                    if labels.shape[1] > 2:  # consensus-flag rows: median grade on the consensus head (offset 0)
                        offset = offset * (labels[:, 2:3] <= 0)
                    second = model.cutpoints(scores, offset)
                    has = labels[:, 1] >= 0
                    if has.any():
                        loss = loss + args.second_weight * F.nll_loss(second[has], labels[has, 1], reduction="sum") / len(bags)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(bags)
            n += len(bags)
    return total / max(n, 1)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flare.init()
    client = flare.system_info()["site_name"]
    site = client.removesuffix("_client")
    fold = Path(args.manifest_set).name
    out = args.output_dir / client / "results"
    out.mkdir(parents=True, exist_ok=True)
    root = args.manifest_root or Path(os.environ["MANIFEST_ROOT"])
    set_dir = root / client / args.manifest_set
    train = pd.read_csv(set_dir / "train_manifest.csv")

    if args.second_column is not None and args.glc:
        raise SystemExit("--second-column and --glc are alternatives.")
    if (args.ai_offsets_dir or args.consensus_flag) and not args.second_column:
        raise SystemExit("--ai-offsets-dir / --consensus-flag need --second-column.")
    ai_offset = transition = None
    columns = ["label"]
    if args.second_column is not None:
        omega = (json.loads((args.ai_offsets_dir / fold / "ai_offsets.json").read_text())
                 if args.ai_offsets_dir else [0.0] * (args.num_classes - 1))
        ai_offset = torch.tensor(omega, dtype=torch.float32, device=device)
        columns += [args.second_column] + ([args.consensus_flag] if args.consensus_flag else [])
    if args.glc:
        transition = torch.tensor(json.loads((set_dir / "transition.json").read_text())["C"], dtype=torch.float32,
                                  device=device)
        columns += ["trusted"]
    loader = make_loader(train, args.features_root, columns, args.batch_size, train=True,
                         max_bag_size=args.max_bag_size, num_workers=args.num_workers)

    mil = GatedAttentionMIL(args.input_dim, args.attention_dim, args.hidden_dim, args.dropout)
    model = CentreOrdinalMIL(mil, args.num_classes, offset_scale=args.offset_lr / args.lr).to(device)
    model.delta_raw.requires_grad_(args.offsets == "trainable")
    init_state = init_delta = None
    if args.phase1_dir is not None:
        init_state, init_delta = phase1_offset(args.phase1_dir, fold, site)
        print(f"{client}: phase-1 offset {init_delta.numpy().round(3).tolist()}")
    first_round = True

    while flare.is_running():
        received = flare.receive()
        if flare.is_submit_model():
            flare.send(flare.FLModel(params={k: v for k, v in model.cpu().state_dict().items() if k != "delta_raw"}))
            continue
        missing, unexpected = model.load_state_dict(received.params, strict=False)
        assert not unexpected and set(missing) <= {"delta_raw"}, (missing, unexpected)
        model.to(device)
        if flare.is_evaluate():
            flare.send(flare.FLModel(metrics={}))
            continue
        if first_round:
            if init_state is not None:
                model.load_state_dict({k: v for k, v in init_state.items() if k.startswith(("mil.", "cutpoints."))},
                                      strict=False)
            else:
                model.cutpoints.init_from_labels(train["label"].to_numpy())
            with torch.no_grad():
                if args.offsets in ("fixed", "trainable") and init_delta is not None:
                    model.delta_raw.copy_(init_delta.to(device) / model.offset_scale)
                else:
                    model.delta_raw.zero_()
            first_round = False
        loss = train_round(model, loader, args, device, ai_offset, transition)
        r = received.current_round
        print(f"{client} round {r}: loss {loss:.4f} delta {model.delta().detach().cpu().numpy().round(3).tolist()}",
              flush=True)
        params = {k: v for k, v in model.cpu().state_dict().items() if k != "delta_raw"}
        torch.save(model.state_dict(), out / "last_model.pt")
        (out / "delta.json").write_text(json.dumps(model.delta().detach().cpu().tolist()))  # applied private offset
        steps = round(len(loader) * args.epochs * args.aggregation_weight)
        flare.send(flare.FLModel(params=params, metrics={"neg_loss": -loss}, meta={"NUM_STEPS_CURRENT_ROUND": steps}))


if __name__ == "__main__":
    main()
