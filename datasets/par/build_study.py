"""Three-centre study on the public PAR cohort: slide table, pseudo-centres, folds, anchors and manifests.

PAR (Muhammad Ali et al., 2026) is a prostate-biopsy cohort scanned on three scanners and graded by pathologists and
AI models. To mimic a multi-centre setting, its patients are split into three pseudo-centres, each with its own reader
and its own scanner:

    centre  reader (its grading convention)        bag = UNI2-h features of the slide scanned on
    r1      pathologist 1                          Leica
    r2      pathologist 2                          Hamamatsu
    vir     AI model (Virchow2-based), Grundium    Grundium
    AI reader common to all centres: the task-specific AI model (TS) on the scan of the slide's own centre.

Consensus (evaluation only) = median of the r1, r2 and vir grades. Only slides graded by both pathologists are used.

1. Pseudo-centres: patients in thirds, StratifiedKFold(3, shuffle, seed 42) on the patient's maximum
   round-half-up((r1 + r2) / 2).
2. Folds: five patient-grouped folds (StratifiedGroupKFold, seed 42, stratified on the same slide grade); fold k tests
   base fold k, validates on base fold k + 1 and trains on the rest.
3. Anchors (per fold, training split only, `--draws` panels): each centre contributes N // 3 of its own training
   biopsies (remainder to the largest centre), drawn uniformly over the grades its own reader assigned (round-robin over
   benign, GG1-GG5 in a random order; rng [11, fold, 2, 68, draw]), drawn once for N = 68 and cut, so panels are nested
   over N. Every centre grades every anchor.
4. Manifest sets under `--out` (`<set>/fold_XX/{train,val,test}_manifest.csv`; -1 = not graded):
     merged/crossed               label_<c> = every centre's grade of every biopsy, label_ai, label_median
                                  (Crossed, Crossed + AI, MedianLabels)
     merged/nested                label_<c> = centre c's grade on its own biopsies only (TraceReg)
     merged/anchors<N>_d<D>       train = the anchors with every grade; val / test as nested (ANCHOR phase 1)
     merged/nested_anchors<N>_d<D> nested + every grade on the anchors (TraceReg + anchors)
     <c>_client/own               label = own grade, label_ai (BaseFed, FedRep-type, BaseFed + AI)
     <c>_client/own_anchors<N>_d<D>  own biopsies + the other centres' anchors graded by c; label_ai (-1 on the
                                  other centres' anchors), label_median and anchor = 1 on the anchors, label_second =
                                  label_ai with the anchors' median on the anchors (ANCHOR-Fed / -Full / -Loc)
     <c>_client/own_anchors<N>_d<D>_glc  same rows, anchors relabelled with the median, trusted = 1 on them,
                                  transition.json = P(c grades j | median i) on the anchors (GLC)
5. `<out>/slides.csv`: one row per slide (centre, patient, grades, AI grade, consensus); `<features>/par3c/<bag>.npz`:
   symlinks to the npz of the centre's scanner.

    python datasets/par/build_study.py --labels <PAR>/annotations/slide_labels.tsv --ai-grades <ESM1>.csv \\
        --features <features> --out <manifests>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

CENTRES = ["r1", "r2", "vir"]
SCANNER = {"r1": "leica", "r2": "hamamatsu", "vir": "grundium"}
FEATURE_DIR = {"leica": "leica", "hamamatsu": "hamamatsu_tf", "grundium": "grundium_tf"}
PATHOLOGIST = {"r1": "isup_1", "r2": "isup_r2"}
CENTRE_AI, EXTERNAL_AI = "AI_Virchow_Grundium_ISUP", "TS"
SEED, N_FOLDS, K = 42, 5, 6
NEST_N, RULE_KEY = 68, 2
SPLITS = ("train", "val", "test")


def slide_table(labels: Path, ai_grades: Path) -> pd.DataFrame:
    tsv = pd.read_csv(labels, sep="\t", dtype={"slide_id": str})
    ai = pd.read_csv(ai_grades).set_index("ID")
    s = pd.DataFrame({"slide_id": tsv["slide_id"], "patient_id": tsv["slide_id"].str[:4]})
    for c, col in PATHOLOGIST.items():
        s[c] = tsv[col].astype("Int64")
    s["vir"] = ai.loc[s["slide_id"], CENTRE_AI].astype(int).to_numpy()
    for sc in ("Leica", "Hamamatsu", "Grundium"):
        s[f"ts_{sc.lower()}"] = ai.loc[s["slide_id"], f"AI_{EXTERNAL_AI}_{sc}_ISUP"].astype(int).to_numpy()
    s = s[s["r1"].notna() & s["r2"].notna()].copy()
    s["strat"] = np.floor((s["r1"] + s["r2"]).astype(float) / 2 + 0.5).astype(int)
    s["median"] = np.median(s[CENTRES].astype(int).to_numpy(), axis=1).astype(int)
    return s


def pseudo_centres(s: pd.DataFrame) -> pd.Series:
    patients = s.groupby("patient_id")["strat"].max().sort_index()
    centre = pd.Series("", index=patients.index)
    for k, (_, idx) in enumerate(StratifiedKFold(len(CENTRES), shuffle=True, random_state=SEED)
                                 .split(patients.index, patients.to_numpy())):
        centre.iloc[idx] = CENTRES[k]
    return centre


def folds(s: pd.DataFrame) -> list[dict[str, set]]:
    base = [set(s.iloc[idx]["patient_id"]) for _, idx in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
            .split(s, s["strat"].to_numpy(), s["patient_id"].to_numpy())]
    every = set().union(*base)
    return [{"test": t, "val": base[(i + 1) % N_FOLDS], "train": every - t - base[(i + 1) % N_FOLDS]}
            for i, t in enumerate(base)]


def pick_uniform(own: pd.Series, n: int, rng: np.random.Generator) -> list[str]:
    """Round-robin over the grade levels in a random order, uniformly within each level; empty levels are skipped."""
    pools = {g: list(rng.permutation(own.index[own == g].to_numpy())) for g in range(K)}
    order = rng.permutation(K)
    picked = []
    while len(picked) < n and any(pools.values()):
        for g in order:
            if pools[g] and len(picked) < n:
                picked.append(pools[g].pop())
    return picked


def quota(n: int, sizes: dict[str, int]) -> dict[str, int]:
    q = {c: n // len(CENTRES) for c in CENTRES}
    q[max(sizes, key=sizes.get)] += n - len(CENTRES) * (n // len(CENTRES))
    return q


def anchor_panels(train: pd.DataFrame, fold_index: int, ns: list[int], draw: int) -> dict[int, list[str]]:
    """train: bag_id, centre, own -> {N: anchors}, nested over N."""
    sizes = train["centre"].value_counts().reindex(CENTRES).to_dict()
    full = quota(NEST_N, sizes)
    rng = np.random.default_rng([11, fold_index, RULE_KEY, NEST_N, draw])
    lists = {}
    for c in CENTRES:
        b = train[train["centre"] == c].sort_values("bag_id").set_index("bag_id")
        lists[c] = pick_uniform(b["own"].astype(int), min(full[c], len(b)), rng)
    return {n: [b for c in CENTRES for b in lists[c][:quota(n, sizes)[c]]] for n in ns}


def transition(median: np.ndarray, read: np.ndarray, alpha: float = 1.0, floor: float = 1e-3) -> np.ndarray:
    counts = np.zeros((K, K))
    np.add.at(counts, (median, read), 1.0)
    counts += alpha * np.eye(K)
    c = np.maximum(counts / counts.sum(1, keepdims=True), floor)
    return c / c.sum(1, keepdims=True)


class Writer:
    def __init__(self, out: Path):
        self.out = out

    def write(self, set_path: str, fold: str, split: str, df: pd.DataFrame) -> None:
        d = self.out / set_path / fold
        d.mkdir(parents=True, exist_ok=True)
        df.to_csv(d / f"{split}_manifest.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--labels", type=Path, required=True, help="PAR annotations/slide_labels.tsv")
    p.add_argument("--ai-grades", type=Path, required=True, help="AI grades (ESM 1 of the PAR validation paper)")
    p.add_argument("--features", type=Path, required=True,
                   help="Root with leica/, hamamatsu_tf/, grundium_tf/ (npz per slide scan); par3c/ links are added")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--anchors", type=int, nargs="+", default=[34])
    p.add_argument("--draws", type=int, nargs="+", default=[0, 1, 2], help="Anchor panels (paper: draw = seed - 7).")
    args = p.parse_args()

    s = slide_table(args.labels, args.ai_grades)
    centre_of_patient = pseudo_centres(s)
    s["centre"] = s["patient_id"].map(centre_of_patient)
    s["bag_id"] = [f"{sid.lower()}_{SCANNER[c]}" for sid, c in zip(s["slide_id"], s["centre"])]
    s["ai"] = [s.at[i, f"ts_{SCANNER[c]}"] for i, c in zip(s.index, s["centre"])]
    links = args.features / "par3c"
    links.mkdir(parents=True, exist_ok=True)
    for bag, c in zip(s["bag_id"], s["centre"]):
        target = args.features / FEATURE_DIR[SCANNER[c]] / f"{bag}.npz"
        if not target.exists():
            raise FileNotFoundError(target)
        if not (links / f"{bag}.npz").is_symlink():
            (links / f"{bag}.npz").symlink_to(Path("..") / FEATURE_DIR[SCANNER[c]] / f"{bag}.npz")
    args.out.mkdir(parents=True, exist_ok=True)
    s[["bag_id", "slide_id", "patient_id", "centre", *CENTRES, "ai", "median"]].rename(
        columns={"centre": "site", "median": "consensus"}).to_csv(args.out / "slides.csv", index=False)
    print(f"{len(s)} slides, {s['patient_id'].nunique()} patients; per centre "
          + ", ".join(f"{c} {int((s['centre'] == c).sum())}" for c in CENTRES))

    w = Writer(args.out)
    for i, split_patients in enumerate(folds(s)):
        fold = f"fold_{i + 1:02d}"
        rows = {sp: s[s["patient_id"].isin(split_patients[sp])].assign(subregion_id=lambda d: d["slide_id"])
                .sort_values(["patient_id", "subregion_id", "bag_id"]).reset_index(drop=True) for sp in SPLITS}
        frames = {}
        for sp, r in rows.items():
            base = r[["bag_id", "patient_id"]]
            reads = {f"label_{c}": r[c].astype(int).to_numpy() for c in CENTRES}
            own = {f"label_{c}": np.where(r["centre"] == c, r[c].astype(int), -1) for c in CENTRES}
            crossed = base.assign(**reads, label_ai=r["ai"].to_numpy(), label_median=r["median"].to_numpy())
            nested = base.assign(**own, label_ai=r["ai"].to_numpy())
            frames[sp] = (crossed, nested, r)
            w.write("merged/crossed", fold, sp, crossed)
            w.write("merged/nested", fold, sp, nested)
            for c in CENTRES:
                m = (r["centre"] == c).to_numpy()
                w.write(f"{c}_client/own", fold, sp, base[m].assign(label=r.loc[m, c].astype(int).to_numpy(),
                                                                    label_ai=r.loc[m, "ai"].to_numpy()))
        crossed_tr, nested_tr, r_tr = frames["train"]
        train = nested_tr.assign(centre=r_tr["centre"].to_numpy(),
                                 own=nested_tr[[f"label_{c}" for c in CENTRES]].max(axis=1).to_numpy())
        for d in args.draws:
            panels = anchor_panels(train, i, args.anchors, d)
            for n in args.anchors:
                tag = f"anchors{n}_d{d}"
                is_a = train["bag_id"].isin(panels[n]).to_numpy()
                full = nested_tr.copy()
                for c in CENTRES:
                    full.loc[is_a, f"label_{c}"] = crossed_tr.loc[is_a, f"label_{c}"]
                w.write(f"merged/{tag}", fold, "train", full[is_a].reset_index(drop=True))
                w.write(f"merged/nested_{tag}", fold, "train", full)
                for sp in ("val", "test"):
                    for name in (tag, f"nested_{tag}"):
                        w.write(f"merged/{name}", fold, sp, frames[sp][1])
                median = crossed_tr.loc[is_a].set_index("bag_id")["label_median"]
                anchor_rows = crossed_tr[is_a]
                for c in CENTRES:
                    m = (r_tr["centre"] == c).to_numpy()
                    own_tr = crossed_tr[m][["bag_id", "patient_id"]].assign(
                        label=crossed_tr.loc[m, f"label_{c}"].to_numpy(), label_ai=crossed_tr.loc[m, "label_ai"].to_numpy())
                    other = anchor_rows[(r_tr.loc[is_a, "centre"] != c).to_numpy()]
                    extra = other[["bag_id", "patient_id"]].assign(label=other[f"label_{c}"].to_numpy(), label_ai=-1)
                    t = pd.concat([own_tr, extra], ignore_index=True)
                    a = t["bag_id"].isin(median.index).to_numpy()
                    med = t["bag_id"].map(median).fillna(-1).astype(int).to_numpy()
                    t = t.assign(anchor=a.astype(int), label_median=med,
                                 label_second=np.where(a, med, t["label_ai"].to_numpy()))
                    w.write(f"{c}_client/own_{tag}", fold, "train", t)
                    glc = t[["bag_id", "patient_id", "label", "label_ai"]].assign(trusted=a.astype(int))
                    glc.loc[a, "label"] = med[a]
                    w.write(f"{c}_client/own_{tag}_glc", fold, "train", glc)
                    C = transition(median.to_numpy(), anchor_rows.set_index("bag_id").loc[median.index, f"label_{c}"]
                                   .to_numpy())
                    (args.out / f"{c}_client/own_{tag}_glc" / fold / "transition.json").write_text(
                        json.dumps({"centre": c, "C": C.round(6).tolist()}))
                    for sp in ("val", "test"):
                        r = frames[sp][2]
                        m = (r["centre"] == c).to_numpy()
                        ev = r[m][["bag_id", "patient_id"]].assign(label=r.loc[m, c].astype(int).to_numpy(),
                                                                  label_ai=r.loc[m, "ai"].to_numpy())
                        for name in (f"own_{tag}", f"own_{tag}_glc"):
                            w.write(f"{c}_client/{name}", fold, sp, ev)
        print(f"{fold}: train / val / test = " + " / ".join(str(len(rows[sp])) for sp in SPLITS))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
