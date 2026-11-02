"""Bag datasets and manifests.

A *bag* is one biopsy: the patch embeddings of all its whole-slide images, stored as `<features_root>/<bag_id>.npz`
with an array `features` of shape (n_patches, dim) (optionally `coords` and `wsi_names`, unused here).

A *manifest* is a CSV with one row per bag and at least the columns
    bag_id, patient_id, label_<source> ...
where every `label_<source>` holds the grade given by that source (benign = 0, ISUP grade group 1-5) or -1 when the
source did not grade the bag. Sources are the centres (their own grades), the AI reader (`label_ai`) and, where a
method uses it, the median of the centres' grades on the anchors (`label_median`). Folds are directories
`<set>/fold_XX/{train,val,test}_manifest.csv`; `anchor.studies` builds them from a study table.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

MISSING = -1
SPLITS = ("train", "val", "test")


def read_fold(set_dir: Path, fold: str) -> dict[str, pd.DataFrame]:
    return {s: pd.read_csv(Path(set_dir) / fold / f"{s}_manifest.csv") for s in SPLITS}


def fold_names(set_dir: Path) -> list[str]:
    return sorted(p.name for p in Path(set_dir).iterdir() if p.is_dir() and p.name.startswith("fold_"))


class BagDataset(Dataset):
    """One bag per item: (features (n, dim), labels (S,), bag_id). Labels are the manifest columns `label_columns`."""

    def __init__(self, manifest: pd.DataFrame, features_root: Path, label_columns: Sequence[str],
                 max_bag_size: int = 0, sample: bool = False):
        missing = {"bag_id", *label_columns} - set(manifest.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        self.manifest = manifest.reset_index(drop=True)
        self.features_root = Path(features_root)
        self.label_columns = list(label_columns)
        self.max_bag_size = max_bag_size
        self.sample = sample

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        with np.load(self.features_root / f"{row['bag_id']}.npz", allow_pickle=True) as z:
            features = np.asarray(z["features"], dtype=np.float32)
        if self.sample and 0 < self.max_bag_size < len(features):
            features = features[np.random.choice(len(features), size=self.max_bag_size, replace=False)]
        labels = torch.tensor(row[self.label_columns].to_numpy(dtype=float), dtype=torch.float32)
        return torch.from_numpy(features), labels, row["bag_id"]


def collate_bags(batch):
    bags, labels, ids = zip(*batch)
    return list(bags), torch.stack(labels), list(ids)


def make_loader(manifest: pd.DataFrame, features_root: Path, label_columns: Sequence[str], batch_size: int,
                train: bool, max_bag_size: int = 0, num_workers: int = 4) -> DataLoader:
    """Training loaders shuffle and subsample bags to `max_bag_size` patches; evaluation loaders use every patch."""
    ds = BagDataset(manifest, features_root, label_columns, max_bag_size=max_bag_size, sample=train)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers, collate_fn=collate_bags)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
