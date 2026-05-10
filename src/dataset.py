"""
src/dataset.py
--------------
Dataset utilities for the DTFD-MIL project.
Handles loading the pre-extracted PKL bags (PLIP 512-dim features).
"""

import pickle
import torch
from torch.utils.data import Dataset


class PCamBagDataset(Dataset):
    """
    Dataset for pre-extracted PatchCamelyon (PCam) MIL bags.

    PKL structure:
        {slide_name: [{"feature": Tensor(512,), "label": int,
                        "cluster_id": int, "name": str}, ...]}

    Args:
        pkl_path   : path to the .pkl bag file
        num_groups : number of pseudo-bags to split each slide into (K)
    """

    def __init__(self, pkl_path: str, num_groups: int = 5):
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)
        self.slide_names = list(self.data.keys())
        self.num_groups  = num_groups

    def __len__(self) -> int:
        return len(self.slide_names)

    def __getitem__(self, idx: int):
        name    = self.slide_names[idx]
        patches = self.data[name]

        feats = torch.stack([p["feature"] for p in patches])   # (N, 512)
        label = int(patches[0]["label"])

        pseudo_bags = self._split(feats)
        return pseudo_bags, label, name

    def _split(self, feats: torch.Tensor) -> list[torch.Tensor]:
        """Randomly split patch features into K pseudo-bags."""
        idx  = torch.randperm(len(feats))
        size = max(1, len(feats) // self.num_groups)
        return [feats[idx[i * size:(i + 1) * size]] for i in range(self.num_groups)]

    def get_feature_dim(self) -> int:
        key = self.slide_names[0]
        return self.data[key][0]["feature"].shape[-1]

    def get_label_counts(self) -> dict:
        from collections import Counter
        labels = [int(self.data[n][0]["label"]) for n in self.slide_names]
        return dict(Counter(labels))


def load_pkl(path: str) -> dict:
    """Load a raw PKL bag dictionary."""
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import os
    pkl = os.path.join("assignment 3", "Dataset", "mDATA_test_PCAM.pkl")
    if os.path.exists(pkl):
        ds = PCamBagDataset(pkl)
        print(f"Slides         : {len(ds)}")
        print(f"Feature dim    : {ds.get_feature_dim()}")
        print(f"Label counts   : {ds.get_label_counts()}")
        bags, label, name = ds[0]
        print(f"Slide          : {name}  |  label={label}")
        print(f"Pseudo-bags    : {len(bags)}")
        print(f"Bag[0] shape   : {bags[0].shape}")
    else:
        print(f"PKL not found at: {pkl}")
