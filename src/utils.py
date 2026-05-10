"""
src/utils.py
------------
Shared utility functions for training, evaluation, and visualisation.
"""

import os, json, csv
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


# ── Evaluation ──────────────────────────────────────────────────────────────

def compute_metrics(labels: list, probs: list) -> dict:
    """
    Compute AUC, accuracy and F1 from lists of true labels and P(tumor) probs.

    Args:
        labels : list of int  (0=Normal, 1=Tumor)
        probs  : list of float (P(Tumor) in [0,1])
    Returns:
        dict with keys auc, accuracy, f1
    """
    preds = [1 if p >= 0.5 else 0 for p in probs]
    auc   = roc_auc_score(labels, probs) if len(set(labels)) >= 2 else 0.0
    acc   = accuracy_score(labels, preds)
    f1    = f1_score(labels, preds, zero_division=0)
    return {"auc": round(auc, 4), "accuracy": round(acc, 4), "f1": round(f1, 4)}


def get_cam_1d(classifier, features: torch.Tensor) -> torch.Tensor:
    """
    Generate 1-D class activation map from a Classifier_1fc layer.
    Mirrors the util function used in Main_DTFD_MIL.py.
    """
    tweight = list(classifier.parameters())[-1]          # (num_cls, mDim)
    cam     = torch.einsum("ikj,kl->ilj", features, tweight.T)
    return cam


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device: str = "cpu") -> dict:
    return torch.load(path, map_location=device)


# ── Metrics I/O ──────────────────────────────────────────────────────────────

def save_metrics(metrics: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics → {path}")


def load_metrics(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_training_log(log: list[dict], path: str) -> None:
    """Save a list of epoch dicts to CSV."""
    if not log:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log[0].keys())
        writer.writeheader()
        writer.writerows(log)
    print(f"  Saved training log → {path}")


# ── Pretty print ─────────────────────────────────────────────────────────────

def print_metrics(metrics: dict, title: str = "Results") -> None:
    sep = "=" * 50
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    for k, v in metrics.items():
        print(f"  {k:<20} {v}")
    print(sep)


if __name__ == "__main__":
    labels = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    probs  = [0.1, 0.3, 0.7, 0.9, 0.2, 0.8, 0.4, 0.6, 0.75, 0.35]
    m = compute_metrics(labels, probs)
    print_metrics(m, "Demo Metrics")
