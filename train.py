"""
train.py
--------
Top-level training script for the Proposed Semantic-DTFD-MIL model.

Usage:
    python train.py                          # uses config.yaml defaults
    python train.py --epochs 30 --lr 1e-4
"""

import argparse, os, sys, json
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.dataset import PCamBagDataset
from src.utils   import compute_metrics, save_checkpoint, save_metrics, save_training_log
from src.model   import SemanticDTFDMIL


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train_one_epoch(model, dataset, optimizer, device):
    model.train()
    total_loss, all_probs, all_labels = 0.0, [], []

    for i in range(len(dataset)):
        pseudo_bags, label, _ = dataset[i]
        pseudo_bags = [pb.to(device) for pb in pseudo_bags]
        lbl_t = torch.tensor([label], dtype=torch.long).to(device)

        t2_logits, t1_logits, _ = model(pseudo_bags)

        # Tier-1 loss: each pseudo-bag individually
        t1_target = lbl_t.expand(t1_logits.size(0))
        loss1 = F.cross_entropy(t1_logits, t1_target)

        # Tier-2 loss: global bag prediction
        t2_target = lbl_t.expand(t2_logits.size(0))
        loss2 = F.cross_entropy(t2_logits, t2_target)

        loss = 0.5 * loss1 + 0.5 * loss2

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        prob = torch.softmax(t2_logits, dim=1)[:, 1].mean().item()
        all_probs.append(prob)
        all_labels.append(label)

    avg_loss = total_loss / len(dataset)
    return avg_loss, all_labels, all_probs


@torch.no_grad()
def evaluate(model, dataset, device):
    model.eval()
    all_probs, all_labels = [], []

    for i in range(len(dataset)):
        pseudo_bags, label, _ = dataset[i]
        pseudo_bags = [pb.to(device) for pb in pseudo_bags]

        t2_logits, _, _ = model(pseudo_bags)
        prob = torch.softmax(t2_logits, dim=1)[:, 1].mean().item()
        all_probs.append(prob)
        all_labels.append(label)

    return compute_metrics(all_labels, all_probs)


def main():
    cfg = load_config()
    m_cfg   = cfg["model"]
    t_cfg   = cfg["training"]
    d_cfg   = cfg["data"]
    p_cfg   = cfg["paths"]

    parser = argparse.ArgumentParser(description="Train Semantic-DTFD-MIL")
    parser.add_argument("--epochs",     type=int,   default=t_cfg["epochs"])
    parser.add_argument("--lr",         type=float, default=t_cfg["learning_rate"])
    parser.add_argument("--train_pkl",  type=str,   default=d_cfg["train_pkl"])
    parser.add_argument("--test_pkl",   type=str,   default=d_cfg["test_pkl"])
    parser.add_argument("--ckpt_dir",   type=str,   default=p_cfg["checkpoint_dir"])
    parser.add_argument("--device",     type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(p_cfg["results_dir"], exist_ok=True)

    device = torch.device(args.device)
    print(f"\nDevice     : {device}")
    print(f"Train PKL  : {args.train_pkl}")
    print(f"Epochs     : {args.epochs}")

    # ── Data ────────────────────────────────────────────────────────────────
    if not os.path.exists(args.train_pkl):
        print(f"ERROR: training PKL not found at {args.train_pkl}")
        sys.exit(1)

    train_ds = PCamBagDataset(args.train_pkl, num_groups=m_cfg["num_pseudobags"])
    test_ds  = PCamBagDataset(args.test_pkl,  num_groups=m_cfg["num_pseudobags"])
    feat_dim = train_ds.get_feature_dim()
    print(f"Train bags : {len(train_ds)}  |  Test bags: {len(test_ds)}")
    print(f"Feature dim: {feat_dim}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = SemanticDTFDMIL(
        in_chn  = feat_dim,
        m_dim   = m_cfg["hidden_dim"],
        num_cls = m_cfg["num_classes"],
        att_dim = m_cfg["attention_dim"],
        dropout = m_cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = args.lr,
        weight_decay = t_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size = t_cfg["lr_step_epoch"],
        gamma     = t_cfg["lr_decay_ratio"],
    )

    # ── Training loop ────────────────────────────────────────────────────────
    best_auc   = 0.0
    log        = []
    best_ckpt  = os.path.join(args.ckpt_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        loss, _, _ = train_one_epoch(model, train_ds, optimizer, device)
        val_m      = evaluate(model, test_ds, device)
        scheduler.step()

        row = {"epoch": epoch, "train_loss": round(loss, 4), **val_m}
        log.append(row)
        print(f"Ep {epoch:3d}/{args.epochs} | loss={loss:.4f} | AUC={val_m['auc']:.4f} | Acc={val_m['accuracy']:.4f}")

        if val_m["auc"] > best_auc:
            best_auc = val_m["auc"]
            save_checkpoint({
                "epoch":         epoch,
                "auc":           best_auc,
                "classifier":    model.classifier.state_dict(),
                "attention":     model.attention.state_dict(),
                "dim_reduction": model.dim_reduction.state_dict(),
                "att_classifier": model.att_classifier.state_dict(),
            }, best_ckpt)
            print(f"  *** New best AUC: {best_auc:.4f} — checkpoint saved ***")

    # ── Save results ─────────────────────────────────────────────────────────
    save_training_log(log, os.path.join(p_cfg["results_dir"], "training_log.csv"))
    final_m = evaluate(model, test_ds, device)
    save_metrics({
        "model":      "Proposed_SAM_PLIP",
        "best_auc":   best_auc,
        "final":      final_m,
        "epochs":     args.epochs,
        "checkpoint": best_ckpt,
    }, os.path.join(p_cfg["results_dir"], "improved_metrics.json"))

    print(f"\nTraining complete. Best AUC = {best_auc:.4f}")
    print(f"Checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
