"""
inference.py
------------
Run inference on the held-out test set using the saved checkpoint.

Usage:
    python inference.py
    python inference.py --checkpoint checkpoints/best_model.pth --test_pkl "assignment 3/Dataset/mDATA_test_PCAM.pkl"
"""

import argparse, os, sys, json
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.dataset import PCamBagDataset
from src.utils   import compute_metrics, save_metrics, print_metrics
from src.model   import SemanticDTFDMIL


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    cfg    = load_config()
    m_cfg  = cfg["model"]
    d_cfg  = cfg["data"]
    p_cfg  = cfg["paths"]

    parser = argparse.ArgumentParser(description="DTFD-MIL Inference")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join("assignment 3", "debug_log", "best_model.pth"))
    parser.add_argument("--test_pkl",   type=str, default=d_cfg["test_pkl"])
    parser.add_argument("--out_json",   type=str,
                        default=os.path.join(p_cfg["results_dir"], "inference_results.json"))
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Load dataset ─────────────────────────────────────────────────────────
    if not os.path.exists(args.test_pkl):
        print(f"ERROR: test PKL not found at {args.test_pkl}")
        sys.exit(1)

    test_ds  = PCamBagDataset(args.test_pkl, num_groups=m_cfg["num_pseudobags"])
    feat_dim = test_ds.get_feature_dim()
    print(f"Test bags   : {len(test_ds)}  |  Feature dim: {feat_dim}")

    # ── Build model ───────────────────────────────────────────────────────────
    model = SemanticDTFDMIL(
        in_chn  = feat_dim,
        m_dim   = m_cfg["hidden_dim"],
        num_cls = m_cfg["num_classes"],
        att_dim = m_cfg["attention_dim"],
        dropout = m_cfg["dropout"],
    ).to(device)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found at {args.checkpoint}")
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.classifier.load_state_dict(ckpt["classifier"])
    model.attention.load_state_dict(ckpt["attention"])
    model.dim_reduction.load_state_dict(ckpt["dim_reduction"])
    model.att_classifier.load_state_dict(ckpt["att_classifier"])
    print(f"Checkpoint  : {args.checkpoint}  (epoch={ckpt.get('epoch','?')}, "
          f"auc={ckpt.get('auc','?')})")

    # ── Inference ─────────────────────────────────────────────────────────────
    model.eval()
    per_bag, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for i in range(len(test_ds)):
            pseudo_bags, label, name = test_ds[i]
            pseudo_bags = [pb.to(device) for pb in pseudo_bags]

            t2_logits, _, _ = model(pseudo_bags)
            prob = torch.softmax(t2_logits, dim=1)[:, 1].mean().item()

            pred   = "Tumor" if prob >= 0.5 else "Normal"
            actual = "Tumor" if label == 1 else "Normal"

            all_probs.append(prob)
            all_labels.append(label)
            per_bag.append({
                "slide":       name,
                "true_label":  label,
                "true_class":  actual,
                "tumor_prob":  round(prob * 100, 2),
                "normal_prob": round((1 - prob) * 100, 2),
                "prediction":  pred,
                "correct":     (pred == actual),
            })

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(all_labels, all_probs)
    output  = {
        "model":       "Proposed_SAM_PLIP",
        "checkpoint":  args.checkpoint,
        "test_slides": len(test_ds),
        **metrics,
        "results":     per_bag,
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(output, f, indent=2)

    print_metrics(metrics, "Inference Results")
    print(f"\n  Saved → {args.out_json}")
    print("\n  Per-bag predictions:")
    for r in per_bag:
        status = "OK" if r["correct"] else "XX"
        print(f"  [{status}] {r['slide']:<25} | True:{r['true_class']:6s} "
              f"| Pred:{r['prediction']:6s} | P(tumor)={r['tumor_prob']:.1f}%")


if __name__ == "__main__":
    main()
