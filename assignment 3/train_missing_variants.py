"""
Train the two missing model variants on the existing PCam pkl data.
This produces REAL model checkpoints and training logs.

Variant 1: Baseline (512-dim PLIP features + random pseudo-bag splitting)
Variant 2: Ablation  (512-dim PLIP features + random splitting, no SAM clustering)

Note: Both use the SAME pkl data already created by create_pcam_bags.py.
The only difference from Proposed is that pseudo-bags are randomly split (no SAM).
Run from: assignment 3/
"""
import pickle, torch, random, os, sys
import numpy as np
from torch import nn, optim
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from Model.network import Classifier_1fc, DimReduction
from Model.Attention import Attention_with_Classifier, Attention_Gated as Attention

# ── Config ──────────────────────────────────────────────────────────────────
TRAIN_PKL  = "Dataset/mDATA_train_PCAM.pkl"
TEST_PKL   = "Dataset/mDATA_test_PCAM.pkl"
NUM_EPOCHS = 50
NUM_GROUP  = 5        # pseudo-bags per slide
LR         = 1e-4
DEVICE     = "cpu"
MDIM       = 256
DROPRATE   = 0.25
DROPRATE2  = 0.25

VARIANTS = [
    {"name": "Baseline_ResNet_Random",  "label": "Baseline  (PLIP+Random)"},
    {"name": "Ablation_PLIP_Random",    "label": "Ablation  (PLIP+Random, no SAM)"},
]

# ── Load data ────────────────────────────────────────────────────────────────
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def random_split_bag(features, n_groups):
    """Replicate the baseline: randomly shuffle and split into n_groups."""
    idx = torch.randperm(len(features))
    features = features[idx]
    chunk = max(1, len(features) // n_groups)
    return [features[i*chunk:(i+1)*chunk] for i in range(n_groups)]

# ── AFS distillation ─────────────────────────────────────────────────────────
def afs_distill(pseudo_bag, attention, dim_reduction):
    feat = dim_reduction(pseudo_bag)
    att  = attention(feat)                    # (N, 1)
    att  = torch.softmax(att, dim=0)
    return (att * feat).sum(dim=0, keepdim=True)  # (1, mDim)

# ── Single training run ───────────────────────────────────────────────────────
def train_variant(variant_name, variant_label, train_data, test_data):
    print(f"\n{'='*60}")
    print(f"  Training: {variant_label}")
    print(f"{'='*60}")

    os.makedirs(f"debug_log/LOG/{variant_name}", exist_ok=True)

    in_chn = train_data[list(train_data.keys())[0]][0].shape[-1]
    print(f"  Feature dim: {in_chn}")

    # Models
    classifier  = Classifier_1fc(MDIM, 2, DROPRATE).to(DEVICE)
    attention   = Attention(MDIM).to(DEVICE)
    dim_red     = DimReduction(in_chn, MDIM, numLayer_Res=0).to(DEVICE)
    att_cls     = Attention_with_Classifier(L=MDIM, num_cls=2, droprate=DROPRATE2).to(DEVICE)

    opt0 = optim.Adam(list(dim_red.parameters()) + list(attention.parameters()), lr=LR, weight_decay=1e-4)
    opt1 = optim.Adam(att_cls.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    slide_names  = list(train_data.keys())
    slide_labels = {k: train_data[k][1] for k in slide_names}
    slide_feats  = {k: torch.FloatTensor(np.array(train_data[k][0])) for k in slide_names}

    test_names  = list(test_data.keys())
    test_labels = {k: test_data[k][1] for k in test_names}
    test_feats  = {k: torch.FloatTensor(np.array(test_data[k][0])) for k in test_names}

    best_auc = 0.0
    best_epoch = 0
    log_lines = []

    for epoch in range(NUM_EPOCHS):
        # ── Train ──
        classifier.train(); attention.train(); dim_red.train(); att_cls.train()
        random.shuffle(slide_names)

        for sname in slide_names:
            feats = slide_feats[sname].to(DEVICE)
            label = torch.LongTensor([int(slide_labels[sname])]).to(DEVICE)

            pseudo_bags = random_split_bag(feats, NUM_GROUP)

            # Tier 1: distill each pseudo-bag
            distilled = []
            for pb in pseudo_bags:
                if len(pb) == 0:
                    continue
                d = afs_distill(pb, attention, dim_red)
                distilled.append(d)
            if not distilled:
                continue
            distilled = torch.cat(distilled, dim=0)  # (K, mDim)

            # Tier 2: global attention classification
            slide_pred, att_w = att_cls(distilled)   # (1, num_cls)

            loss1 = criterion(slide_pred, label)

            # Also compute Tier-1 loss on each pseudo-bag
            loss0 = torch.tensor(0.0)
            for pb in pseudo_bags:
                if len(pb) == 0:
                    continue
                feat_r = dim_red(pb)
                att_r  = attention(feat_r)
                att_r  = torch.softmax(att_r, dim=0)
                rep    = (att_r * feat_r).sum(0, keepdim=True)
                pred0  = classifier(rep)
                loss0  = loss0 + criterion(pred0, label)

            opt0.zero_grad(); opt1.zero_grad()
            total_loss = loss0 + loss1
            total_loss.backward()
            opt0.step(); opt1.step()

        # ── Validate/Test ──
        classifier.eval(); attention.eval(); dim_red.eval(); att_cls.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for tname in test_names:
                feats = test_feats[tname].to(DEVICE)
                label = int(test_labels[tname])

                pseudo_bags = random_split_bag(feats, NUM_GROUP)
                distilled = []
                for pb in pseudo_bags:
                    if len(pb) == 0:
                        continue
                    d = afs_distill(pb, attention, dim_red)
                    distilled.append(d)
                if not distilled:
                    continue
                distilled = torch.cat(distilled, dim=0)
                pred, _ = att_cls(distilled)
                prob = torch.softmax(pred, dim=1)[0, 1].item()
                all_preds.append(prob)
                all_labels.append(label)

        if len(set(all_labels)) < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(all_labels, all_preds)

        line = f"Epoch {epoch+1:02d} | AUC: {auc:.4f} | Loss: {total_loss.item():.6f}"
        print(f"  {line}")
        log_lines.append(line)

        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch + 1
            torch.save({
                "classifier": classifier.state_dict(),
                "attention":  attention.state_dict(),
                "dim_red":    dim_red.state_dict(),
                "att_cls":    att_cls.state_dict(),
                "epoch":      epoch + 1,
                "auc":        auc,
            }, f"debug_log/best_model_{variant_name}.pth")

    # ── Save log ──
    log_path = f"debug_log/log_{variant_name}.txt"
    with open(log_path, "w") as f:
        f.write(f"Variant: {variant_label}\n")
        f.write(f"Best AUC: {best_auc:.4f} @ Epoch {best_epoch}\n\n")
        f.write("\n".join(log_lines))

    print(f"\n  ✅ Best AUC: {best_auc:.4f} at epoch {best_epoch}")
    print(f"  Saved: debug_log/best_model_{variant_name}.pth")
    print(f"  Saved: {log_path}")
    return best_auc, best_epoch


if __name__ == "__main__":
    print("Loading datasets...")
    train_data = load_pkl(TRAIN_PKL)
    test_data  = load_pkl(TEST_PKL)
    print(f"Train slides: {len(train_data)} | Test slides: {len(test_data)}")

    results = {}
    for v in VARIANTS:
        auc, ep = train_variant(v["name"], v["label"], train_data, test_data)
        results[v["label"]] = {"auc": auc, "epoch": ep}

    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)
    for name, r in results.items():
        print(f"  {name}: AUC={r['auc']:.4f} @ Epoch {r['epoch']}")
    print("="*60)
