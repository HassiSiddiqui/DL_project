"""
Run inference using the saved best_model.pth checkpoint on the test PKL.
Saves real predictions to debug_log/inference_results.json for the frontend.
Run from: assignment 3/
"""
import pickle, torch, json, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from Model.network import Classifier_1fc, DimReduction
from Model.Attention import Attention_with_Classifier, Attention_Gated as Attention

# ── Config ──────────────────────────────────────────────────────────────────
CHECKPOINT = "debug_log/best_model.pth"
TEST_PKL   = "Dataset/mDATA_test_PCAM.pkl"
OUT_JSON   = "debug_log/inference_results.json"
NUM_GROUP  = 5
MDIM       = 256
DEVICE     = "cpu"

def load_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def extract_bag(data, slide_name):
    """
    PKL structure: {slide_name: [patch_dict, ...]}
    Each patch_dict has keys: 'feature' (tensor, 512-dim), 'label' (int), 'cluster_id', 'name'
    Bag label = label of any patch (all patches in a bag share the same label)
    """
    patches = data[slide_name]   # list of 20 dicts
    feats = torch.stack([p['feature'] for p in patches])  # (20, 512)
    label = int(patches[0]['label'])
    return feats, label

def get_feat_dim(data):
    key = list(data.keys())[0]
    patches = data[key]
    return patches[0]['feature'].shape[-1]

def random_split(feats, n):
    idx = torch.randperm(len(feats))
    feats = feats[idx]
    chunk = max(1, len(feats) // n)
    return [feats[i*chunk:(i+1)*chunk] for i in range(n)]

if __name__ == "__main__":
    print("Loading checkpoint and test data...")
    test_data = load_data(TEST_PKL)
    in_chn = get_feat_dim(test_data)
    print(f"  Feature dim: {in_chn} | Test slides: {len(test_data)}")

    # ── Build model ──────────────────────────────────────────────────────────
    classifier  = Classifier_1fc(MDIM, 2, 0).to(DEVICE)
    attention   = Attention(MDIM).to(DEVICE)
    dim_red     = DimReduction(in_chn, MDIM, numLayer_Res=0).to(DEVICE)
    att_cls     = Attention_with_Classifier(L=MDIM, num_cls=2, droprate=0).to(DEVICE)

    # ── Load weights ─────────────────────────────────────────────────────────
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    # Keys saved by Main_DTFD_MIL.py: classifier, dim_reduction, attention, att_classifier
    if "classifier" in ckpt:
        classifier.load_state_dict(ckpt["classifier"])
        attention.load_state_dict(ckpt["attention"])
        dim_red.load_state_dict(ckpt["dim_reduction"])
        att_cls.load_state_dict(ckpt["att_classifier"])
        saved_epoch = ckpt.get("epoch", "?")
        saved_auc   = ckpt.get("auc", "?")
    else:
        print("  Warning: unrecognised checkpoint format.")
        saved_epoch = "N/A"
        saved_auc   = "N/A"

    print(f"  Loaded checkpoint (epoch={saved_epoch}, auc={saved_auc})")

    # ── Run inference ─────────────────────────────────────────────────────────
    classifier.eval(); attention.eval(); dim_red.eval(); att_cls.eval()

    results = []
    all_probs, all_labels = [], []

    with torch.no_grad():
        for slide_name in test_data.keys():
            feats, label = extract_bag(test_data, slide_name)

            pseudo_bags = random_split(feats, NUM_GROUP)
            distilled = []
            for pb in pseudo_bags:
                if len(pb) == 0:
                    continue
                feat_r = dim_red(pb)           # (N, MDIM)
                att_w  = attention(feat_r)     # (1, N)  — Attention_Gated returns K x N
                dist   = torch.mm(att_w, feat_r)  # (1, MDIM)
                distilled.append(dist)

            if not distilled:
                continue

            distilled = torch.cat(distilled, dim=0)
            pred = att_cls(distilled)           # (K, 2)
            prob_tumor = torch.softmax(pred, dim=1)[:, 1].mean().item()

            all_probs.append(prob_tumor)
            all_labels.append(label)

            results.append({
                "slide": slide_name,
                "true_label": label,
                "true_class": "Tumor" if label == 1 else "Normal",
                "tumor_prob": round(prob_tumor * 100, 2),
                "normal_prob": round((1 - prob_tumor) * 100, 2),
                "prediction": "Tumor" if prob_tumor >= 0.5 else "Normal",
                "correct": (prob_tumor >= 0.5) == (label == 1)
            })

    # ── Summary metrics ───────────────────────────────────────────────────────
    if len(set(all_labels)) >= 2:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = 0.0

    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / len(results) if results else 0

    summary = {
        "model": "Proposed_SAM_PLIP",
        "checkpoint": CHECKPOINT,
        "test_slides": len(results),
        "auc": round(auc, 4),
        "accuracy": round(accuracy, 4),
        "results": results
    }

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  AUC:      {auc:.4f}")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{len(results)})")
    print(f"  Saved:    {OUT_JSON}")
    print(f"{'='*50}")
    for r in results:
        tick = "OK" if r["correct"] else "XX"
        print(f"  [{tick}] {r['slide'][:30]:30s} | True: {r['true_class']:6s} | Pred: {r['prediction']:6s} | P(tumor)={r['tumor_prob']:.1f}%")
