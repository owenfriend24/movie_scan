#!/usr/bin/env python3
"""
plot_child_roc_overlay.py

Overlay ROC curves for TWO subjects on one figure.

Usage:
  python plot_child_roc_overlay.py \
    /path/to/subjA_y_true.npy /path/to/subjA_y_score.npy \
    /path/to/subjB_y_true.npy /path/to/subjB_y_score.npy \
    --out /path/to/figs/overlay_roc.pdf \
    --label1 subjA --label2 subjB

Notes:
  - y_true: 0/1 labels (0=across, 1=within)
  - y_score: continuous scores/probabilities for class 1
  - Helvetica font, light-grey thicker random line, dark blue vs light blue curves
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# --- Colors ---
DARK_BLUE  = "#1f3a93"
LIGHT_BLUE = "#89c2ff"
RAND_LINE  = "#c7c7c7"   # light grey

def load_and_mask(y_true_path, y_score_path):
    y_true  = np.load(y_true_path)
    y_score = np.load(y_score_path)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(f"Mismatched lengths: {y_true_path} ({y_true.shape[0]}) vs {y_score_path} ({y_score.shape[0]})")
    m = np.isfinite(y_true) & np.isfinite(y_score)
    y_true  = y_true[m].astype(int)
    y_score = y_score[m].astype(float)
    return y_true, y_score

def roc_arrays(y_true, y_score):
    # handle cases where only one class present (AUC undefined)
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])
        auc = np.nan
    return fpr, tpr, auc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("y_true_1",  help="Subject 1: path to *_y_true.npy")
    ap.add_argument("y_score_1", help="Subject 1: path to *_y_score.npy")
    ap.add_argument("y_true_2",  help="Subject 2: path to *_y_true.npy")
    ap.add_argument("y_score_2", help="Subject 2: path to *_y_score.npy")
    ap.add_argument("--out",    default="overlay_roc.pdf", help="Output PDF path")
    ap.add_argument("--label1", default="Subject 1",       help="Legend label for subject 1")
    ap.add_argument("--label2", default="Subject 2",       help="Legend label for subject 2")
    args = ap.parse_args()

    # Helvetica font
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,   # embed text as TrueType
        "ps.fonttype": 42
    })

    # Load & compute ROC for both subjects
    y_true_1, y_score_1 = load_and_mask(args.y_true_1, args.y_score_1)
    y_true_2, y_score_2 = load_and_mask(args.y_true_2, args.y_score_2)

    fpr1, tpr1, auc1 = roc_arrays(y_true_1, y_score_1)
    fpr2, tpr2, auc2 = roc_arrays(y_true_2, y_score_2)

    # Plot
    plt.figure(figsize=(4.8, 4.8))
    # random (chance) line: dashed, light grey, thicker
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2.0, color=RAND_LINE)

    # subject curves
    plt.plot(fpr1, tpr1, lw=2.5, color=DARK_BLUE,  label=f"{args.label1}  AUC={auc1:.3f}" if np.isfinite(auc1) else f"{args.label1}  AUC=NA")
    plt.plot(fpr2, tpr2, lw=2.5, color=LIGHT_BLUE, label=f"{args.label2}  AUC={auc2:.3f}" if np.isfinite(auc2) else f"{args.label2}  AUC=NA")

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig(args.out)
    plt.close()
    print(f"Saved overlay ROC to {args.out}")

if __name__ == "__main__":
    main()
