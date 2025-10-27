#!/usr/bin/env python3
"""
plot_child_roc.py

Usage:
  python plot_child_roc.py /path/to/<child>_y_true.npy /path/to/<child>_y_score.npy \
      --out /path/to/figs/<child>_roc.pdf --label child012

Inputs:
  y_true  : 0/1 ground-truth labels (0=across, 1=within)
  y_score : continuous classifier scores/probabilities for class 1
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("y_true", help="Path to *_y_true.npy")
    ap.add_argument("y_score", help="Path to *_y_score.npy")
    ap.add_argument("--out", default="/scratch/09123/ofriend/movie_scan/example_roc.pdf", help="Output PDF path")
    ap.add_argument("--label", default="", help="Optional subject label for title")
    args = ap.parse_args()

    y_true  = np.load(args.y_true)
    y_score = np.load(args.y_score)

    # Basic sanity checks
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have same length.")
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true  = y_true[mask].astype(int)
    y_score = y_score[mask].astype(float)

    # ROC + AUC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    # Plot
    plt.figure(figsize=(4.5, 4.5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}", color="#00a9b7")   # teal
    plt.plot([0, 1], [0, 1], "--", lw=1, color="#a6cd57")                 # green
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    title = "ROC"
    if args.label:
        title = f"{args.label} · {title}"
    plt.title(title)
    plt.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig(args.out)
    plt.close()
    print(f"Saved ROC to {args.out} (AUC={auc:.3f})")

if __name__ == "__main__":
    main()
