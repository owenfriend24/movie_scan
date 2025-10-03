#!/usr/bin/env python
"""
Train on adults, test on children (generalization; Run=4 by default).

Features:
  - For each subject: all item pairs (C(12,2)=66)
  - X_pair = |β_i - β_j| over a provided binary mask (e.g., group top-% Wilcoxon z mask)
  - y_pair = 1 if same triplet else 0
  - Optional per-voxel item-wise z-scoring across the 12 items before forming pairs

Train:
  - Pool all ADULT pairs (age_group == 'adult') for the specified run
  - Classifier: L2 Logistic Regression (class_weight='balanced') by default; Linear SVM optional

Test (Children only):
  - Evaluate per child: accuracy, ROC-AUC, precision/recall/F1, confusion matrix
  - Also compute “adult-likeness” scalars:
      * mean score on within pairs
      * mean score on across pairs
      * delta = mean(within) - mean(across)
  - Save per-child metrics CSV + summary
  - If columns 'age' (for children) and/or 'memory_score' or 'memory' exist, compute correlations with metrics

Usage:
  python adult_to_child_generalization.py meta.csv group_mask.nii.gz /scratch/out_adult2child \
      --run 4 --clf logreg --C 1.0 --zscore_items
"""

import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from scipy.stats import pearsonr, spearmanr

# ---------------------- robust correlation helpers ---------------------- #
def _pearson_r_p(x, y):
    """Return (r, p) robustly across SciPy versions."""
    res = pearsonr(x, y)
    if hasattr(res, "statistic"):
        return float(res.statistic), float(res.pvalue)
    r, p = res
    return float(r), float(p)

def _spearman_rho_p(x, y):
    """Return (rho, p) robustly across SciPy versions."""
    res = spearmanr(x, y)
    if hasattr(res, "statistic"):
        return float(res.statistic), float(res.pvalue)
    rho, p = res
    return float(rho), float(p)

# ---------------------- CLI ---------------------- #
def get_args():
    p = argparse.ArgumentParser(description="Train on adults, evaluate on children (Run=4).")
    p.add_argument("meta_csv", help="CSV with: subject,age_group,run,item_id,triplet_id,beta_path (+ age/memory if available)")
    p.add_argument("mask_nii", help="Binary NIfTI mask for features (e.g., group top-% Wilcoxon z mask)")
    p.add_argument("outdir", help="Output directory")
    p.add_argument("--run", type=int, default=4, help="Run to use (default: 4)")
    p.add_argument("--clf", choices=["logreg","svm"], default="logreg", help="Classifier (default: logreg)")
    p.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength (default: 1.0)")
    p.add_argument("--zscore_items", action="store_true",
                   help="Z-score the 12 item betas per voxel before pairwise diffs.")
    p.add_argument("--random_state", type=int, default=13, help="Random seed")
    return p.parse_args()

# ---------------------- helpers ---------------------- #
def load_mask(mask_path):
    mimg = nib.load(mask_path)
    mdat = mimg.get_fdata()
    mask = mdat > 0.5
    return mimg, mask

def flat_indices(mask_bool):
    return np.where(mask_bool.reshape(-1))[0]

def load_items_for_subject(rows_df, flat_mask_idx):
    mats = []
    for _, r in rows_df.iterrows():
        vol = nib.load(r["beta_path"]).get_fdata().reshape(-1)
        mats.append(vol[flat_mask_idx])
    return np.vstack(mats).astype(np.float32)  # (12, Vmask)

def pairs_features_and_labels(X_items, triplet_ids, zscore_items=False):
    """
    Build all item pairs for a subject.
    Feature = |β_i - β_j| across voxels (length Vmask).
    Label   = 1 if same triplet else 0.
    Returns X_pairs (66, Vmask), y_pairs (66,)
    """
    X = X_items.copy()
    if zscore_items:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - mu) / sd

    idx_pairs = list(combinations(range(X.shape[0]), 2))  # 66 pairs
    X_pairs, y_pairs = [], []
    trip = np.asarray(triplet_ids)
    for i, j in idx_pairs:
        feat = np.abs(X[i, :] - X[j, :])  # (Vmask,)
        lab = 1 if trip[i] == trip[j] else 0
        X_pairs.append(feat); y_pairs.append(lab)
    return np.vstack(X_pairs), np.array(y_pairs, dtype=int)

def make_classifier(kind, C, random_state):
    if kind == "logreg":
        clf = LogisticRegression(
            penalty="l2", C=C, solver="lbfgs", max_iter=2000,
            class_weight="balanced", n_jobs=None, random_state=random_state
        )
        pipe = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                         ("clf", clf)])
        return pipe, "prob"
    else:
        clf = LinearSVC(C=C, class_weight="balanced", max_iter=5000, random_state=random_state)
        pipe = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                         ("clf", clf)])
        return pipe, "dec"

def safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan

# ---------------------- main ---------------------- #
def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load meta; split adults/children for the chosen run
    meta = pd.read_csv(args.meta_csv)
    req = {"subject","age_group","run","item_id","triplet_id","beta_path"}
    missing = req - set(meta.columns)
    if missing:
        raise ValueError(f"meta_csv missing required columns: {sorted(missing)}")

    adults = meta[(meta["age_group"] == "adult") & (meta["run"] == args.run)].copy()
    kids   = meta[(meta["age_group"] != "adult") & (meta["run"] == args.run)].copy()
    if adults.empty:
        raise ValueError("No adult rows for the requested run.")
    if kids.empty:
        print("[WARN] No child rows for the requested run; script will still train but not evaluate.")

    # Mask
    mask_img, mask_bool = load_mask(args.mask_nii)
    flat_idx = flat_indices(mask_bool)
    Vmask = flat_idx.size
    print(f"[INFO] Using mask {os.path.basename(args.mask_nii)} with {Vmask} voxels")

    # Build adult training set (pooled)
    subjects_adult = sorted(adults["subject"].unique())
    X_train_list, y_train_list = [], []
    for sid in subjects_adult:
        rows = adults[adults["subject"] == sid].sort_values("item_id")
        Xi = load_items_for_subject(rows, flat_idx)  # (12, Vmask)
        trip = rows["triplet_id"].to_numpy()
        X_pairs, y_pairs = pairs_features_and_labels(Xi, trip, zscore_items=args.zscore_items)
        X_train_list.append(X_pairs); y_train_list.append(y_pairs)
    X_train = np.vstack(X_train_list)
    y_train = np.hstack(y_train_list)
    print(f"[TRAIN] adults={len(subjects_adult)}  pairs={X_train.shape[0]}  features={X_train.shape[1]}")

    # Fit model
    clf, score_mode = make_classifier(args.clf, args.C, args.random_state)
    clf.fit(X_train, y_train)

    # Evaluate on children
    child_metrics = []
    preds_dir = os.path.join(args.outdir, "child_predictions")
    os.makedirs(preds_dir, exist_ok=True)

    subjects_child = sorted(kids["subject"].unique())
    for sid in subjects_child:
        rows = kids[kids["subject"] == sid].sort_values("item_id")
        if rows.empty:
            continue
        Xi = load_items_for_subject(rows, flat_idx)
        trip = rows["triplet_id"].to_numpy()
        X_pairs, y_pairs = pairs_features_and_labels(Xi, trip, zscore_items=args.zscore_items)
        if score_mode == "prob":
            y_score = clf.predict_proba(X_pairs)[:, 1]
            y_pred  = (y_score >= 0.5).astype(int)
        else:
            y_score = clf.decision_function(X_pairs)
            y_pred  = (y_score >= 0).astype(int)

        acc = accuracy_score(y_pairs, y_pred)
        auc = safe_auc(y_pairs, y_score)
        prec, rec, f1, _ = precision_recall_fscore_support(y_pairs, y_pred, average="binary", zero_division=0)
        cm = confusion_matrix(y_pairs, y_pred, labels=[0,1])

        # “adult-likeness” scalars
        within_scores  = y_score[y_pairs == 1]
        across_scores  = y_score[y_pairs == 0]
        mean_within    = float(np.mean(within_scores)) if within_scores.size else np.nan
        mean_across    = float(np.mean(across_scores)) if across_scores.size else np.nan
        score_delta    = mean_within - mean_across

        # optional fields from meta
        age = meta.loc[(meta["subject"] == sid) & (meta["run"] == args.run), "age"]
        age_val = float(age.iloc[0]) if not age.empty else np.nan
        mem = np.nan
        for col in ["memory_score", "memory"]:
            if col in meta.columns:
                tmp = meta.loc[(meta["subject"] == sid) & (meta["run"] == args.run), col]
                if not tmp.empty:
                    mem = float(tmp.iloc[0]); break

        child_metrics.append({
            "subject": sid,
            "n_pairs": int(y_pairs.size),
            "accuracy": float(acc),
            "auc": float(auc),
            "precision_within": float(prec),
            "recall_within": float(rec),
            "f1_within": float(f1),
            "tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1]),
            "mean_score_within": mean_within,
            "mean_score_across": mean_across,
            "score_delta_within_minus_across": float(score_delta),
            "age": age_val,
            "memory": mem,
        })

        # save per-child predictions for deep dives
        np.save(os.path.join(preds_dir, f"{sid}_y_true.npy"), y_pairs)
        np.save(os.path.join(preds_dir, f"{sid}_y_pred.npy"), y_pred)
        np.save(os.path.join(preds_dir, f"{sid}_y_score.npy"), y_score)

        print(f"[CHILD] {sid}: acc={acc:.3f} auc={auc:.3f} within_rec={rec:.3f} Δscore={score_delta:.3f}")

    # Save per-child metrics
    metrics_df = pd.DataFrame(child_metrics)
    metrics_csv = os.path.join(args.outdir, "child_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    # Overall child summary
    if not metrics_df.empty:
        acc_mean = metrics_df["accuracy"].mean()
        auc_mean = metrics_df["auc"].mean()
        rec_mean = metrics_df["recall_within"].mean()
        print("\n===== CHILD SUMMARY =====")
        print(f"N children: {metrics_df.shape[0]}")
        print(f"Mean accuracy: {acc_mean:.4f}")
        print(f"Mean AUC:      {auc_mean:.4f}")
        print(f"Mean recall(within): {rec_mean:.4f}")

        # Correlations (kids only) with age / memory if present
        corr_txt = []
        for var in ["age", "memory", "memory_score"]:
            if var in metrics_df.columns and metrics_df[var].notna().sum() >= 3:
                for met in ["accuracy","auc","recall_within","score_delta_within_minus_across"]:
                    x = metrics_df[var].astype(float)
                    y = metrics_df[met].astype(float)
                    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
                    if mask.sum() >= 3 and (x[mask].nunique() > 1) and (y[mask].nunique() > 1):
                        r, rp = _pearson_r_p(x[mask].values, y[mask].values)
                        rho, sp = _spearman_rho_p(x[mask].values, y[mask].values)
                        line = (f"{met} ~ {var}: Pearson r={r:.3f}, p={rp:.3g} | "
                                f"Spearman ρ={rho:.3f}, p={sp:.3g}")
                        corr_txt.append(line)
                        print(line)

        # Write summary file
        with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
            f.write("Adult→Child generalization\n")
            f.write(f"Run: {args.run}\n")
            f.write(f"Classifier: {args.clf}  C={args.C}  class_weight=balanced  zscore_items={bool(args.zscore_items)}\n")
            f.write(f"Mask: {args.mask_nii}  Voxels: {Vmask}\n")
            f.write(f"N adults (train): {len(subjects_adult)}\n")
            f.write(f"N children (test): {metrics_df.shape[0]}\n")
            f.write(f"Mean accuracy: {acc_mean:.6f}\n")
            f.write(f"Mean AUC:      {auc_mean:.6f}\n")
            f.write(f"Mean recall(within): {rec_mean:.6f}\n")
            if corr_txt:
                f.write("\nCorrelations (children only):\n")
                for line in corr_txt:
                    f.write(line + "\n")
    else:
        print("[INFO] No child metrics to summarize (no child rows for run).")

if __name__ == "__main__":
    main()
