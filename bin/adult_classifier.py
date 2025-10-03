#!/usr/bin/env python
"""
Adults-only LOSO classifier for within(1) vs across(0) triplet pairs (Run=4 by default),
using class-weighted (balanced) linear models.

Features per subject:
  - 12 item betas → all pairs (C(12,2)=66)
  - X_pair = |β_i - β_j| per voxel within mask
  - y_pair = 1 if same triplet else 0

Model & CV:
  - LOSO across adult subjects
  - Default: L2 Logistic Regression with class_weight='balanced'
  - Optional: Linear SVM (also class_weight='balanced')
  - Reports accuracy, ROC-AUC, precision/recall/F1, confusion matrix, per-subject accuracies

Inputs:
  meta_csv: subject, age_group, run, item_id, triplet_id, beta_path
  mask_nii: binary NIfTI (e.g., your group top-% Wilcoxon-z mask)
  outdir:   output directory
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
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

# ---------------------- args ---------------------- #
def get_args():
    p = argparse.ArgumentParser(description="Adults LOSO within-vs-across with class-weighted linear models.")
    p.add_argument("meta_csv", help="CSV: subject,age_group,run,item_id,triplet_id,beta_path")
    p.add_argument("mask_nii", help="Binary NIfTI mask used for features")
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
            penalty="l2",
            C=C,
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",   # <--- balanced
            n_jobs=None,
            random_state=random_state,
        )
        pipe = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                         ("clf", clf)])
        return pipe, "prob"
    else:
        clf = LinearSVC(C=C, class_weight="balanced", max_iter=5000, random_state=random_state)  # <--- balanced
        pipe = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                         ("clf", clf)])
        return pipe, "dec"

# ---------------------- main ---------------------- #
def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load meta and filter adults + run
    meta = pd.read_csv(args.meta_csv)
    required = {"subject","age_group","run","item_id","triplet_id","beta_path"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"meta_csv missing columns: {sorted(missing)}")

    adults = meta[(meta["age_group"] == "adult") & (meta["run"] == args.run)].copy()
    if adults.empty:
        raise ValueError("No adult rows for the requested run.")

    # Mask
    mask_img, mask_bool = load_mask(args.mask_nii)
    flat_idx = flat_indices(mask_bool)
    Vmask = flat_idx.size
    print(f"[INFO] Using mask {os.path.basename(args.mask_nii)} with {Vmask} voxels")

    # Build per-subject pairwise datasets
    subjects = sorted(adults["subject"].unique())
    per_subj = {}
    for sid in subjects:
        rows = adults[adults["subject"] == sid].sort_values("item_id")
        if rows.empty:
            continue
        X_items = load_items_for_subject(rows, flat_idx)  # (12, Vmask)
        trip = rows["triplet_id"].to_numpy()
        X_pairs, y_pairs = pairs_features_and_labels(X_items, trip, zscore_items=args.zscore_items)
        per_subj[sid] = (X_pairs, y_pairs)

    # LOSO CV with class-weighted models
    clf, score_mode = make_classifier(args.clf, args.C, args.random_state)
    all_y_true, all_y_score, all_y_pred = [], [], []
    per_subject_acc, per_subject_counts = [], []

    for test_sid in subjects:
        # training data
        X_train, y_train = [], []
        for sid in subjects:
            if sid == test_sid:
                continue
            Xt, yt = per_subj[sid]
            X_train.append(Xt); y_train.append(yt)
        X_train = np.vstack(X_train)
        y_train = np.hstack(y_train)

        # fit
        clf.fit(X_train, y_train)

        # test
        X_test, y_test = per_subj[test_sid]
        if score_mode == "prob":
            y_score = clf.predict_proba(X_test)[:, 1]
            y_pred  = (y_score >= 0.5).astype(int)
        else:
            y_score = clf.decision_function(X_test)
            y_pred  = (y_score >= 0).astype(int)

        all_y_true.append(y_test)
        all_y_score.append(y_score)
        all_y_pred.append(y_pred)

        acc = accuracy_score(y_test, y_pred)
        per_subject_acc.append({"subject": test_sid, "accuracy": acc, "n_test": y_test.size})
        per_subject_counts.append({
            "subject": test_sid,
            "n_within": int((y_test==1).sum()),
            "n_across": int((y_test==0).sum())
        })
        print(f"[LOSO] test={test_sid}  acc={acc:.3f}  n={y_test.size}")

    # concatenate folds
    y_true = np.hstack(all_y_true)
    y_pred = np.hstack(all_y_pred)
    y_score = np.hstack(all_y_score)

    # metrics
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float("nan")
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])

    # print report
    print("\n===== OVERALL REPORT (Adults, LOSO, Run={}) =====".format(args.run))
    print(f"Classifier: {args.clf}  C={args.C}  zscore_items={args.zscore_items}  class_weight=balanced")
    print(f"Mask: {os.path.basename(args.mask_nii)}  Voxels: {Vmask}")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print(f"Precision (within=1): {prec:.4f}")
    print(f"Recall    (within=1): {rec:.4f}")
    print(f"F1        (within=1): {f1:.4f}")
    print("Confusion matrix [rows=true 0/1, cols=pred 0/1]:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["across(0)","within(1)"], zero_division=0))

    # save reports
    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame(per_subject_acc).to_csv(os.path.join(args.outdir, "per_subject_accuracy.csv"), index=False)
    pd.DataFrame(per_subject_counts).to_csv(os.path.join(args.outdir, "per_subject_test_counts.csv"), index=False)

    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write("Adults-only LOSO classification (within=1 vs across=0)\n")
        f.write(f"Run: {args.run}\n")
        f.write(f"Classifier: {args.clf}  C={args.C}  class_weight=balanced\n")
        f.write(f"zscore_items: {bool(args.zscore_items)}\n")
        f.write(f"Mask: {args.mask_nii}\n")
        f.write(f"Mask voxels: {Vmask}\n\n")
        f.write(f"Accuracy: {acc:.6f}\n")
        f.write(f"ROC_AUC:  {auc:.6f}\n")
        f.write(f"Precision(within=1): {prec:.6f}\n")
        f.write(f"Recall(within=1):    {rec:.6f}\n")
        f.write(f"F1(within=1):        {f1:.6f}\n")
        f.write("Confusion matrix [rows=true 0/1, cols=pred 0/1]:\n")
        f.write(np.array2string(cm, separator=', ') + "\n")

    np.save(os.path.join(args.outdir, "y_true.npy"), y_true)
    np.save(os.path.join(args.outdir, "y_pred.npy"), y_pred)
    np.save(os.path.join(args.outdir, "y_score.npy"), y_score)

if __name__ == "__main__":
    main()
