#!/usr/bin/env python
"""
Combine LOSO-adult metrics with child generalization metrics into a single CSV.

- Recomputes per-adult (LOSO) metrics using the same feature pipeline:
    * 12 items → all pairs (66)
    * feature = |beta_i - beta_j| within a provided binary mask (e.g., group top-% Wilcoxon-z mask)
    * label   = 1 within-triplet, 0 across
    * classifier: L2 logistic (balanced) by default; linear SVM optional
    * metrics per adult: accuracy, AUC, precision/recall/F1 (within=1), confusion matrix,
                         mean score within/across, and their delta
- Loads the child metrics CSV you already created (from adult_to_child_generalization.py)
- Harmonizes columns and concatenates into one CSV.

Usage:
  python combine_adults_children_metrics.py \
      meta.csv \
      group_adults_run-4_top5.00pct_GM_posz_wilcoxon_z_triplet_mask.nii.gz \
      child_metrics.csv \
      /scratch/out/combined \
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
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_recall_fscore_support,
                             confusion_matrix)

# ---------------------- CLI ---------------------- #
def get_args():
    p = argparse.ArgumentParser(description="Build combined CSV: LOSO adults + child generalization.")
    p.add_argument("meta_csv", help="CSV with: subject,age_group,run,item_id,triplet_id,beta_path (+ age optional)")
    p.add_argument("mask_nii", help="Binary NIfTI mask for features (e.g., group top-% Wilcoxon-z mask)")
    p.add_argument("child_metrics_csv", help="child_metrics.csv produced by adult_to_child_generalization.py")
    p.add_argument("outdir", help="Output directory")
    p.add_argument("--run", type=int, default=4, help="Run to use (default: 4)")
    p.add_argument("--clf", choices=["logreg","svm"], default="logreg", help="Classifier (default: logreg)")
    p.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength (default: 1.0)")
    p.add_argument("--zscore_items", action="store_true",
                   help="Z-score the 12 item betas per voxel before pairwise diffs (optional).")
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

    # Load meta & mask
    meta = pd.read_csv(args.meta_csv)
    req = {"subject","age_group","run","item_id","triplet_id","beta_path"}
    missing = req - set(meta.columns)
    if missing:
        raise ValueError(f"meta_csv missing required columns: {sorted(missing)}")

    mask_img, mask_bool = load_mask(args.mask_nii)
    flat_idx = flat_indices(mask_bool)
    Vmask = flat_idx.size
    print(f"[INFO] Using mask {os.path.basename(args.mask_nii)} with {Vmask} voxels")

    # Adults for LOSO recompute
    adults = meta[(meta["age_group"] == "adult") & (meta["run"] == args.run)].copy()
    if adults.empty:
        raise ValueError("No adult rows for the requested run in meta_csv.")
    adult_subjects = sorted(adults["subject"].unique())

    # Build per-adult item matrices (avoid reloading many times)
    subj_items = {}
    for sid in adult_subjects:
        rows = adults[adults["subject"] == sid].sort_values("item_id")
        Xi = load_items_for_subject(rows, flat_idx)  # (12, Vmask)
        trip = rows["triplet_id"].to_numpy()
        subj_items[sid] = (Xi, trip)

    clf, score_mode = make_classifier(args.clf, args.C, args.random_state)

    # LOSO: compute per-adult metrics (out-of-sample)
    adult_rows = []
    for test_sid in adult_subjects:
        # training pool
        X_train, y_train = [], []
        for sid in adult_subjects:
            if sid == test_sid:
                continue
            Xi, trip = subj_items[sid]
            Xt, yt = pairs_features_and_labels(Xi, trip, zscore_items=args.zscore_items)
            X_train.append(Xt); y_train.append(yt)
        X_train = np.vstack(X_train); y_train = np.hstack(y_train)

        # fit
        clf.fit(X_train, y_train)

        # test on held-out adult
        Xi, trip = subj_items[test_sid]
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

        within_scores  = y_score[y_pairs == 1]
        across_scores  = y_score[y_pairs == 0]
        mean_within    = float(np.mean(within_scores)) if within_scores.size else np.nan
        mean_across    = float(np.mean(across_scores)) if across_scores.size else np.nan
        score_delta    = mean_within - mean_across

        # optional age if present
        age_val = np.nan
        if "age" in meta.columns:
            a = meta.loc[(meta["subject"] == test_sid) & (meta["run"] == args.run), "age"]
            if not a.empty:
                age_val = float(a.iloc[0])

        adult_rows.append({
            "subject": test_sid,
            "age_group": "adult",
            "age": age_val,
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
        })
        print(f"[ADULT LOSO] {test_sid}: acc={acc:.3f} auc={auc:.3f} within_rec={rec:.3f} Δscore={score_delta:.3f}")

    adults_df = pd.DataFrame(adult_rows)

    # Load child metrics CSV and keep consistent columns
    kids_df = pd.read_csv(args.child_metrics_csv)
    # If age_group missing, set to 'child' for safety (or whatever is in your CSV)
    if "age_group" not in kids_df.columns:
        kids_df["age_group"] = "child"

    # Ensure all expected columns exist (fill missing with NaN)
    expected_cols = [
        "subject","age_group","age","n_pairs","accuracy","auc",
        "precision_within","recall_within","f1_within",
        "tn","fp","fn","tp",
        "mean_score_within","mean_score_across","score_delta_within_minus_across"
    ]
    for df in (adults_df, kids_df):
        for c in expected_cols:
            if c not in df.columns:
                df[c] = np.nan
        df[:] = df[expected_cols]  # reorder

    combined = pd.concat([kids_df[expected_cols], adults_df[expected_cols]], ignore_index=True)

    out_csv = os.path.join(args.outdir, "combined_metrics.csv")
    combined.to_csv(out_csv, index=False)
    print(f"\n[OK] Wrote combined CSV with {combined.shape[0]} rows → {out_csv}")

if __name__ == "__main__":
    main()
