#!/usr/bin/env python
"""
Combine LOSO-adult metrics (recomputed with item-level z-scoring) + child metrics into one CSV.
- Item-level z-scoring of the 12 items per voxel is ON by default (matches your RSA/classifier setup).
- Strict sanity checks (12 items; 4 triplets x 3 items).
- Deterministic item ordering (--sort_mode {item_id,beta_path,none}).
- Per-adult diagnostics: item order, within/across counts, pair-label SHA1, positive rate.
- Optional diff vs a reference adult CSV from your original adult script.

Usage:
  python combine_adults_children_metrics.py \
    meta.csv \
    group_adults_run-4_top5.00pct_GM_posz_wilcoxon_z_triplet_mask.nii.gz \
    child_metrics.csv \
    /scratch/out/combined \
    --run 4 --clf logreg --C 1.0 \
    --sort_mode beta_path
"""

import os, argparse, hashlib
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

# ---------------- CLI ---------------- #
def get_args():
    p = argparse.ArgumentParser(description="Combine kids + LOSO adults (z-scored items) into one CSV.")
    p.add_argument("meta_csv")
    p.add_argument("mask_nii")
    p.add_argument("child_metrics_csv")
    p.add_argument("outdir")
    p.add_argument("--run", type=int, default=4)
    p.add_argument("--clf", choices=["logreg","svm"], default="logreg")
    p.add_argument("--C", type=float, default=1.0)
    # z-scoring ON by default; pass --no_zscore_items to disable
    p.add_argument("--no_zscore_items", action="store_true",
                   help="Disable item-level z-scoring (default is z-score ON).")
    p.add_argument("--random_state", type=int, default=13)
    p.add_argument("--adult_ref_csv", default=None,
                   help="Optional CSV from your original adult script to diff per-subject accuracy.")
    p.add_argument("--sort_mode", choices=["item_id","beta_path","none"], default="item_id",
                   help="How to order the 12 items within subject (default: item_id).")
    return p.parse_args()

# --------------- helpers --------------- #
def load_mask(mask_path):
    img = nib.load(mask_path)
    data = img.get_fdata()
    return img, (data > 0.5)

def flat_idx(mask_bool):
    return np.where(mask_bool.reshape(-1))[0]

def sort_rows(df, mode):
    if mode == "item_id":
        return df.sort_values("item_id")
    elif mode == "beta_path":
        return df.sort_values("beta_path")
    else:
        return df  # keep original CSV order

def load_items(rows_df, flat_mask_idx):
    mats = []
    for _, r in rows_df.iterrows():
        vol = nib.load(r["beta_path"]).get_fdata().reshape(-1)
        mats.append(vol[flat_mask_idx])
    return np.vstack(mats).astype(np.float32)  # (12, Vmask)

def pairs_from_items(X_items, triplet_ids, zscore_items=True):
    X = X_items.copy()
    if zscore_items:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - mu) / sd
    idx_pairs = list(combinations(range(X.shape[0]), 2))  # 66
    Xp, yp = [], []
    trip = np.asarray(triplet_ids)
    for i, j in idx_pairs:
        Xp.append(np.abs(X[i] - X[j]))
        yp.append(1 if trip[i] == trip[j] else 0)
    return np.vstack(Xp), np.array(yp, dtype=int)

def make_clf(kind, C, seed):
    if kind == "logreg":
        model = LogisticRegression(penalty="l2", C=C, solver="lbfgs",
                                   max_iter=2000, class_weight="balanced",
                                   random_state=seed)
        return Pipeline([("scaler", StandardScaler()), ("clf", model)]), "prob"
    else:
        model = LinearSVC(C=C, class_weight="balanced", max_iter=5000, random_state=seed)
        return Pipeline([("scaler", StandardScaler()), ("clf", model)]), "dec"

def safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan

def sha1(a):
    return hashlib.sha1(np.ascontiguousarray(a)).hexdigest()

# --------------- main --------------- #
def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)
    zscore_items = not args.no_zscore_items

    # Load meta
    meta = pd.read_csv(args.meta_csv)
    required = {"subject","age_group","run","item_id","triplet_id","beta_path"}
    miss = required - set(meta.columns)
    if miss:
        raise ValueError(f"meta_csv missing: {sorted(miss)}")

    # Mask
    _, mbool = load_mask(args.mask_nii)
    fidx = flat_idx(mbool)
    print(f"[INFO] Mask voxels: {fidx.size}")

    # Adults (strict checks)
    adults = meta[(meta["age_group"] == "adult") & (meta["run"] == args.run)].copy()
    subs_adult = sorted(adults["subject"].unique())
    if not subs_adult:
        raise ValueError("No adult rows for requested run.")

    # Build subject item matrices with diagnostics
    subj_X, subj_trip, subj_item_order = {}, {}, {}
    for sid in subs_adult:
        df = adults[adults["subject"] == sid].copy()
        if df.shape[0] != 12:
            raise ValueError(f"[{sid}] has {df.shape[0]} rows for run {args.run}, expected 12.")
        df = sort_rows(df, args.sort_mode)

        tri_counts = df["triplet_id"].value_counts().sort_index()
        if not all(tri_counts.values == 3) or tri_counts.shape[0] != 4:
            raise ValueError(f"[{sid}] triplet composition invalid: {tri_counts.to_dict()} (expect 4 keys, each=3)")

        Xi = load_items(df, fidx)  # (12, V)
        subj_X[sid] = Xi
        subj_trip[sid] = df["triplet_id"].to_numpy()
        subj_item_order[sid] = list(df["item_id"].tolist())

        _, ytmp = pairs_from_items(Xi, subj_trip[sid], zscore_items=zscore_items)
        n_within = int((ytmp==1).sum())
        n_across = int((ytmp==0).sum())
        ysha = sha1(ytmp)
        print(f"[CHECK] {sid}: items=12, order(by {args.sort_mode})={subj_item_order[sid]}  "
              f"pairs within/across={n_within}/{n_across}  y_sha1={ysha[:10]}")

    # LOSO adults (z-scored items)
    clf, mode = make_clf(args.clf, args.C, args.random_state)
    adult_rows = []
    for test_sid in subs_adult:
        # training pool
        Xtr_list, ytr_list, fid_parts = [], [], []
        for sid in subs_adult:
            if sid == test_sid:
                continue
            Xi = subj_X[sid]; tri = subj_trip[sid]
            Xp, yp = pairs_from_items(Xi, tri, zscore_items=zscore_items)
            Xtr_list.append(Xp); ytr_list.append(yp)
            fid_parts.append(sha1(yp))
        Xtr = np.vstack(Xtr_list); ytr = np.hstack(ytr_list)
        design_fid = hashlib.sha1(("-".join(fid_parts)).encode()).hexdigest()

        clf.fit(Xtr, ytr)

        # test set
        Xi = subj_X[test_sid]; tri = subj_trip[test_sid]
        Xte, yte = pairs_from_items(Xi, tri, zscore_items=zscore_items)
        if mode == "prob":
            ysc = clf.predict_proba(Xte)[:,1]
            ypr = (ysc >= 0.5).astype(int)
        else:
            ysc = clf.decision_function(Xte)
            ypr = (ysc >= 0).astype(int)

        acc = accuracy_score(yte, ypr)
        auc = safe_auc(yte, ysc)
        prec, rec, f1, _ = precision_recall_fscore_support(yte, ypr, average="binary", zero_division=0)
        cm = confusion_matrix(yte, ypr, labels=[0,1])
        pos_rate = float((ypr == 1).mean())

        adult_rows.append({
            "subject": test_sid,
            "age_group": "adult",
            "age": float(meta.loc[(meta["subject"] == test_sid) & (meta["run"] == args.run), "age"].iloc[0]) if "age" in meta.columns else np.nan,
            "n_pairs": int(yte.size),
            "accuracy": float(acc),
            "auc": float(auc),
            "precision_within": float(prec),
            "recall_within": float(rec),
            "f1_within": float(f1),
            "tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1]),
            "mean_score_within": float(np.mean(ysc[yte==1])) if (yte==1).any() else np.nan,
            "mean_score_across": float(np.mean(ysc[yte==0])) if (yte==0).any() else np.nan,
            "score_delta_within_minus_across": float(np.mean(ysc[yte==1]) - np.mean(ysc[yte==0])),
            "train_design_fingerprint": design_fid,
            "pred_positive_rate": pos_rate,
            "item_order": ",".join(map(str, subj_item_order[test_sid])),
        })
        print(f"[LOSO ADULT] {test_sid}: acc={acc:.3f} auc={auc:.3f} within_rec={rec:.3f}  "
              f"pos_rate={pos_rate:.3f}  fid={design_fid[:10]}")

    adults_df = pd.DataFrame(adult_rows)

    # Optional: compare with reference adult CSV
    if args.adult_ref_csv and os.path.exists(args.adult_ref_csv):
        ref = pd.read_csv(args.adult_ref_csv)
        subj_col = "subject" if "subject" in ref.columns else ref.columns[0]
        acc_col = "accuracy" if "accuracy" in ref.columns else ("acc" if "acc" in ref.columns else None)
        if acc_col:
            merged = adults_df.merge(ref[[subj_col, acc_col]], left_on="subject", right_on=subj_col, how="left", suffixes=("","_ref"))
            merged["acc_diff"] = merged["accuracy"] - merged[f"{acc_col}_ref"]
            print("\n[DIAG] Per-subject accuracy difference vs reference (this - ref):")
            print(merged[["subject","accuracy",f"{acc_col}_ref","acc_diff"]].sort_values("acc_diff"))
        else:
            print("[DIAG] adult_ref_csv present but no recognizable accuracy column; skipping diff.")

    # Load child metrics and align columns (reindex avoids slice errors)
    kids_df = pd.read_csv(args.child_metrics_csv)
    if "age_group" not in kids_df.columns:
        kids_df["age_group"] = "child"

    expected = [
        "subject","age_group","age","n_pairs","accuracy","auc",
        "precision_within","recall_within","f1_within",
        "tn","fp","fn","tp",
        "mean_score_within","mean_score_across","score_delta_within_minus_across"
    ]
    adults_df = adults_df.reindex(columns=expected, fill_value=np.nan)
    kids_df   = kids_df.reindex(columns=expected,   fill_value=np.nan)

    combined = pd.concat([kids_df, adults_df], ignore_index=True)

    out_csv = os.path.join(args.outdir, "combined_metrics.csv")
    combined.to_csv(out_csv, index=False)
    print(f"\n[OK] Wrote combined CSV with {combined.shape[0]} rows → {out_csv}")
    print(f"[INFO] Item-level z-scoring: {zscore_items} | Sort mode: {args.sort_mode}")

if __name__ == "__main__":
    main()
