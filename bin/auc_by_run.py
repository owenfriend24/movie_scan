#!/usr/bin/env python
"""
AUC across Runs 1-4 using adult-run4 classifier(s).

- Adults: LOSO model trained on adults' Run 4 (exclude the test adult), then applied to that adult's Runs 1..4.
- Children: One pooled model trained on ALL adults' Run 4, applied to each child's Runs 1..4.

Features:
  - For each (subject, run): 12 items -> 66 pairs
  - X_pair = |beta_i - beta_j| within the provided mask
  - y_pair = 1 if same triplet else 0
  - Item-level z-scoring across the 12 items per voxel (default ON)

Inputs:
  meta_csv: columns must include {subject, age_group, run, item_id, triplet_id, beta_path}
  mask_nii: binary NIfTI mask (e.g., your group top-% Wilcoxon-z mask)
  outdir:   output directory

Produces:
  - per_run_metrics.csv  with rows: subject, age_group, run, auc, accuracy, precision_within, recall_within, ...
  - prints summaries of mean AUC by run and age_group
"""

import os, argparse, numpy as np, pandas as pd, nibabel as nib
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_recall_fscore_support,
                             confusion_matrix)

# ---------------- CLI ---------------- #
def get_args():
    p = argparse.ArgumentParser(description="AUC change from Runs 1-4 using adult-run4 classifier(s).")
    p.add_argument("meta_csv")
    p.add_argument("mask_nii")
    p.add_argument("outdir")
    p.add_argument("--clf", choices=["logreg","svm"], default="logreg")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--run_train", type=int, default=4, help="Adult training run (default: 4)")
    p.add_argument("--runs_eval", type=int, nargs="+", default=[1,2,3,4], help="Runs to evaluate")
    p.add_argument("--no_zscore_items", action="store_true",
                   help="Disable item-wise zscoring (default ON)")
    p.add_argument("--sort_mode", choices=["item_id","beta_path","none"], default="item_id",
                   help="How to order the 12 items within a subject/run")
    p.add_argument("--random_state", type=int, default=13)
    return p.parse_args()

# ---------------- helpers ---------------- #
def load_mask(mask_path):
    img = nib.load(mask_path)
    data = img.get_fdata()
    return (data > 0.5)

def sort_rows(df, mode):
    if mode == "item_id":
        return df.sort_values("item_id")
    elif mode == "beta_path":
        return df.sort_values("beta_path")
    else:
        return df  # keep meta order

def load_items(rows_df, flat_idx):
    mats = []
    for _, r in rows_df.iterrows():
        vol = nib.load(r["beta_path"]).get_fdata().reshape(-1)
        mats.append(vol[flat_idx])
    return np.vstack(mats).astype(np.float32)  # (12, V)

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
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
        return pipe, "prob"
    else:
        model = LinearSVC(C=C, class_weight="balanced", max_iter=5000, random_state=seed)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
        return pipe, "dec"

def safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan

# ---------------- main ---------------- #
def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load meta & check
    meta = pd.read_csv(args.meta_csv)
    required = {"subject","age_group","run","item_id","triplet_id","beta_path"}
    miss = required - set(meta.columns)
    if miss:
        raise ValueError(f"meta_csv missing: {sorted(miss)}")

    # Mask
    mask_bool = load_mask(args.mask_nii)
    flat_idx = np.where(mask_bool.reshape(-1))[0]
    print(f"[INFO] Mask voxels: {flat_idx.size}")

    # Split adult training set (Run = run_train)
    adults_train = meta[(meta["age_group"] == "adult") & (meta["run"] == args.run_train)].copy()
    adult_ids = sorted(adults_train["subject"].unique())
    if not adult_ids:
        raise ValueError("No adult rows for the specified training run.")

    # Build per-(subject, run) pair datasets for ALL subjects we might evaluate
    # { (sid, run) : (X_pairs, y_pairs) }
    zscore_items = not args.no_zscore_items
    per_sr = {}
    eval_subs = sorted(meta["subject"].unique())
    # Only evaluate requested runs and subjects that have 12 items with correct triplet structure
    for sid in eval_subs:
        for r in args.runs_eval:
            rows = meta[(meta["subject"] == sid) & (meta["run"] == r)].copy()
            if rows.shape[0] != 12:
                continue
            rows = sort_rows(rows, args.sort_mode)
            tri_counts = rows["triplet_id"].value_counts().sort_index()
            if not (tri_counts.shape[0] == 4 and np.all(tri_counts.values == 3)):
                continue
            Xi = load_items(rows, flat_idx)      # (12, V)
            trip = rows["triplet_id"].to_numpy()
            Xp, yp = pairs_from_items(Xi, trip, zscore_items=zscore_items)
            per_sr[(sid, r)] = (Xp, yp)

    # Pre-build classifiers
    clf_template, mode = make_clf(args.clf, args.C, args.random_state)

    # 1) Adult LOSO models trained on adults' Run-4
    adult_model_by_test = {}
    for test_sid in adult_ids:
        Xtr_list, ytr_list = [], []
        for sid in adult_ids:
            if sid == test_sid:
                continue
            key = (sid, args.run_train)
            if key not in per_sr:
                raise ValueError(f"Missing pairs for adult {sid} run {args.run_train}")
            Xp, yp = per_sr[key]
            Xtr_list.append(Xp); ytr_list.append(yp)
        Xtr = np.vstack(Xtr_list); ytr = np.hstack(ytr_list)

        # fresh clone-ish pipeline
        clf = Pipeline(clf_template.steps)
        clf.fit(Xtr, ytr)
        adult_model_by_test[test_sid] = clf

    # 2) Pooled adult model (for children)
    Xtr_list, ytr_list = [], []
    for sid in adult_ids:
        key = (sid, args.run_train)
        Xp, yp = per_sr[key]
        Xtr_list.append(Xp); ytr_list.append(yp)
    Xtr_all = np.vstack(Xtr_list); ytr_all = np.hstack(ytr_list)
    pooled_adult_clf = Pipeline(clf_template.steps)
    pooled_adult_clf.fit(Xtr_all, ytr_all)

    # Evaluate AUC per subject per run
    rows_out = []
    all_subs = sorted({sid for (sid, r) in per_sr.keys()})
    for sid in all_subs:
        # choose model
        # adults: LOSO model; children/other: pooled adult model
        is_adult = (meta.loc[meta["subject"] == sid, "age_group"].iloc[0] == "adult")
        model = adult_model_by_test[sid] if is_adult else pooled_adult_clf
        age_val = np.nan
        if "age" in meta.columns:
            ages = meta.loc[(meta["subject"] == sid), "age"]
            if not ages.empty:
                age_val = float(ages.iloc[0])

        for r in args.runs_eval:
            key = (sid, r)
            if key not in per_sr:
                continue
            Xte, yte = per_sr[key]
            if mode == "prob":
                y_score = model.predict_proba(Xte)[:, 1]
                y_pred  = (y_score >= 0.5).astype(int)
            else:
                y_score = model.decision_function(Xte)
                y_pred  = (y_score >= 0).astype(int)

            auc = safe_auc(yte, y_score)
            acc = accuracy_score(yte, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(yte, y_pred, average="binary", zero_division=0)
            cm = confusion_matrix(yte, y_pred, labels=[0,1])

            rows_out.append({
                "subject": sid,
                "age_group": "adult" if is_adult else "child",
                "age": age_val,
                "run": r,
                "n_pairs": int(yte.size),
                "auc": float(auc),
                "accuracy": float(acc),
                "precision_within": float(prec),
                "recall_within": float(rec),
                "f1_within": float(f1),
                "tn": int(cm[0,0]), "fp": int(cm[0,1]),
                "fn": int(cm[1,0]), "tp": int(cm[1,1]),
            })
        print(f"[EVAL] subject={sid}  done runs={sorted([rr for (ss, rr) in per_sr.keys() if ss==sid])}")

    out_df = pd.DataFrame(rows_out).sort_values(["age_group","subject","run"])
    out_csv = os.path.join(args.outdir, "per_run_metrics.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"\n[OK] wrote {out_csv}  (rows={out_df.shape[0]})")

    # Summaries: mean AUC by run & age_group
    if not out_df.empty:
        print("\n===== Mean AUC by run & age_group =====")
        summ = (out_df.groupby(["age_group","run"])["auc"]
                .mean().reset_index().pivot(index="run", columns="age_group", values="auc"))
        print(summ.round(4).fillna("NA"))

if __name__ == "__main__":
    main()
