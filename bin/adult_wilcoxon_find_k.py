#!/usr/bin/env python
import subprocess
subprocess.run(['/bin/bash', '-c', 'source /home1/09123/ofriend/analysis/temple/rsa/bin/activate'])

import os, argparse
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import wilcoxon, norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

# ---------------- args ---------------- #
def get_args():
    p = argparse.ArgumentParser(
        description="Step 2: LOSO choose K (adults, run 4) using training-only Wilcoxon ranking."
    )
    p.add_argument("meta_csv",
                   help="Master CSV with subject,age,age_group,run,item_id,triplet_id,beta_path")
    p.add_argument("gm_mask", help="MNI GM mask (binary NIfTI)")
    p.add_argument("outdir", help="Output directory")
    p.add_argument("--k_grid", type=str, default="100,250,500,750,1000,1500,2000,3000",
                   help="Comma-separated K values to evaluate (top-K voxels)")
    p.add_argument("--min_effect", type=float, default=0.0,
                   help="Require training-set mean Δ >= min_effect for inclusion when ranking")
    p.add_argument("--balance_pairs", action="store_true",
                   help="Downsample across-pairs to match within count per subject")
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()

# ---------------- helpers ---------------- #
def gm_flat_indices(mask_img):
    gm = mask_img.get_fdata().astype(bool).reshape(-1)
    return np.where(gm)[0], gm

def load_subject_item_stack(rows, gm_flat_idx):
    mats = []
    for _, r in rows.iterrows():
        arr = nib.load(r["beta_path"]).get_fdata().reshape(-1)
        mats.append(arr[gm_flat_idx])
    return np.vstack(mats).astype(np.float32)  # 12 × Vgm

def all_pairs_12():
    P = []
    for i in range(12):
        for j in range(i+1,12):
            P.append((i,j))
    return np.array(P, dtype=int)

def subject_delta_abs(X_items, triplets, pairs):
    # Δ(v) = mean_across - mean_within using absolute differences per voxel
    diffs = X_items[pairs[:,0], :] - X_items[pairs[:,1], :]
    D = np.abs(diffs)
    labs = (triplets[pairs[:,0]] == triplets[pairs[:,1]])
    m_within = D[labs,:].mean(axis=0)
    m_across = D[~labs,:].mean(axis=0)
    return (m_across - m_within).astype(np.float32)

def wilcoxon_signed_z(deltas_subj_by_voxel):
    """
    deltas_subj_by_voxel: array shape (Nsub, Vgm) of subject Δ values
    returns z (signed by median) and p arrays of shape (Vgm,)
    """
    Nsub, Vgm = deltas_subj_by_voxel.shape
    z = np.zeros(Vgm, np.float32)
    p = np.ones(Vgm, np.float32)
    med = np.median(deltas_subj_by_voxel, axis=0)
    for v in range(Vgm):
        x = deltas_subj_by_voxel[:, v]
        nz = x[x != 0]
        if nz.size < 2:
            continue
        try:
            w = wilcoxon(nz, zero_method="wilcox", correction=False, alternative="two-sided", mode="auto")
            pv = w.pvalue if w is not None else 1.0
        except Exception:
            pv = 1.0
        pv = np.clip(pv, 1e-300, 1.0)
        z0 = norm.isf(pv/2.0)  # unsigned
        z[v] = z0 * np.sign(med[v])
        p[v] = pv
    return z, p

def build_pair_features(X_items, triplets, vox_idx, balance_pairs=False):
    """
    X_items: (12, Vgm) float32, already masked to GM.
    vox_idx: indices (into GM-flat) to keep as features.
    Returns X_pairs (n_pairs x K), y_pairs (0/1)
    """
    pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
    X = X_items[:, vox_idx].astype(np.float32)  # 12 x K
    # z-score each item vector across voxels (stabilizes scale across items)
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True) + 1e-8
    X = (X - m) / s
    feats, labels = [], []
    for (i,j) in pairs:
        feats.append((X[i] - X[j]))                 # K-dim feature
        labels.append(int(triplets[i] == triplets[j]))
    Xp = np.vstack(feats).astype(np.float32)
    yp = np.array(labels, dtype=np.int32)
    if balance_pairs:
        pos = np.where(yp == 1)[0]
        neg = np.where(yp == 0)[0]
        if len(pos) and len(neg):
            rng = np.random.RandomState(123)
            n = min(len(pos), len(neg))
            posk = rng.choice(pos, n, replace=False)
            negk = rng.choice(neg, n, replace=False)
            keep = np.sort(np.r_[posk, negk])
            Xp = Xp[keep]
            yp = yp[keep]
    return Xp, yp

# ---------------- main ---------------- #
if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load and filter meta
    meta = pd.read_csv(args.meta_csv)
    subset = meta[(meta["age_group"] == "adult") & (meta["run"] == 4)].copy()
    subjects = sorted(subset["subject"].unique().tolist())

    # GM mask + indices
    gm_img = nib.load(args.gm_mask)
    affine, shape, hdr = gm_img.affine, gm_img.shape, gm_img.header
    gm_flat_idx, _ = gm_flat_indices(gm_img)

    # Preload adult item betas and per-subject Δ (fast later)
    pairs = all_pairs_12()
    subj_X = {}          # sid -> (12, Vgm)
    subj_trip = {}       # sid -> np.array(12,)
    subj_delta = {}      # sid -> (Vgm,)
    for sid in subjects:
        rows = subset[subset["subject"] == sid].sort_values("item_id")
        X = load_subject_item_stack(rows, gm_flat_idx)  # 12 x Vgm
        trips = rows["triplet_id"].to_numpy()
        subj_X[sid] = X
        subj_trip[sid] = trips
        subj_delta[sid] = subject_delta_abs(X, trips, pairs)  # Vgm

    # LOSO over adults
    K_grid = [int(x) for x in args.k_grid.split(",") if x.strip()]
    K_grid = sorted(list(set([k for k in K_grid if k > 0])))

    # Prepare arrays for fast indexing
    deltas_matrix = np.vstack([subj_delta[sid] for sid in subjects])  # Nsub x Vgm
    groups = np.array(subjects, dtype=object)
    logo = LeaveOneGroupOut()

    rng = np.random.RandomState(args.seed)
    aucs_per_K = {K: [] for K in K_grid}

    for tr_idx, te_idx in logo.split(groups, groups, groups):
        train_subs = [subjects[i] for i in tr_idx]
        test_sub   = subjects[te_idx[0]]

        # Training-only Wilcoxon z map
        train_rows = [subjects.index(s) for s in train_subs]
        z_train, _ = wilcoxon_signed_z(deltas_matrix[train_rows, :])

        # Optionally require positive direction AND min_effect (on training set mean Δ)
        mean_delta_train = deltas_matrix[train_rows, :].mean(axis=0)
        keepable = (z_train > 0) & (mean_delta_train >= args.min_effect)
        # rank by z (descending), but only among keepable voxels
        rank_vals = z_train.copy()
        rank_vals[~keepable] = -np.inf
        order = np.argsort(rank_vals)[::-1]
        order = order[np.isfinite(rank_vals[order])]

        for K in K_grid:
            if order.size == 0:
                aucs_per_K[K].append(np.nan)
                continue
            vox_idx = order[:min(K, order.size)]

            # build train set
            Xtr_list, ytr_list = [], []
            for sid in train_subs:
                Xi, yi = build_pair_features(subj_X[sid], subj_trip[sid], vox_idx,
                                             balance_pairs=args.balance_pairs)
                Xtr_list.append(Xi); ytr_list.append(yi)
            Xtr = np.vstack(Xtr_list)
            ytr = np.concatenate(ytr_list)

            # skip fold if y has <2 classes (unlikely with balancing)
            if len(np.unique(ytr)) < 2:
                aucs_per_K[K].append(np.nan)
                continue

            clf = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter=1000)
            clf.fit(Xtr, ytr)

            # test on held-out subject
            Xte, yte = build_pair_features(subj_X[test_sub], subj_trip[test_sub], vox_idx,
                                           balance_pairs=args.balance_pairs)
            # if test has one class only (rare), fallback to nan
            if len(np.unique(yte)) < 2:
                aucs_per_K[K].append(np.nan)
                continue
            p = clf.predict_proba(Xte)[:, 1]
            auc = roc_auc_score(yte, p)
            aucs_per_K[K].append(auc)

    # Aggregate across folds and select K by 1-SE rule
    rows = []
    means, ses = {}, {}
    for K in K_grid:
        vals = np.array([a for a in aucs_per_K[K] if np.isfinite(a)])
        m = float(np.nanmean(vals)) if vals.size else np.nan
        se = float(np.nanstd(vals, ddof=1)/np.sqrt(vals.size)) if vals.size > 1 else np.nan
        rows.append({"K": K, "mean_auc": m, "se": se, "n_folds": int(vals.size)})
        means[K] = m; ses[K] = se

    df = pd.DataFrame(rows).sort_values("K")
    df.to_csv(os.path.join(args.outdir, "loso_k_selection.csv"), index=False)

    # choose K
    valid = [K for K in K_grid if np.isfinite(means.get(K, np.nan))]
    if len(valid) == 0:
        chosen_K = K_grid[0]
    else:
        bestK = max(valid, key=lambda k: means[k])
        target = means[bestK] - (ses[bestK] if np.isfinite(ses[bestK]) else 0.0)
        # smallest K with mean ≥ target
        candidates = [K for K in valid if means[K] >= target]
        chosen_K = min(candidates) if len(candidates) else bestK

    with open(os.path.join(args.outdir, "chosen_K.txt"), "w") as f:
        f.write(str(chosen_K) + "\n")

    print(f"[K-SELECTION] Chosen K = {chosen_K}")

    # Build FINAL top-K mask using ALL adults (positive direction + min_effect)
    z_all, _ = wilcoxon_signed_z(deltas_matrix)
    mean_delta_all = deltas_matrix.mean(axis=0)
    keepable_all = (z_all > 0) & (mean_delta_all >= args.min_effect)
    ranks = z_all.copy(); ranks[~keepable_all] = -np.inf
    order_all = np.argsort(ranks)[::-1]
    order_all = order_all[np.isfinite(ranks[order_all])]
    vox_idx_final = order_all[:min(chosen_K, order_all.size)]

    # save mask
    Vgm = deltas_matrix.shape[1]
    topk_flat = np.zeros(Vgm, np.uint8); topk_flat[vox_idx_final] = 1
    def to_full(arr_gm):
        full = np.zeros(np.prod(shape), np.float32)
        full[np.where(nib.load(args.gm_mask).get_fdata().astype(bool).reshape(-1))[0]] = arr_gm
        return full.reshape(shape)
    nib.save(nib.Nifti1Image(to_full(topk_flat), affine, hdr),
             os.path.join(args.outdir, f"adult_run4_top{chosen_K}_mask.nii.gz"))

    # also save the ranking (GM-flat indices) for reproducibility
    np.save(os.path.join(args.outdir, "voxel_ranking_all_adults.npy"), order_all.astype(np.int64))
    np.save(os.path.join(args.outdir, f"top{chosen_K}_voxel_indices.npy"), np.array(vox_idx_final, dtype=np.int64))

    print(f"Saved K sweep (loso_k_selection.csv), chosen_K.txt, and final mask in {args.outdir}")
