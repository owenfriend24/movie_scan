#!/usr/bin/env python
# Step 1: Adult Run-4 voxelwise Wilcoxon (+ top-K mask), with item-wise z-scoring and min cluster extent

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import wilcoxon, norm
from scipy.ndimage import label
import argparse, os

# ---------------- args (exactly as requested) ---------------- #
def get_args():
    p = argparse.ArgumentParser(description="Adult Run-4 Wilcoxon (Δ = mean_across - mean_within) + top-K mask")
    p.add_argument("meta_csv",
                   help="CSV: subject,age,age_group,run,item_id,triplet_id,beta_path")
    p.add_argument("gm_mask", help="MNI GM mask (binary NIfTI)")
    p.add_argument("outdir",  help="Output directory")
    p.add_argument("--topk", type=int, default=1000, help="Top-K voxels (positive direction only)")
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

def subject_delta_abs_zscored(X_items, triplets):
    """
    X_items: (12, Vgm) betas for one subject, GM-masked
    triplets: length-12 array aligned to rows of X_items
    Returns Δ(v) = mean_across(v) - mean_within(v) using absolute differences,
    after z-scoring the 12 items per voxel (column-wise).
    """
    # z-score across items for each voxel (column-wise)
    m = X_items.mean(axis=0, keepdims=True)
    s = X_items.std(axis=0, keepdims=True) + 1e-8
    Xz = (X_items - m) / s

    n_items, _ = Xz.shape
    I, J = np.triu_indices(n_items, k=1)          # 66 pairs
    diffs = Xz[I, :] - Xz[J, :]                   # (66, Vgm)
    D = np.abs(diffs)

    same_trip = (triplets[I] == triplets[J])      # 12 within pairs
    m_within = D[same_trip, :].mean(axis=0)
    m_across = D[~same_trip, :].mean(axis=0)      # 54 across pairs
    return (m_across - m_within).astype(np.float32)   # positive ⇒ within < across

def to_full(arr_gm, shape, gm_flat_idx, dtype=np.float32):
    full = np.zeros(np.prod(shape), dtype=dtype)
    full[gm_flat_idx] = arr_gm
    return full.reshape(shape)

# ---------------- main ---------------- #
if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    # load meta and filter to adults, run 4
    meta = pd.read_csv(args.meta_csv)
    subset = meta[(meta["age_group"] == "adult") & (meta["run"] == 4)].copy()
    subjects = sorted(subset["subject"].unique())
    if len(subjects) < 2:
        raise ValueError("Need at least 2 adults for Wilcoxon group test.")

    # GM mask
    gm_img = nib.load(args.gm_mask)
    affine, shape, header = gm_img.affine, gm_img.shape, gm_img.header
    gm_flat_idx, _ = gm_flat_indices(gm_img)
    Vgm = gm_flat_idx.size

    # compute subject Δ maps (Δ = mean_across - mean_within), with item-wise z-scoring
    subj_deltas = []
    for sid in subjects:
        rows = subset[subset["subject"] == sid].sort_values("item_id")
        X = load_subject_item_stack(rows, gm_flat_idx)              # 12 × Vgm
        trips = rows["triplet_id"].to_numpy()
        delta = subject_delta_abs_zscored(X, trips)                 # Vgm
        subj_deltas.append(delta)
    subj_deltas = np.vstack(subj_deltas)                             # Nsub × Vgm

    # group Wilcoxon (signed by median Δ)
    effect_mean = subj_deltas.mean(axis=0).astype(np.float32)
    z_map = np.zeros(Vgm, np.float32)
    p_map = np.ones(Vgm, np.float32)
    med = np.median(subj_deltas, axis=0)

    for v in range(Vgm):
        x = subj_deltas[:, v]
        nz = x[x != 0]
        if nz.size < 2:
            continue
        try:
            w = wilcoxon(nz, zero_method="wilcox", correction=False, alternative="two-sided", mode="auto")
            p = w.pvalue if w is not None else 1.0
        except Exception:
            p = 1.0
        p = np.clip(p, 1e-300, 1.0)
        z_unsigned = norm.isf(p / 2.0)
        z_map[v] = z_unsigned * np.sign(med[v])
        p_map[v] = p

    # save maps
    nib.save(nib.Nifti1Image(to_full(effect_mean, shape, gm_flat_idx), affine, header),
             os.path.join(args.outdir, "adult_run4_effect_mean.nii.gz"))
    nib.save(nib.Nifti1Image(to_full(z_map, shape, gm_flat_idx), affine, header),
             os.path.join(args.outdir, "adult_run4_z.nii.gz"))
    nib.save(nib.Nifti1Image(to_full(p_map, shape, gm_flat_idx), affine, header),
             os.path.join(args.outdir, "adult_run4_p.nii.gz"))

    # --------- cluster-extent filtering (min 10 voxels) BEFORE top-K selection ---------
    # Build a full-volume boolean of positive z within GM, then label connected components
    pos_full = to_full((z_map > 0).astype(np.uint8), shape, gm_flat_idx, dtype=np.uint8)

    # 26-connectivity structure (3x3x3 ones) for more inclusive clusters
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, ncomp = label(pos_full, structure=structure)

    if ncomp == 0:
        print("[WARN] No positive clusters found. Top-K mask will be empty.")
        keepable_full = np.zeros(shape, dtype=bool)
    else:
        # Keep only clusters with size >= 10 voxels
        sizes = np.bincount(labeled.reshape(-1))
        # sizes[0] is background; create a mask of voxels in clusters meeting the extent threshold
        big_labels = np.where(sizes >= 10)[0]
        big_labels = big_labels[big_labels != 0]  # exclude background
        keepable_full = np.isin(labeled, big_labels)

    # Map the cluster-extent mask back to GM-flat indices
    keepable_flat = keepable_full.reshape(-1)[gm_flat_idx]

    # rank positive z among keepable voxels only
    z_pos = z_map.copy()
    z_pos[~keepable_flat] = -np.inf
    order = np.argsort(z_pos)[::-1]
    order = order[np.isfinite(z_pos[order])]
    keep_idx = order[:min(args.topk, order.size)]

    # build and save the Top-K mask (GM-flat → full volume)
    topk_flat = np.zeros(Vgm, np.uint8)
    if keep_idx.size > 0:
        topk_flat[keep_idx] = 1
    topk_img = nib.Nifti1Image(to_full(topk_flat, shape, gm_flat_idx, dtype=np.uint8), affine, header)
    nib.save(topk_img, os.path.join(args.outdir, f"adult_run4_top{args.topk}_mask.nii.gz"))

    # small summary
    with open(os.path.join(args.outdir, "summary_step1.txt"), "w") as f:
        f.write(f"N adults: {len(subjects)}\n")
        f.write("Item-wise z-scoring: yes\n")
        f.write("Smoothing: no\n")
        f.write("Cluster extent threshold: 10 voxels (26-connectivity)\n")
        f.write(f"Positive clusters found: {int((labeled>0).any())}\n")
        f.write(f"Top-K requested: {args.topk}, Top-K kept: {int(keep_idx.size)}\n")
        if keep_idx.size > 0:
            f.write(f"Last-kept z (threshold among keepable): {z_map[keep_idx[-1]]:.4f}\n")

    print(f"Done. Wrote maps and clustered top-{args.topk} mask to: {args.outdir}")
