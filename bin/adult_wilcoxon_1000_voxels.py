#!/usr/bin/env python
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import wilcoxon, norm
import argparse, os

# ---------------- args ---------------- #
def get_args():
    p = argparse.ArgumentParser(description="Step 1: Adult Run-4 Wilcoxon + top-1000 mask")
    p.add_argument("meta_csv",
                   help="CSV with subject,age,age_group,run,item_id,triplet_id,beta_path")
    p.add_argument("gm_mask", help="MNI gray matter mask (binary NIfTI)")
    p.add_argument("outdir",help="Output directory")
    p.add_argument("--topk", type=int, default=1000, help="Number of voxels to keep in top-K mask")
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

def subject_delta(X_items, triplets):
    n_items, V = X_items.shape
    deltas = np.zeros(V, np.float32)
    # all 66 pairs
    for i in range(n_items):
        for j in range(i+1, n_items):
            d = np.abs(X_items[i] - X_items[j])
            if triplets[i] == triplets[j]:
                deltas -= d / 3.0   # 3 within pairs per triplet
            else:
                deltas += d / 54.0  # 54 across pairs total
    return deltas  # positive = triplet selectivity

# ---------------- main ---------------- #
if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    # load meta
    meta = pd.read_csv(args.meta_csv)
    subset = meta[(meta["age_group"] == "adult") & (meta["run"] == 4)]

    # load GM mask
    gm_img = nib.load(args.gm_mask)
    affine, shape, header = gm_img.affine, gm_img.shape, gm_img.header
    gm_flat_idx, _ = gm_flat_indices(gm_img)

    # compute subject deltas
    subjects = sorted(subset["subject"].unique())
    subj_deltas = []
    for sid in subjects:
        rows = subset[subset["subject"] == sid].sort_values("item_id")
        X = load_subject_item_stack(rows, gm_flat_idx)     # 12 × Vgm
        triplets = rows["triplet_id"].to_numpy()
        delta = subject_delta(X, triplets)                 # Vgm
        subj_deltas.append(delta)
    subj_deltas = np.vstack(subj_deltas)                   # Nsub × Vgm
    effect_mean = subj_deltas.mean(axis=0)

    # Wilcoxon signed-rank per voxel
    Vgm = subj_deltas.shape[1]
    z_map = np.zeros(Vgm, np.float32)
    p_map = np.ones(Vgm, np.float32)
    med = np.median(subj_deltas, axis=0)
    for v in range(Vgm):
        x = subj_deltas[:, v]
        nz = x[x != 0]
        if nz.size < 2:
            continue
        try:
            w = wilcoxon(nz, zero_method="wilcox", alternative="two-sided", mode="auto")
            p = w.pvalue if w is not None else 1.0
        except Exception:
            p = 1.0
        p = np.clip(p, 1e-300, 1.0)
        z = norm.isf(p/2.0)   # unsigned
        z_map[v] = z * np.sign(med[v])
        p_map[v] = p

    # save maps
    def to_full(arr_gm):
        full = np.zeros(np.prod(shape), np.float32)
        full[gm_flat_idx] = arr_gm
        return full.reshape(shape)

    nib.save(nib.Nifti1Image(to_full(effect_mean), affine, header),
             os.path.join(args.outdir, "adult_run4_effect_mean.nii.gz"))
    nib.save(nib.Nifti1Image(to_full(z_map), affine, header),
             os.path.join(args.outdir, "adult_run4_z.nii.gz"))
    nib.save(nib.Nifti1Image(to_full(p_map), affine, header),
             os.path.join(args.outdir, "adult_run4_p.nii.gz"))

    # top-K mask (positive z only)
    pos = z_map > 0
    order = np.argsort(z_map)[::-1]
    order = order[pos[order]]
    keep = order[:min(args.topk, len(order))]
    topk = np.zeros(Vgm, np.uint8); topk[keep] = 1
    nib.save(nib.Nifti1Image(to_full(topk), affine, header),
             os.path.join(args.outdir, f"adult_run4_top{args.topk}_mask.nii.gz"))

    print(f"Finished Step 1. Saved maps and top-{args.topk} mask in {args.outdir}")
