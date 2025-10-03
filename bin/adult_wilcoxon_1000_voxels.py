#!/usr/bin/env python
# Step 1: Adult Run-4 voxelwise Wilcoxon (+ top-K mask), correct Δ with 12 within and 54 across pairs
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import wilcoxon, norm
import argparse, os

# ---------------- args ---------------- #
def get_args():
    p = argparse.ArgumentParser(description="Adult Run-4 Wilcoxon (Δ = mean_across - mean_within) + top-K mask")
    p.add_argument("meta_csv", required=True,
                   help="CSV: subject,age,age_group,run,item_id,triplet_id,beta_path")
    p.add_argument("gm_mask", required=True, help="MNI GM mask (binary NIfTI)")
    p.add_argument("outdir", required=True, help="Output directory")
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

def subject_delta_abs(X_items, triplets):
    """
    X_items: (12, Vgm) betas for one subject
    triplets: length-12 array of triplet IDs (e.g., 1..4), aligned to rows of X_items
    Returns Δ(v) = mean_across(v) - mean_within(v), using absolute differences.
    This averages over the correct counts: 12 within pairs, 54 across pairs.
    """
    n_items, V = X_items.shape
    # build all 66 unique pairs (i<j)
    I, J = np.triu_indices(n_items, k=1)
    diffs = X_items[I, :] - X_items[J, :]        # (66, Vgm)
    D = np.abs(diffs)                             # pairwise absolute differences per voxel
    same_trip = (triplets[I] == triplets[J])      # boolean mask: within-triplet pairs (size 66)

    m_within = D[same_trip, :].mean(axis=0)       # average over 12 within pairs
    m_across = D[~same_trip, :].mean(axis=0)      # average over 54 across pairs
    return (m_across - m_within).astype(np.float32)  # positive ⇒ within < across (triplet selectivity)

# ---------------- main ---------------- #
if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    # load meta and filter to adults, run 4
    meta = pd.read_csv(args.meta_csv)
    subset = meta[(meta["age_group"] == "adult") & (meta["run"] == 4)].copy()

    # GM mask
    gm_img = nib.load(args.gm_mask)
    affine, shape, header = gm_img.affine, gm_img.shape, gm_img.header
    gm_flat_idx, _ = gm_flat_indices(gm_img)

    # compute subject Δ maps
    subjects = sorted(subset["subject"].unique())
    subj_deltas = []
    for sid in subjects:
        rows = subset[subset["subject"] == sid].sort_values("item_id")
        X = load_subject_item_stack(rows, gm_flat_idx)              # 12 × Vgm
        trips = rows["triplet_id"].to_numpy()
        delta = subject_delta_abs(X, trips)                          # Vgm
        subj_deltas.append(delta)
    subj_deltas = np.vstack(subj_deltas)                             # Nsub × Vgm

    # group Wilcoxon (signed by median Δ)
    Vgm = subj_deltas.shape[1]
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

    # write NIfTI maps
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

    # Top-K mask in the positive (adult-like) direction
    order = np.argsort(z_map)[::-1]                 # descending z
    order = order[z_map[order] > 0]                 # keep only positive z (within < across)
    keep = order[:min(args.topk, order.size)]
    topk = np.zeros(Vgm, np.uint8); topk[keep] = 1
    nib.save(nib.Nifti1Image(to_full(topk), affine, header),
             os.path.join(args.outdir, f"adult_run4_top{args.topk}_mask.nii.gz"))

    print(f"Done. Saved mean Δ, Wilcoxon z/p, and top-{args.topk} mask to {args.outdir}")
