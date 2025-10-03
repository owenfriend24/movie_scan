#!/usr/bin/env python
"""
Adult group-level triplet-selective mask using Wilcoxon z-map and top-% threshold.

Pipeline:
  1) For each ADULT subject at a given run:
       Δ(v) = mean_across |β_i - β_j| - mean_within |β_i - β_j|
       (optionally z-score the 12 items per voxel before pairwise diffs)
  2) Stack Δ across adults (Nsub × Vgm) and run Wilcoxon signed-rank per voxel.
     Convert p to z and sign by the median(Δ) so positive z => within more similar than across.
  3) Threshold the group z-map at the top X% within the mask (GM by default):
       - Default: positive direction only (z > 0)
       - With --bidirectional: either direction using |z|

Outputs to OUTDIR:
  - group_adults_run-<R>_wilcoxon_z_triplet_delta.nii.gz
  - group_adults_run-<R>_top<percent>pct_<MASKNAME>_<mode>_wilcoxon_z_triplet_mask.nii.gz
  - summary_group.txt
"""

import numpy as np
import pandas as pd
import nibabel as nib
import argparse, os
from scipy.stats import wilcoxon, norm

# ---------------- args ---------------- #
def get_args():
    p = argparse.ArgumentParser(
        description="Adult group Wilcoxon z-map + top-% threshold (positive or bidirectional)."
    )
    p.add_argument("meta_csv",
                   help="CSV with at least: subject,age_group,run,item_id,triplet_id,beta_path")
    p.add_argument("gm_mask", help="MNI gray-matter mask (binary NIfTI)")
    p.add_argument("outdir",  help="Output directory root")

    p.add_argument("--run", type=int, default=4, help="Run to use (default: 4)")
    p.add_argument("--top_percent", type=float, default=5.0,
                   help="Percentage of voxels to keep within mask. Default: 5.0")
    p.add_argument("--roi_mask", default=None,
                   help="Optional ROI mask NIfTI; if given, threshold within ROI∩GM; else within GM.")
    p.add_argument("--zscore_items", action="store_true",
                   help="Z-score the 12 item betas per voxel before pairwise diffs (optional).")
    p.add_argument("--bidirectional", action="store_true",
                   help="Select by |z| (either direction) instead of positive z only.")
    return p.parse_args()

# ---------------- helpers ---------------- #
def load_mask(path):
    img = nib.load(path)
    data = img.get_fdata().astype(bool)
    return img, data

def gm_indices(gm_bool):
    return np.where(gm_bool.reshape(-1))[0]

def load_item_stack(rows, gm_idx):
    mats = []
    for _, r in rows.iterrows():
        arr = nib.load(r["beta_path"]).get_fdata().reshape(-1)
        mats.append(arr[gm_idx])
    return np.vstack(mats).astype(np.float32)  # 12 x Vgm

def compute_delta(X_items, triplets, zscore_items=False):
    """
    Δ(v) = mean_across |β_i - β_j| - mean_within |β_i - β_j|
    X_items: (12, Vgm), triplets: (12,)
    """
    X = X_items
    if zscore_items:
        m = X.mean(axis=0, keepdims=True)
        s = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - m) / s

    I, J = np.triu_indices(12, k=1)               # 66 pairs
    D = np.abs(X[I, :] - X[J, :])                 # 66 x Vgm
    within = (triplets[I] == triplets[J])         # 12 within
    m_within = D[within, :].mean(axis=0)
    m_across = D[~within, :].mean(axis=0)         # 54 across
    return (m_across - m_within).astype(np.float32)  # positive => within more similar

def save_full_like(vec_gm, mask_bool, like_img, out_path, dtype=np.float32):
    shape, aff, hdr = like_img.shape, like_img.affine, like_img.header
    out = np.zeros(np.prod(shape), dtype=dtype)
    out[mask_bool.reshape(-1)] = vec_gm
    nib.save(nib.Nifti1Image(out.reshape(shape), aff, hdr), out_path)

def wilcoxon_z_map(deltas_subj_by_voxel):  # Nsub × Vgm
    N, V = deltas_subj_by_voxel.shape
    z = np.zeros(V, dtype=np.float32)
    med = np.median(deltas_subj_by_voxel, axis=0)
    for v in range(V):
        x = deltas_subj_by_voxel[:, v]
        nz = x[x != 0]
        if nz.size < 2:
            z[v] = 0.0
            continue
        try:
            w = wilcoxon(nz, zero_method="wilcox", correction=False, alternative="two-sided", mode="auto")
            p = max(min(w.pvalue, 1.0), 1e-300)
            z_unsigned = norm.isf(p / 2.0)
            z[v] = z_unsigned * np.sign(med[v])   # sign by median Δ
        except Exception:
            z[v] = 0.0
    return z

# ---------------- main ---------------- #
def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load meta & filter to ADULTS + the chosen RUN
    meta = pd.read_csv(args.meta_csv)
    required = {"subject", "age_group", "run", "item_id", "triplet_id", "beta_path"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"meta_csv missing required columns: {sorted(missing)}")

    adults = meta[(meta["age_group"] == "adult") & (meta["run"] == args.run)].copy()
    if adults.empty:
        raise ValueError("No adult rows for the requested run in meta_csv.")

    # Masks
    gm_img, gm_bool = load_mask(args.gm_mask)
    gm_idx = gm_indices(gm_bool)

    if args.roi_mask:
        roi_img, roi_bool = load_mask(args.roi_mask)
        if roi_img.shape != gm_img.shape:
            raise ValueError("ROI mask shape does not match GM mask shape.")
        mask_bool = roi_bool & gm_bool
        mask_name = os.path.splitext(os.path.basename(args.roi_mask))[0]
    else:
        mask_bool = gm_bool.copy()
        mask_name = "GM"

    # Compute per-adult Δ maps (GM only) and stack
    subjects = sorted(adults["subject"].unique())
    delta_list = []
    for sid in subjects:
        rows = adults[adults["subject"] == sid].sort_values("item_id")
        if rows.shape[0] == 0:
            continue
        X = load_item_stack(rows, gm_idx)              # 12 × Vgm
        trip = rows["triplet_id"].to_numpy()
        delta = compute_delta(X, trip, zscore_items=args.zscore_items)  # (Vgm,)
        delta_list.append(delta)

    if len(delta_list) == 0:
        raise RuntimeError("No Δ maps computed; check betas/labels in meta.")

    deltas = np.vstack(delta_list)  # Nsub × Vgm

    # Wilcoxon z-map across adults
    z_vec = wilcoxon_z_map(deltas).astype(np.float32)

    # Save group z-map
    z_map_path = os.path.join(args.outdir, f"group_adults_run-{args.run}_wilcoxon_z_triplet_delta.nii.gz")
    save_full_like(z_vec, gm_bool, gm_img, z_map_path)
    print(f"[Group z-map] {z_map_path}")

    # Threshold: top X% within mask
    z_full = nib.load(z_map_path).get_fdata().astype(np.float32)
    in_mask = mask_bool

    top_pct = float(args.top_percent)
    if not (0.0 < top_pct < 100.0):
        raise ValueError("--top_percent must be in (0, 100).")

    if args.bidirectional:
        # rank by |z| within mask
        vals = np.abs(z_full[in_mask])
        if vals.size == 0:
            final_mask = np.zeros_like(z_full, dtype=np.uint8); thr = 0.0
        else:
            cutoff = np.percentile(vals, 100.0 - top_pct)
            final_mask = ((np.abs(z_full) >= cutoff) & in_mask).astype(np.uint8)
            thr = float(cutoff)
        mode_tag = "absz"
    else:
        # positive direction only
        vals = z_full[in_mask & (z_full > 0)]
        if vals.size == 0:
            final_mask = np.zeros_like(z_full, dtype=np.uint8); thr = 0.0
        else:
            cutoff = np.percentile(vals, 100.0 - top_pct)
            final_mask = ((z_full >= cutoff) & in_mask & (z_full > 0)).astype(np.uint8)
            thr = float(cutoff)
        mode_tag = "posz"

    # Save final mask
    mask_path = os.path.join(
        args.outdir,
        f"group_adults_run-{args.run}_top{args.top_percent:.2f}pct_{mask_name}_{mode_tag}_wilcoxon_z_triplet_mask.nii.gz"
    )
    nib.save(nib.Nifti1Image(final_mask, gm_img.affine, gm_img.header), mask_path)

    kept = int(final_mask.sum())
    print(f"[Top-{args.top_percent:.2f}% {mode_tag}] kept={kept}  cutoff {'|z|' if args.bidirectional else 'z'}={thr:.4f}  → {mask_path}")

    # Summary
    with open(os.path.join(args.outdir, "summary_group.txt"), "w") as f:
        f.write(f"N adults: {len(subjects)}\n")
        f.write(f"run: {args.run}\n")
        f.write(f"mask: {mask_name}\n")
        f.write(f"aggregation: Wilcoxon z (signed by median Δ)\n")
        f.write(f"selection mode: {'|z| (bidirectional)' if args.bidirectional else 'positive z only'}\n")
        f.write(f"top_percent: {args.top_percent:.2f}\n")
        f.write(f"kept voxels: {kept}\n")
        f.write(f"cutoff {'|z|' if args.bidirectional else 'z'}: {thr:.4f}\n")
        f.write(f"zscore_items: {bool(args.zscore_items)}\n")

if __name__ == "__main__":
    main()
