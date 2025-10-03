#!/usr/bin/env python
"""
Triplet-selective Top-K voxel masks via iterative thresholding (no Wilcoxon).

For each subject listed in meta_csv (and the specified run):
  1) Build Δ(v) = mean_across |β_i - β_j| - mean_within |β_i - β_j| from 12 item betas
  2) Inside ROI (default GM), iteratively threshold Δ to obtain ~Top-K voxels
     - Seed from Pth percentile of positive Δ in ROI
     - Raise/lower threshold; halve step on direction flips
     - Stop on exact hit or at max iterations; otherwise use closest

Outputs per subject:
  - sub-<ID>_run-<R>_triplet_delta.nii.gz
  - sub-<ID>_run-<R>_top<K>_<ROI>_triplet_mask.nii.gz
  - summary.txt
"""

import numpy as np
import pandas as pd
import nibabel as nib
import argparse, os


# ---------------- args ---------------- #
def get_args():
    p = argparse.ArgumentParser(
        description="Compute Δ maps and threshold to Top-K triplet-selective voxels (ROI default = GM)."
    )
    p.add_argument("meta_csv",
                   help="CSV with columns: subject,run,item_id,triplet_id,beta_path (age/age_group ok too)")
    p.add_argument("gm_mask", help="MNI gray-matter mask (binary NIfTI)")
    p.add_argument("outdir",  help="Output directory root")
    p.add_argument("--run", type=int, default=4, help="Run to use (default: 4)")
    p.add_argument("--size", type=int, default=1000, help="Top-K voxels to select (default: 1000)")
    p.add_argument("--roi_mask", default=None,
                   help="Optional ROI mask NIfTI. If omitted, GM is used as ROI.")
    p.add_argument("--pthresh", type=float, default=99.75,
                   help="Percentile to seed Δ threshold from positive Δ within ROI (default 99.75)")
    p.add_argument("--max_iter", type=int, default=50,
                   help="Max iterations for threshold search (default 50)")
    p.add_argument("--init_increment", type=float, default=None,
                   help="Initial step size for threshold updates; if None, auto-derived from Δ distribution.")
    p.add_argument("--zscore_items", action="store_true",
                   help="Z-score β across items per voxel before pairwise diffs (optional).")
    return p.parse_args()

# ---------------- helpers ---------------- #
def load_mask(path):
    img = nib.load(path)
    data = img.get_fdata().astype(bool)
    return img, data

def gm_indices(gm_bool):
    flat = gm_bool.reshape(-1)
    return np.where(flat)[0]

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

    n_items = X.shape[0]
    I, J = np.triu_indices(n_items, k=1)          # 66 pairs
    D = np.abs(X[I, :] - X[J, :])                 # 66 x Vgm
    within = (triplets[I] == triplets[J])         # 12 within
    m_within = D[within, :].mean(axis=0)
    m_across = D[~within, :].mean(axis=0)         # 54 across
    return (m_across - m_within).astype(np.float32)  # positive => within more similar

def save_full_like_gm(vec_gm, gm_bool, like_img, out_path, dtype=np.float32):
    shape, aff, hdr = like_img.shape, like_img.affine, like_img.header
    out = np.zeros(np.prod(shape), dtype=dtype)
    out[gm_bool.reshape(-1)] = vec_gm
    nib.save(nib.Nifti1Image(out.reshape(shape), aff, hdr), out_path)

def iterative_threshold(delta_full, roi_bool, size, pthresh=99.75,
                        max_iter=50, init_increment=None):
    """
    Threshold search mirroring the bash logic:
      - Seed from Pth percentile of positive Δ in ROI
      - Raise/lower threshold; halve step when crossing over/under target
      - Stop on exact hit or after max_iter; return closest solution otherwise
    Returns: (final_mask_uint8, chosen_threshold)
    """
    # positive Δ inside ROI
    pos = delta_full[(delta_full > 0) & roi_bool]
    if pos.size == 0:
        # fallback: pick top-K by value within ROI
        vals = delta_full[roi_bool]
        if vals.size == 0:
            return np.zeros_like(roi_bool, dtype=np.uint8), 0.0
        order = np.argsort(vals)[::-1]
        k = min(size, order.size)
        thr = vals[order[k-1]] if k > 0 else np.inf
        mask = roi_bool & (delta_full >= thr)
        return mask.astype(np.uint8), float(thr)

    # seed threshold
    zthr = float(np.percentile(pos, pthresh))
    # pick initial increment
    if init_increment is None:
        p95, p50 = np.percentile(pos, 95.0), np.percentile(pos, 50.0)
        inc = max((p95 - p50) * 0.1, 1e-6)
    else:
        inc = float(init_increment)

    relpos = 0   # 0=unset, 1=over, 2=under
    mindiff = float("inf")
    best_thr = zthr
    best_mask = None

    for it in range(1, max_iter + 1):
        cand = roi_bool & (delta_full >= zthr)
        nvox = int(cand.sum())

        # track best
        diff = abs(nvox - size)
        if diff < mindiff:
            mindiff = diff
            best_thr = zthr
            best_mask = cand.copy()

        if nvox == size:
            return cand.astype(np.uint8), zthr

        if nvox > size:
            # too many voxels => raise threshold
            if relpos == 2:
                inc *= 0.5
            zthr += inc
            relpos = 1
        else:
            # too few voxels => lower threshold
            if relpos == 1:
                inc *= 0.5
            zthr -= inc
            relpos = 2

    # use closest after max_iter
    return best_mask.astype(np.uint8), best_thr

# ---------------- main ---------------- #
def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load meta & masks
    meta = pd.read_csv(args.meta_csv)
    if "subject" not in meta.columns or "run" not in meta.columns:
        raise ValueError("meta_csv must include at least columns: subject, run, item_id, triplet_id, beta_path")
    gm_img, gm_bool = load_mask(args.gm_mask)
    gm_idx = gm_indices(gm_bool)

    # ROI
    if args.roi_mask is not None:
        roi_img, roi_bool = load_mask(args.roi_mask)
        if roi_img.shape != gm_img.shape:
            raise ValueError("ROI mask shape does not match GM mask shape.")
        roi_name = os.path.splitext(os.path.basename(args.roi_mask))[0]
        roi_bool = roi_bool & gm_bool  # confine to GM
    else:
        roi_bool = gm_bool.copy()
        roi_name = "GM"

    # Subjects from meta
    subjects = sorted(meta["subject"].unique())

    for sid in subjects:
        subrows = meta[(meta["subject"] == sid) & (meta["run"] == args.run)].copy()
        if subrows.shape[0] == 0:
            print(f"[WARN] No rows for subject={sid}, run={args.run}. Skipping.")
            continue

        # Load 12 betas (sorted by item) and compute Δ
        subrows = subrows.sort_values("item_id")
        X = load_item_stack(subrows, gm_idx)            # 12 x Vgm
        trips = subrows["triplet_id"].to_numpy()
        delta_gm = compute_delta(X, trips, zscore_items=args.zscore_items)

        # Paths
        subdir = os.path.join(args.outdir, f"sub-{sid}", "triplet_topk", roi_name)
        os.makedirs(subdir, exist_ok=True)
        delta_path = os.path.join(subdir, f"sub-{sid}_run-{args.run}_triplet_delta.nii.gz")

        # Save Δ map
        save_full_like_gm(delta_gm, gm_bool, gm_img, delta_path)
        print(f"[Δ-map] {delta_path}")

        # Iterative thresholding inside ROI to get Top-K
        delta_full = nib.load(delta_path).get_fdata().astype(np.float32)
        mask_uint8, thr = iterative_threshold(
            delta_full=delta_full,
            roi_bool=roi_bool,
            size=args.size,
            pthresh=args.pthresh,
            max_iter=args.max_iter,
            init_increment=args.init_increment
        )

        # Save final mask
        topk_path = os.path.join(subdir, f"sub-{sid}_run-{args.run}_top{args.size}_{roi_name}_triplet_mask.nii.gz")
        nib.save(nib.Nifti1Image(mask_uint8.astype(np.uint8), gm_img.affine, gm_img.header), topk_path)

        kept = int(mask_uint8.sum())
        print(f"[Top-K] subject={sid} run={args.run} ROI={roi_name}  target={args.size} kept={kept} thr={thr:.6g}")
        with open(os.path.join(subdir, "summary.txt"), "w") as f:
            f.write(f"subject: {sid}\n")
            f.write(f"run: {args.run}\n")
            f.write(f"ROI: {roi_name}\n")
            f.write(f"Top-K target: {args.size}\n")
            f.write(f"kept voxels: {kept}\n")
            f.write(f"seed percentile (positive Δ): {args.pthresh}\n")
            f.write(f"final threshold: {thr:.6g}\n")
            f.write(f"zscore_items: {bool(args.zscore_items)}\n")
            f.write(f"max_iter: {args.max_iter}\n")
            f.write(f"init_increment: {args.init_increment if args.init_increment is not None else 'auto'}\n")

if __name__ == "__main__":
    main()
