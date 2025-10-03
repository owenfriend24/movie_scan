#!/usr/bin/env python
"""
Group-level Top-K triplet-selective mask (adults only) via iterative thresholding.

Per adult subject (run=R):
  Δ(v) = mean_across |β_i - β_j| - mean_within |β_i - β_j|
Then aggregate across adults into a group map (mean/median/percent positive),
and iteratively threshold within ROI (default GM) to obtain ~Top-K voxels:
  - seed from Pth percentile of positive group values in ROI
  - raise/lower threshold; halve step on direction flips
  - stop on exact hit or at max iterations; else use closest

Outputs:
  - group_adults_run-<R>_<STAT>_triplet_delta.nii.gz
  - group_adults_run-<R>_top<K>_<ROI>_<STAT>_triplet_mask.nii.gz
  - summary_group.txt
"""

import numpy as np
import pandas as pd
import nibabel as nib
import argparse, os

# ---------------- args ---------------- #
def get_args():
    p = argparse.ArgumentParser(
        description="Build group-level (adults) Top-K triplet-selective mask via iterative thresholding (no Wilcoxon)."
    )
    p.add_argument("meta_csv",
                   help="CSV with at least: subject,age_group,run,item_id,triplet_id,beta_path")
    p.add_argument("gm_mask", help="MNI gray-matter mask (binary NIfTI)")
    p.add_argument("outdir",  help="Output directory root")

    p.add_argument("--run", type=int, default=4, help="Run to use (default: 4)")
    p.add_argument("--topk", type=int, default=1000, help="Target # voxels (default: 1000)")
    p.add_argument("--roi_mask", default=None,
                   help="Optional ROI mask NIfTI (if omitted, GM is used as ROI)")
    p.add_argument("--pthresh", type=float, default=99.75,
                   help="Percentile to seed threshold from positive group values in ROI")
    p.add_argument("--max_iter", type=int, default=50,
                   help="Max iterations for threshold search (default 50)")
    p.add_argument("--init_increment", type=float, default=None,
                   help="Initial step size; if None, auto-derived from group value distribution")
    p.add_argument("--zscore_items", action="store_true",
                   help="Z-score β across items per voxel before pairwise diffs (optional)")
    p.add_argument("--stat", choices=["mean", "median", "percent_pos"], default="mean",
                   help="How to aggregate adult Δ maps into group map (default: mean)")
    p.add_argument("--percent_pos_cut", type=float, default=0.0,
                   help="For percent_pos, count a subject voxel as 'positive' if Δ > this cut (default 0)")
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

def save_full_like_gm(vec_gm, gm_bool, like_img, out_path, dtype=np.float32):
    shape, aff, hdr = like_img.shape, like_img.affine, like_img.header
    out = np.zeros(np.prod(shape), dtype=dtype)
    out[gm_bool.reshape(-1)] = vec_gm
    nib.save(nib.Nifti1Image(out.reshape(shape), aff, hdr), out_path)

def iterative_threshold(group_full, roi_bool, topk, pthresh=99.75,
                        max_iter=50, init_increment=None):
    """
    Bash-like threshold search on a single group map (positive direction):
      - Seed from Pth percentile of positive values in ROI
      - Raise/lower threshold; halve step when crossing over/under target
      - Stop on exact hit or after max_iter; use closest solution otherwise
    Returns: (final_mask_uint8, chosen_threshold)
    """
    pos = group_full[(group_full > 0) & roi_bool]
    if pos.size == 0:
        vals = group_full[roi_bool]
        if vals.size == 0:
            return np.zeros_like(roi_bool, dtype=np.uint8), 0.0
        order = np.argsort(vals)[::-1]
        k = min(topk, order.size)
        thr = vals[order[k-1]] if k > 0 else np.inf
        mask = roi_bool & (group_full >= thr)
        return mask.astype(np.uint8), float(thr)

    thr = float(np.percentile(pos, pthresh))
    if init_increment is None:
        p95, p50 = np.percentile(pos, 95.0), np.percentile(pos, 50.0)
        inc = max((p95 - p50) * 0.1, 1e-6)
    else:
        inc = float(init_increment)

    relpos = 0   # 0=unset, 1=over, 2=under
    mindiff = float("inf")
    best_thr = thr
    best_mask = None

    for _ in range(1, max_iter + 1):
        cand = roi_bool & (group_full >= thr)
        nvox = int(cand.sum())

        diff = abs(nvox - topk)
        if diff < mindiff:
            mindiff = diff
            best_thr = thr
            best_mask = cand.copy()

        if nvox == topk:
            return cand.astype(np.uint8), thr

        if nvox > topk:
            if relpos == 2:
                inc *= 0.5
            thr += inc
            relpos = 1
        else:
            if relpos == 1:
                inc *= 0.5
            thr -= inc
            relpos = 2

    return best_mask.astype(np.uint8), best_thr

# ---------------- main ---------------- #
def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load meta & filter adults + run
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
        roi_bool = roi_bool & gm_bool
        roi_name = os.path.splitext(os.path.basename(args.roi_mask))[0]
    else:
        roi_bool = gm_bool.copy()
        roi_name = "GM"

    # Compute Δ maps for each adult
    subjects = sorted(adults["subject"].unique())
    delta_list = []
    for sid in subjects:
        rows = adults[adults["subject"] == sid].sort_values("item_id")
        if rows.empty:
            continue
        X = load_item_stack(rows, gm_idx)         # 12 x Vgm
        trip = rows["triplet_id"].to_numpy()
        delta = compute_delta(X, trip, zscore_items=args.zscore_items)  # Vgm
        delta_list.append(delta)

    if len(delta_list) == 0:
        raise RuntimeError("No Δ maps computed; check meta paths and filters.")

    deltas = np.vstack(delta_list)  # Nsub x Vgm

    # Aggregate to group map
    if args.stat == "mean":
        group_vec = deltas.mean(axis=0).astype(np.float32)
        stat_tag = "mean"
    elif args.stat == "median":
        group_vec = np.median(deltas, axis=0).astype(np.float32)
        stat_tag = "median"
    else:  # percent_pos
        cut = float(args.percent_pos_cut)
        group_vec = (deltas > cut).mean(axis=0).astype(np.float32)  # fraction of adults positive
        stat_tag = f"pctpos_gt{cut:g}"

    # Save group map
    group_map_path = os.path.join(
        args.outdir, f"group_adults_run-{args.run}_{stat_tag}_triplet_delta.nii.gz"
    )
    save_full_like_gm(group_vec, gm_bool, gm_img, group_map_path)
    print(f"[Group map] {group_map_path}")

    # Iteratively threshold group map to Top-K inside ROI
    group_full = nib.load(group_map_path).get_fdata().astype(np.float32)
    mask_uint8, thr = iterative_threshold(
        group_full=group_full,
        roi_bool=roi_bool,
        topk=args.topk,
        pthresh=args.pthresh,
        max_iter=args.max_iter,
        init_increment=args.init_increment
    )

    # Save Top-K mask
    topk_path = os.path.join(
        args.outdir, f"group_adults_run-{args.run}_top{args.topk}_{roi_name}_{stat_tag}_triplet_mask.nii.gz"
    )
    nib.save(nib.Nifti1Image(mask_uint8.astype(np.uint8), gm_img.affine, gm_img.header), topk_path)
    kept = int(mask_uint8.sum())
    print(f"[Top-K] target={args.topk} kept={kept} thr={thr:.6g} → {topk_path}")

    # Summary
    with open(os.path.join(args.outdir, "summary_group.txt"), "w") as f:
        f.write(f"subjects (adults): {len(subjects)}\n")
        f.write(f"run: {args.run}\n")
        f.write(f"roi: {roi_name}\n")
        f.write(f"aggregation: {stat_tag}\n")
        f.write(f"topk target: {args.topk}\n")
        f.write(f"kept voxels: {kept}\n")
        f.write(f"seed percentile: {args.pthresh}\n")
        f.write(f"final threshold: {thr:.6g}\n")
        f.write(f"zscore_items: {bool(args.zscore_items)}\n")
        f.write(f"max_iter: {args.max_iter}\n")
        f.write(f"init_increment: {args.init_increment if args.init_increment is not None else 'auto'}\n")

if __name__ == "__main__":
    main()
