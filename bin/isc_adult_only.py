#!/usr/bin/env python3
"""
isc_adult_only.py
-----------------
Compute voxelwise ISC maps for adults using a leave-one-out approach.
Each adult is compared to the mean of all other adults.

Outputs:
  - <sub>_movie_coin_iscAdultLOO_z.nii.gz
  - <sub>_movie_jinx_iscAdultLOO_z.nii.gz
  - <sub>_iscAdultLOO_z_merged.nii.gz
"""

from pathlib import Path
import argparse, sys
import numpy as np
import nibabel as nib
from temple_utils.get_age_groups import get_adults


def label_movie(T: int) -> str:
    return "movie_coin" if T > 170 else "movie_jinx"


def load_TxV(img_path: Path, mask_bool: np.ndarray):
    """Load NIfTI, mask, and z-score each voxel timecourse over time."""
    img = nib.load(str(img_path))
    data = img.get_fdata()  # X,Y,Z,T
    vox = data[mask_bool, :].astype(np.float32).T  # T×V
    vox -= vox.mean(0, keepdims=True)
    vox /= (vox.std(0, keepdims=True) + 1e-8)
    return vox, img


def corr_time_zscored(x_TV: np.ndarray, y_TV: np.ndarray):
    T = x_TV.shape[0]
    return (x_TV * y_TV).sum(0) / max(T - 1, 1)


def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r).astype(np.float32)


def trim_to_common_T(arrs):
    Tmin = min(a.shape[0] for a in arrs)
    if len(set(a.shape[0] for a in arrs)) > 1:
        arrs = [a[-Tmin:, :] for a in arrs]
    return arrs


def compute_isc_adult_loo(
    subject_files_by_bin,
    adult_files_by_bin,
    bins,
    runs_by_bin,
    mask_bool,
    subject_id,
    out_dir,
    aff,
    shape3d,
):
    """Compute leave-one-out ISC maps for one adult."""
    per_bin = {}

    for bin_key in bins:
        if bin_key not in runs_by_bin:
            continue

        z_maps = []

        for r in runs_by_bin[bin_key]:
            if bin_key not in subject_files_by_bin or r not in subject_files_by_bin[bin_key]:
                continue

            # reference = all other adults
            ref_files = [
                p for p in adult_files_by_bin.get(bin_key, {}).get(r, [])
                if p != subject_files_by_bin[bin_key][r]
            ]
            if not ref_files:
                continue

            Xs, _ = load_TxV(subject_files_by_bin[bin_key][r], mask_bool)
            ref_TVs = [load_TxV(p, mask_bool)[0] for p in ref_files]

            arrs = [Xs] + ref_TVs
            arrs = trim_to_common_T(arrs)
            Xs = arrs[0]
            ref_stack = np.stack(arrs[1:], axis=0)  # A × T × V

            ref_mean = ref_stack.mean(axis=0)
            ref_mean -= ref_mean.mean(0, keepdims=True)
            ref_mean /= (ref_mean.std(0, keepdims=True) + 1e-8)

            rvec = corr_time_zscored(Xs, ref_mean)
            z_maps.append(fisher_z(rvec))

        if not z_maps:
            continue

        z_mean = np.mean(np.stack(z_maps, axis=0), axis=0)
        per_bin[bin_key] = z_mean

        vol = np.zeros(mask_bool.size, dtype=np.float32)
        vol[mask_bool.ravel()] = z_mean
        nib.save(
            nib.Nifti1Image(vol.reshape(shape3d), aff),
            out_dir / f"{subject_id}_{bin_key}_iscAdultLOO_z.nii.gz",
        )

    return per_bin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prep_dir", help="Root with sub-*/ prepped NIfTIs")
    ap.add_argument("mask", help="MNI mask NIfTI")
    ap.add_argument("out_dir", help="Output directory")
    ap.add_argument("--only_sub", default=None, help="Process only this subject")
    args = ap.parse_args()

    prep_dir = Path(args.prep_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sub_dirs = sorted([p for p in prep_dir.glob("sub-*") if p.is_dir()])
    subs = [p.name for p in sub_dirs]

    to_sub_fmt = lambda s: s if s.startswith("sub-") else f"sub-{s}"
    adults_all = set(map(to_sub_fmt, get_adults()))

    mask_img = nib.load(args.mask)
    mask_bool = mask_img.get_fdata().astype(bool)
    aff, shape3d = mask_img.affine, mask_img.shape

    # gather files
    files_by_sub = {}
    for s, sdir in zip(subs, sub_dirs):
        runs_dict = {}
        for run in [1, 2]:
            fpath = sdir / f"{s}_run-{run}_MNI_movie_ISC_prepped.nii.gz"
            if fpath.exists():
                T = nib.load(str(fpath)).shape[-1]
                bin_key = label_movie(T)
                runs_dict.setdefault(bin_key, {})[str(run)] = fpath
        if runs_dict:
            files_by_sub[s] = runs_dict

    bin_keys = ["movie_coin", "movie_jinx"]

    # build adult file table
    adult_files = {bk: {} for bk in bin_keys}
    for s, bins in files_by_sub.items():
        if s in adults_all:
            for bk, runs in bins.items():
                for r, p in runs.items():
                    adult_files[bk].setdefault(r, []).append(p)

    runs_by_bin = {bk: sorted(adult_files[bk].keys(), key=int) for bk in bin_keys}

    loop_subs = subs
    if args.only_sub:
        only_sub = to_sub_fmt(args.only_sub)
        if only_sub not in files_by_sub:
            print(f"[ERROR] --only_sub {only_sub} not found", file=sys.stderr)
            sys.exit(1)
        loop_subs = [only_sub]

    per_bin_adult = {}
    for s in loop_subs:
        if s not in adults_all or s not in files_by_sub:
            continue

        per_bin_adult[s] = compute_isc_adult_loo(
            subject_files_by_bin=files_by_sub[s],
            adult_files_by_bin=adult_files,
            bins=bin_keys,
            runs_by_bin=runs_by_bin,
            mask_bool=mask_bool,
            subject_id=s,
            out_dir=out_dir,
            aff=aff,
            shape3d=shape3d,
        )

    # merged across movies
    for s in per_bin_adult:
        z_coin = per_bin_adult[s].get("movie_coin")
        z_jinx = per_bin_adult[s].get("movie_jinx")
        if z_coin is None or z_jinx is None:
            continue

        z_mean = np.mean(np.stack([z_coin, z_jinx], axis=0), axis=0)
        vol = np.zeros(mask_bool.size, dtype=np.float32)
        vol[mask_bool.ravel()] = z_mean
        nib.save(
            nib.Nifti1Image(vol.reshape(shape3d), aff),
            out_dir / f"{s}_iscAdultLOO_z_merged.nii.gz",
        )

    print("\nDone. Wrote leave-one-out ISC maps for adults.")


if __name__ == "__main__":
    main()
