#!/usr/bin/env python3
"""
isc_to_adult_group.py
---------------------
Compute voxelwise ISC maps for children and adolescents relative to the adult group.
Outputs:
  - <sub>_movie_coin_iscToAdult_z.nii.gz
  - <sub>_movie_jinx_iscToAdult_z.nii.gz
  - <sub>_iscToAdult_z_merged.nii.gz
"""

from pathlib import Path
import argparse, sys
import numpy as np
import nibabel as nib
from temple_utils.get_age_groups import get_adults, get_children, get_adolescents

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

def compute_isc_to_adult(subject_files_by_bin, ref_adult_files_by_bin, bins, runs_by_bin, mask_bool, subject_id, out_dir, aff, shape3d):
    """Compute ISC maps for one subject vs adult group."""
    per_bin = {}
    for bin_key in bins:
        if bin_key not in runs_by_bin:
            continue
        z_maps = []
        for r in runs_by_bin[bin_key]:
            if bin_key not in subject_files_by_bin or r not in subject_files_by_bin[bin_key]:
                continue
            ref_files_r = ref_adult_files_by_bin.get(bin_key, {}).get(r, [])
            if not ref_files_r:
                continue

            Xs, _ = load_TxV(subject_files_by_bin[bin_key][r], mask_bool)
            ref_TVs = [load_TxV(p, mask_bool)[0] for p in ref_files_r]
            arrs = [Xs] + ref_TVs
            arrs = trim_to_common_T(arrs)
            Xs = arrs[0]
            ref_stack = np.stack(arrs[1:], axis=0)  # A x T x V

            # mean adult timecourse
            ref_mean = ref_stack.mean(axis=0)
            ref_mean -= ref_mean.mean(0, keepdims=True)
            ref_mean /= (ref_mean.std(0, keepdims=True) + 1e-8)

            rvec = corr_time_zscored(Xs, ref_mean)
            z_maps.append(fisher_z(rvec))

        if not z_maps:
            continue
        z_mean = np.mean(np.stack(z_maps, axis=0), axis=0)
        per_bin[bin_key] = z_mean

        # write per-bin NIfTI
        vol = np.zeros(mask_bool.size, dtype=np.float32)
        vol[mask_bool.ravel()] = z_mean
        nib.save(nib.Nifti1Image(vol.reshape(shape3d), aff), out_dir / f"{subject_id}_{bin_key}_iscToAdult_z.nii.gz")
    return per_bin

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prep_dir", help="Root with sub-*/ prepped NIfTIs")
    ap.add_argument("mask", help="MNI mask NIfTI (GM or brain)")
    ap.add_argument("out_dir", help="Output directory")
    ap.add_argument("--only_sub", default=None, help="Process only this subject")
    args = ap.parse_args()

    prep_dir = Path(args.prep_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # discover subjects
    sub_dirs = sorted([p for p in prep_dir.glob("sub-*") if p.is_dir()])
    subs = [p.name for p in sub_dirs]

    to_sub_fmt = lambda s: s if s.startswith("sub-") else f"sub-{s}"
    adults_all = set(map(to_sub_fmt, get_adults()))
    children_all = set(map(to_sub_fmt, get_children()))
    adolescents_all = set(map(to_sub_fmt, get_adolescents()))

    mask_img = nib.load(args.mask)
    mask_bool = mask_img.get_fdata().astype(bool)
    aff, shape3d = mask_img.affine, mask_img.shape

    # gather files
    files_by_sub = {}
    for s, sdir in zip(subs, sub_dirs):
        runs_dict = {}
        for run in [1,2]:
            fpath = sdir / f"{s}_run-{run}_MNI_movie_ISC_prepped.nii.gz"
            if fpath.exists():
                T = nib.load(str(fpath)).shape[-1]
                bin_key = label_movie(T)
                runs_dict.setdefault(bin_key, {})[str(run)] = fpath
        if runs_dict:
            files_by_sub[s] = runs_dict

    bin_keys = ["movie_coin", "movie_jinx"]

    # build adult reference table
    ref_adult = {bk:{} for bk in bin_keys}
    for s, bins in files_by_sub.items():
        if s in adults_all:
            for bk, runs in bins.items():
                for r, p in runs.items():
                    ref_adult[bk].setdefault(r, []).append(p)

    runs_by_bin = {bk: sorted(ref_adult[bk].keys(), key=int) for bk in bin_keys}

    # restrict to single subject if requested
    loop_subs = subs
    if args.only_sub:
        only_sub = to_sub_fmt(args.only_sub)
        if only_sub not in files_by_sub:
            print(f"[ERROR] --only_sub {only_sub} not found", file=sys.stderr)
            sys.exit(1)
        loop_subs = [only_sub]

    # compute ISC: only children/adolescents
    per_bin_adult = {}
    for s in loop_subs:
        if s not in files_by_sub:
            continue
        if s in adults_all:
            continue  # skip adults
        per_bin_adult[s] = compute_isc_to_adult(
            subject_files_by_bin=files_by_sub[s],
            ref_adult_files_by_bin=ref_adult,
            bins=bin_keys,
            runs_by_bin=runs_by_bin,
            mask_bool=mask_bool,
            subject_id=s,
            out_dir=out_dir,
            aff=aff,
            shape3d=shape3d
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
        nib.save(nib.Nifti1Image(vol.reshape(shape3d), aff), out_dir / f"{s}_iscToAdult_z_merged.nii.gz")

    print("\nDone. Wrote ISC maps for children and adolescents to adult reference.")

if __name__ == "__main__":
    main()
