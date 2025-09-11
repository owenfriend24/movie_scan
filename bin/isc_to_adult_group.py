#!/usr/bin/env python3
"""
isc_to_adult_group.py
---------------------
Compute voxelwise ISC-to-adult maps from prepped MNI NIfTIs.

Expected input files (per subject):
  {prep_dir}/sub-<ID>/func/sub-<ID>_run-<N>_MNI_movie_ISC_prepped.nii.gz

Movie identity:
  - T > 170  -> movie_coin
  - T < 170  -> movie_jinx

Adults are determined by temple_utils.get_age_groups.get_adults().

Outputs (in --out_dir):
  sub-<ID>_movie-<bin>_iscToAdult_z.nii.gz
  <bin>_iscToAdult_z_4D.nii.gz
  [optional] sub-<ID>_iscToAdult_z_meanAcrossMovies.nii.gz
  [optional] iscToAdult_z_meanAcrossMovies_4D.nii.gz
"""

from pathlib import Path
import argparse, sys, subprocess
import numpy as np
import nibabel as nib

# Your adult selector
from temple_utils.get_age_groups import get_adults

def label_movie(T: int) -> str:
    return "movie_coin" if T > 170 else "movie_jinx"

def load_TxV(img_path: Path, mask_bool: np.ndarray):
    img = nib.load(str(img_path))
    data = img.get_fdata()  # X,Y,Z,T
    T = data.shape[-1]
    vox = data[mask_bool, :].astype(np.float32).T  # T×V
    # z-score over time per voxel
    vox -= vox.mean(0, keepdims=True)
    vox /= (vox.std(0, keepdims=True) + 1e-8)
    return vox, img, T

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

def norm_sub_id(x: str) -> str:
    return x[4:] if x.startswith("sub-") else x

def gather_runs(prep_dir: Path, sub: str):
    """
    For a given subject ID (e.g. temple056),
    return a dict of {bin_key: {run: Path}} for existing runs.
    """
    func_dir = prep_dir / f"sub-{sub}" / "func"
    runs_dict = {}
    for run in [1, 2]:  # if you only ever have 1 or 2 runs
        fpath = func_dir / f"sub-{sub}_run-{run}_MNI_movie_ISC_prepped.nii.gz"
        if fpath.exists():
            img = nib.load(str(fpath))
            T = img.shape[-1]
            bin_key = "movie_coin" if T > 170 else "movie_jinx"
            runs_dict.setdefault(bin_key, {})[str(run)] = fpath
    return runs_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prep_dir", help="Root with sub-*/func/ prepped NIfTIs")
    ap.add_argument("mask",  help="MNI mask NIfTI (GM or brain) matching prepped grid")
    ap.add_argument("out_dir", help="Output directory")
    ap.add_argument("--make_grand_average", action="store_true", help="Also average across movie bins per subject")
    args = ap.parse_args()

    prep_dir = Path(args.prep_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Discover subjects and adults
    sub_dirs = sorted([p for p in prep_dir.glob("sub-*") if p.is_dir()])
    subs = [norm_sub_id(p.name) for p in sub_dirs]
    adults_all = {norm_sub_id(s) for s in get_adults()}

    # Mask
    mask_img = nib.load(args.mask)
    mask_bool = mask_img.get_fdata().astype(bool)
    aff, shape3d = mask_img.affine, mask_img.shape

    # Gather files per subject according to your actual naming
    files_by_sub = {}
    for sdir in sorted(prep_dir.glob("sub-*")):
        sub = sdir.name.replace("sub-", "")
        runs = gather_runs(prep_dir, sub)
        if not runs:
            print(f"[WARN] No prepped runs for {sub}")
            continue
        files_by_sub[sub] = runs

    bin_keys = ["movie_coin", "movie_jinx"]
    per_bin_maps = {s:{} for s in subs}

    # Process each bin separately
    for bin_key in bin_keys:
        have_subs = [s for s in subs if s in files_by_sub and bin_key in files_by_sub[s]]
        if not have_subs:
            continue

        # adults intersect those with data in this bin
        adults = [s for s in have_subs if s in adults_all]
        if not adults:
            print(f"[WARN] No adults for {bin_key}; skipping", file=sys.stderr)
            continue

        # available run IDs among adults (e.g., "1","2")
        run_ids = sorted({r for s in adults for r in files_by_sub[s][bin_key].keys()}, key=lambda x: int(x))

        for s in have_subs:
            z_maps = []
            for r in run_ids:
                if r not in files_by_sub[s][bin_key]:
                    continue

                # Subject run
                Xs, _, _ = load_TxV(files_by_sub[s][bin_key][r], mask_bool)

                # Adult refs for the same run id
                adult_files = [files_by_sub[a][bin_key][r] for a in adults if r in files_by_sub[a][bin_key]]
                if not adult_files:
                    continue

                adult_TVs = [load_TxV(apath, mask_bool)[0] for apath in adult_files]
                arrs = [Xs] + adult_TVs
                arrs = trim_to_common_T(arrs)  # if any T differs by a few TRs
                Xs = arrs[0]
                adult_stack = np.stack(arrs[1:], axis=0)  # A×T×V
                A = adult_stack.shape[0]
                adult_ref = adult_stack.mean(axis=0)

                # LOAO if subject is adult in this bin+run
                if s in adults and A > 1:
                    adult_list = [a for a in adults if r in files_by_sub[a][bin_key]]
                    idx = adult_list.index(s) if s in adult_list else -1
                    ref = (adult_ref * A - adult_stack[idx]) / (A - 1) if idx >= 0 else adult_ref
                else:
                    ref = adult_ref

                rvec = corr_time_zscored(Xs, ref)
                z_maps.append(fisher_z(rvec))

            if not z_maps:
                continue

            z_mean = np.mean(np.stack(z_maps, axis=0), axis=0)
            vol = np.zeros(mask_bool.size, dtype=np.float32)
            vol[mask_bool.ravel()] = z_mean
            vol3d = vol.reshape(shape3d)
            nib.save(nib.Nifti1Image(vol3d, aff),
                     out_dir / f"sub-{s}_{bin_key}_iscToAdult_z.nii.gz")
            per_bin_maps[s][bin_key] = z_mean

        # Merge per-bin 4D
        bin_subs = [s for s in subs if bin_key in per_bin_maps.get(s, {})]
        if bin_subs:
            to_merge = [str(out_dir / f"sub-{s}_{bin_key}_iscToAdult_z.nii.gz") for s in bin_subs]
            merged = out_dir / f"{bin_key}_iscToAdult_z_4D.nii.gz"
            try:
                subprocess.run(["fslmerge", "-t", str(merged)] + to_merge, check=True)
            except Exception:
                imgs = [nib.load(p) for p in to_merge]
                data4d = np.stack([im.get_fdata() for im in imgs], axis=-1)
                nib.save(nib.Nifti1Image(data4d, imgs[0].affine, imgs[0].header), str(merged))
            print(f"[OK] Merged {bin_key} → {merged.name} ({len(bin_subs)} subjects)")

    # Optional grand average across movie bins per subject (z-mean)
    if args.make_grand_average:
        ga_subs = []
        for s in subs:
            zlist = [per_bin_maps[s][bk] for bk in bin_keys if bk in per_bin_maps[s]]
            if not zlist:
                continue
            z_mean_all = np.mean(np.stack(zlist, axis=0), axis=0)
            vol = np.zeros(mask_bool.size, dtype=np.float32)
            vol[mask_bool.ravel()] = z_mean_all
            nib.save(nib.Nifti1Image(vol.reshape(shape3d), aff),
                     out_dir / f"sub-{s}_iscToAdult_z_meanAcrossMovies.nii.gz")
            ga_subs.append(s)
        if ga_subs:
            to_merge = [str(out_dir / f"sub-{s}_iscToAdult_z_meanAcrossMovies.nii.gz") for s in ga_subs]
            merged = out_dir / "iscToAdult_z_meanAcrossMovies_4D.nii.gz"
            imgs = [nib.load(p) for p in to_merge]
            data4d = np.stack([im.get_fdata() for im in imgs], axis=-1)
            nib.save(nib.Nifti1Image(data4d, imgs[0].affine, imgs[0].header), str(merged))
            print(f"[OK] Merged grand-average → {merged.name} ({len(ga_subs)} subjects)")

    print("\nDone. Per-bin stacks: movie_coin_iscToAdult_z_4D.nii.gz, movie_jinx_iscToAdult_z_4D.nii.gz")

if __name__ == "__main__":
    main()
