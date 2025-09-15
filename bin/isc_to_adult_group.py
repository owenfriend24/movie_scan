#!/usr/bin/env python3
"""
isc_to_adult_group.py
---------------------
Compute voxelwise ISC maps from prepped MNI NIfTIs:
- ISC to ADULT group (existing behavior)
- NEW: ISC within CHILD group (child→child)
- NEW: ISC within ADOLESCENT group (adolescent→adolescent)

Expected inputs per subject (directly under subject folder):
  {prep_dir}/sub-<ID>/{sub-<ID>}_run-<N>_MNI_movie_ISC_prepped.nii.gz   (N in {1,2})

Movie identity:
  - T > 170  -> movie_coin
  - T < 170  -> movie_jinx

Groups are determined by temple_utils.get_age_groups.

Outputs (in --out_dir):
  # per-movie (bin) maps
  <sub>_movie_coin_iscToAdult_z.nii.gz
  <sub>_movie_jinx_iscToAdult_z.nii.gz
  <sub>_movie_coin_iscToChild_z.nii.gz
  <sub>_movie_jinx_iscToChild_z.nii.gz
  <sub>_movie_coin_iscToAdolescent_z.nii.gz
  <sub>_movie_jinx_iscToAdolescent_z.nii.gz

  # merged across movies (mean of per-movie z-maps; written only if both exist)
  <sub>_iscToAdult_z_merged.nii.gz
  <sub>_iscToChild_z_merged.nii.gz
  <sub>_iscToAdolescent_z_merged.nii.gz
"""

from pathlib import Path
import argparse, sys
import numpy as np
import nibabel as nib

# Group selectors
from temple_utils.get_age_groups import get_adults, get_children, get_adolescents

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

def compute_isc_to_reference(subject_files_by_bin, reference_files_by_bin, bins, runs_by_bin, mask_bool,
                             subject_id, out_dir, aff, shape3d, label, leave_one_out=False):
    """
    Compute ISC maps for one subject vs a reference group for each bin (movie_coin/movie_jinx),
    with optional leave-one-out if the subject belongs to the reference group.
    Writes per-bin NIfTIs named: <sub>_<bin>_iscTo{Label}_z.nii.gz
    Returns: dict bin -> 1D z-map array (masked voxels)
    """
    per_bin = {}
    for bin_key in bins:
        if bin_key not in runs_by_bin:
            continue
        z_maps = []
        # which runs exist for the reference in this bin?
        for r in runs_by_bin[bin_key]:
            # need the subject's run and at least one ref run
            if bin_key not in subject_files_by_bin or r not in subject_files_by_bin[bin_key]:
                continue
            ref_files_r = [p for p in reference_files_by_bin.get(bin_key, {}).get(r, [])]
            if not ref_files_r:
                continue

            # subject data
            Xs, _, _ = load_TxV(subject_files_by_bin[bin_key][r], mask_bool)
            # reference stack
            ref_TVs = [load_TxV(p, mask_bool)[0] for p in ref_files_r]
            arrs = [Xs] + ref_TVs
            arrs = trim_to_common_T(arrs)
            Xs = arrs[0]
            ref_stack = np.stack(arrs[1:], axis=0)  # A×T×V
            A = ref_stack.shape[0]

            # leave-one-out if requested and subject is in reference set for this run
            if leave_one_out:
                # Detect if subject contributed this run to the reference
                subj_in_ref_idx = -1
                for i, p in enumerate(ref_files_r):
                    if subject_id in p.name:
                        subj_in_ref_idx = i
                        break
                if subj_in_ref_idx >= 0 and A > 1:
                    ref_mean = (ref_stack.mean(axis=0) * A - ref_stack[subj_in_ref_idx]) / (A - 1)
                else:
                    ref_mean = ref_stack.mean(axis=0)
            else:
                ref_mean = ref_stack.mean(axis=0)

            rvec = corr_time_zscored(Xs, ref_mean)
            z_maps.append(fisher_z(rvec))

        if not z_maps:
            continue

        z_mean = np.mean(np.stack(z_maps, axis=0), axis=0)
        per_bin[bin_key] = z_mean

        # write per-bin NIfTI
        vol = np.zeros(mask_bool.size, dtype=np.float32)
        vol[mask_bool.ravel()] = z_mean
        vol3d = vol.reshape(shape3d)
        nib.save(nib.Nifti1Image(vol3d, aff),
                 out_dir / f"{subject_id}_{bin_key}_iscTo{label}_z.nii.gz")
    return per_bin

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prep_dir", help="Root with sub-*/ prepped NIfTIs")
    ap.add_argument("mask",  help="MNI mask NIfTI (GM or brain) matching prepped grid")
    ap.add_argument("out_dir", help="Output directory")
    args = ap.parse_args()

    prep_dir = Path(args.prep_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Discover subjects (keep 'sub-...' intact)
    sub_dirs = sorted([p for p in prep_dir.glob("sub-*") if p.is_dir()])
    subs = [p.name for p in sub_dirs]  # e.g., "sub-temple056"

    # Groups (normalize to 'sub-...' format)
    to_sub_fmt = lambda s: s if s.startswith("sub-") else f"sub-{s}"
    adults_all      = set(map(to_sub_fmt, get_adults()))
    children_all    = set(map(to_sub_fmt, get_children()))
    adolescents_all = set(map(to_sub_fmt, get_adolescents()))

    # Mask
    mask_img = nib.load(args.mask)
    mask_bool = mask_img.get_fdata().astype(bool)
    aff, shape3d = mask_img.affine, mask_img.shape

    # Gather files: {sub: {bin: {run: Path}}}
    files_by_sub = {}
    for s, sdir in zip(subs, sub_dirs):
        runs_dict = {}
        for run in [1, 2]:
            fpath = sdir / f"{s}_run-{run}_MNI_movie_ISC_prepped.nii.gz"
            if fpath.exists():
                T = nib.load(str(fpath)).shape[-1]
                bin_key = label_movie(T)  # movie_coin / movie_jinx
                runs_dict.setdefault(bin_key, {})[str(run)] = fpath
        if not runs_dict:
            print(f"[WARN] No prepped runs for {s}", file=sys.stderr)
            continue
        files_by_sub[s] = runs_dict

    bin_keys = ["movie_coin", "movie_jinx"]

    # Build reference file tables per group & bin: {bin: {run: [Paths...]}}
    ref_adult = {bk:{} for bk in bin_keys}
    ref_child = {bk:{} for bk in bin_keys}
    ref_ado   = {bk:{} for bk in bin_keys}

    for s, bins in files_by_sub.items():
        for bk, runs in bins.items():
            for r, p in runs.items():
                if s in adults_all:
                    ref_adult[bk].setdefault(r, []).append(p)
                if s in children_all:
                    ref_child[bk].setdefault(r, []).append(p)
                if s in adolescents_all:
                    ref_ado[bk].setdefault(r, []).append(p)

    # Determine available runs per bin among ADULTS (drives which runs we compute)
    runs_by_bin = {bk: sorted(ref_adult[bk].keys(), key=int) for bk in bin_keys}

    # Compute per-subject ISC maps:
    # 1) to ADULT (LOAO for adult subjects)
    per_bin_adult = {}
    for s in subs:
        if s not in files_by_sub:  # skipped earlier if no runs
            continue
        per_bin_adult[s] = compute_isc_to_reference(
            subject_files_by_bin = files_by_sub[s],
            reference_files_by_bin = ref_adult,
            bins = bin_keys,
            runs_by_bin = runs_by_bin,
            mask_bool = mask_bool,
            subject_id = s,
            out_dir = out_dir,
            aff = aff, shape3d = shape3d,
            label = "Adult",
            leave_one_out = (s in adults_all)
        )

    # 2) within CHILD group (only for child subjects; LOAO within children)
    per_bin_child = {}
    child_runs_by_bin = {bk: sorted(ref_child[bk].keys(), key=int) for bk in bin_keys}
    for s in subs:
        if s not in files_by_sub or s not in children_all:
            continue
        per_bin_child[s] = compute_isc_to_reference(
            files_by_sub[s], ref_child, bin_keys, child_runs_by_bin,
            mask_bool, s, out_dir, aff, shape3d, label="Child", leave_one_out=True
        )

    # 3) within ADOLESCENT group (only for adolescent subjects; LOAO within adolescents)
    per_bin_ado = {}
    ado_runs_by_bin = {bk: sorted(ref_ado[bk].keys(), key=int) for bk in bin_keys}
    for s in subs:
        if s not in files_by_sub or s not in adolescents_all:
            continue
        per_bin_ado[s] = compute_isc_to_reference(
            files_by_sub[s], ref_ado, bin_keys, ado_runs_by_bin,
            mask_bool, s, out_dir, aff, shape3d, label="Adolescent", leave_one_out=True
        )

    # --- Merged across movies (mean of the two per-bin z-maps), per subject ---
    def write_merged(per_bin_dict, label):
        wrote = 0
        for s in subs:
            z_coin = per_bin_dict.get(s, {}).get("movie_coin")
            z_jinx = per_bin_dict.get(s, {}).get("movie_jinx")
            if z_coin is None or z_jinx is None:
                continue
            z_mean = np.mean(np.stack([z_coin, z_jinx], axis=0), axis=0)
            vol = np.zeros(mask_bool.size, dtype=np.float32)
            vol[mask_bool.ravel()] = z_mean
            nib.save(nib.Nifti1Image(vol.reshape(shape3d), aff),
                     out_dir / f"{s}_iscTo{label}_z_merged.nii.gz")
            wrote += 1
        if wrote:
            print(f"[OK] Wrote {wrote} merged maps for iscTo{label}")

    write_merged(per_bin_adult, "Adult")
    write_merged(per_bin_child, "Child")
    write_merged(per_bin_ado,   "Adolescent")

    print("\nDone. Wrote per-movie ISC maps to Adult/Child/Adolescent, and per-subject merged maps (when both movies exist).")

if __name__ == "__main__":
    main()
