#!/usr/bin/env python3
"""
prep_isc_data.py
----------------
Prepares BOLD movie data for ISC by regressing out nuisance confounds
and applying a high-pass filter (128s cutoff).

Inputs:
- fMRIPrep outputs in MNI space for a given subject
- Confound TSVs from fMRIPrep

Outputs:
- Residualized, high-pass filtered BOLD NIfTI(s) per run
"""

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
from nilearn.signal import clean
import os
import subprocess


def movie_to_mni(sub, fmriprep_dir, out_dir):
    fmriprep_dir = Path(fmriprep_dir)
    out_dir = Path(out_dir)

    movie_runs = fmriprep_dir.glob(
        f"sub-{sub}/func/sub-{sub}_task-movie*_4mm.nii.gz"
    )
    run = 1
    for func_run in movie_runs:
        comm = [
            'antsApplyTransforms',
            '-d', '3',
            '-e', '3',
            '-i', str(func_run),
            '-o', str(out_dir / f"sub-{sub}_run-{run}_MNI_movie.nii.gz"),
            '-r', '/home1/09123/ofriend/analysis/movie_scan/MNI152_T1_2mm_brain.nii.gz',
            # forward (native -> MNI): warp then affine, as you intended
            '-t', f'{fmriprep_dir}/sub-{sub}/transforms/movie/movie_to_MNI_Warp.nii.gz',
            '-t', f'{fmriprep_dir}/sub-{sub}/transforms/movie/movie_to_MNI_Affine.txt'
        ]
        run += 1
        subprocess.run(comm, shell=False)


def main():
    parser = argparse.ArgumentParser(description="Prepare BOLD data for ISC")
    parser.add_argument("sub",
                        help="Subject ID (e.g., 001, matching fMRIPrep naming)")
    parser.add_argument("fmriprep_dir",
                        help="Root directory containing fMRIPrep derivatives")
    parser.add_argument("out_dir",
                        help="Directory to save prepped ISC data")
    args = parser.parse_args()

    sub = args.sub
    fmriprep_dir = Path(args.fmriprep_dir)
    out_dir = Path(args.out_dir) / f"sub-{sub}"
    out_dir.mkdir(parents=True, exist_ok=True)

    movie_to_mni(sub, fmriprep_dir, out_dir)

    # Grab all preprocessed MNI runs for this subject (the ones we just wrote)
    func_files = sorted(out_dir.glob("*MNI_movie.nii.gz"))

    if len(func_files) == 0:
        raise FileNotFoundError(f"No functional files found for {sub}")

    run = 1
    for func_file in func_files:

        # Load BOLD data
        img = nib.load(str(func_file))
        bold = img.get_fdata()  # X,Y,Z,T
        n_scans = bold.shape[-1]
        flat = bold.reshape(-1, n_scans).T  # T×V

        # Load matching confounds (fix typo: "confounds")
        conf_file = fmriprep_dir / f'sub-{sub}/func/sub-{sub}_task-movie_run-0{run}_desc-confounds_timeseries.tsv'
        conf = pd.read_csv(conf_file, sep="\t")

        # Select regressors
        cols = [
            "trans_x", "trans_y", "trans_z",
            "rot_x", "rot_y", "rot_z",
            "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
            "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
            "framewise_displacement", "dvars"
        ]
        conf = conf[cols].fillna(0).values  # T×R

        # Clean: regress confounds + high-pass filter (cutoff=128s)
        tr = img.header.get_zooms()[-1]
        hp = 1. / 128.  # Hz cutoff
        cleaned = clean(
            flat,
            confounds=conf,
            detrend=True,
            standardize=False,
            t_r=tr,
            high_pass=hp
        )  # T×V

        # Back to 4D
        cleaned_4d = cleaned.T.reshape(img.shape)

        out_file = out_dir / func_file.name.replace(
            ".nii.gz",
            "_ISC_prepped.nii.gz"
        )
        nib.save(nib.Nifti1Image(cleaned_4d, img.affine, img.header), str(out_file))
        print(f"Saved {out_file}")
        run += 1


if __name__ == "__main__":
    main()
