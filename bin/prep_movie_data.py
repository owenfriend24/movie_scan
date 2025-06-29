#!/usr/bin/env python

import os
import subprocess
from pathlib import Path
import argparse

def run(command):
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(f"Output: {result.stdout}")
    if result.stderr:
        print(f"Error: {result.stderr}")


def extract_func(fs_dir, fmriprep_dir, sub, task, num_runs): 
    # set up freesurfer paths and functional data paths
    fs_dir = Path(fs_dir)
    src = fs_dir / f'sub-{sub}/mri'
    fmriprep_dir = Path(fmriprep_dir)/f'sub-{sub}'
    func_dir = fmriprep_dir /'func'
    
    # create directory to store affine transform files
    run(f'mkdir -p {fmriprep_dir}/transforms')
    
    # create directory within func to store skullstripped func data
    run(f'mkdir -p {func_dir}/skullstripped_T1')
    dest = func_dir / 'skullstripped_T1'
    
    # path to mask created from freesurfer output
    highres_mask = fs_dir / f'sub-{sub}/mri/out/brainmask'
    
    # output path for mask transformed to functional space
    mask_func = fs_dir / f'sub-{sub}/mri/out/brainmask_func'
    
    # functional data to reference in transform for dimensions/space
    ref_func = func_dir / f'sub-{sub}_task-{task}_run-01_space-T1w_boldref'
    
    # # create affine txt file to go from anatomical to functional space
    # run(f'ANTS 3 -m MI[ {ref_func}.nii.gz, {highres_mask}.nii.gz,1,32] -o {fmriprep_dir}/transforms/mask_to_func_ref_ --rigid-affine true -i 0')
    # print('created affine file')
    #
    # apply affine file to transform mask from T1 anatomical to T1 functional space to match functional data's dimensions
    run(f'antsApplyTransforms -d 3 -i {highres_mask}.nii.gz -o {mask_func}.nii.gz -r {ref_func}.nii.gz -t {fmriprep_dir}/transforms/mask_to_func_ref_Affine.txt')
    print('transformed mask to functional space')
    
    
    run(f'fslmaths {mask_func}.nii.gz -kernel sphere 3 -dilD {mask_func}_dilated.nii.gz')
    print('dilated mask')
    
    # skullstrip the functional data
    for func_run in range(1, num_runs +1):
        print(f'attempting to run: fslmaths {func_dir}/sub-{sub}_task-{task}_run-0{func_run}_space-T1w_desc-preproc_bold.nii.gz -mas {mask_func}_dilated.nii.gz {func_dir}/skullstripped_T1/sub-{sub}_task-{task}_run-0{func_run}_space-T1w_desc-preproc_bold_ss.nii.gz')
        
        run(f'fslmaths {func_dir}/sub-{sub}_task-{task}_run-0{func_run}_space-T1w_desc-preproc_bold.nii.gz -mas {mask_func}_dilated.nii.gz {func_dir}/skullstripped_T1/sub-{sub}_task-{task}_run-0{func_run}_space-T1w_desc-preproc_bold_ss.nii.gz')
        print(f'skullstripped run {func_run}')


def main(fs_dir, fmriprep_dir, sub):
    run('source /home1/09123/ofriend/analysis/temple/profile')
    # extract_fs(fs_dir, sub)
    # print(f'\n\nMASK EXTRACTED FOR SUB-{sub}\n\n')
    extract_func(fs_dir, fmriprep_dir, sub, 'movie', 2)
    print(f'\n\nSKULLSTRIPPING COMPLETE FOR SUB-{sub}\n\n')

          
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fs_dir", help="freesurfer directory")
    parser.add_argument("fmriprep_dir", help="fmriprep derivatives directory")
    parser.add_argument("sub", help="subject number; include full templeXXX")
    args = parser.parse_args()
    main(args.fs_dir, args.fmriprep_dir, args.sub)
