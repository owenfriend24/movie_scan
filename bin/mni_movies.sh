#!/bin/bash

if [[ $# -lt 2 ]]; then
    echo "Usage: mni_transforms.sh fmriprep_dir subject"
    exit 1
fi

fmriprep_dir=$1
subject=$2

#fslmaths "${fmriprep_dir}/sub-${subject}/anat/sub-${subject}_desc-preproc_T1w.nii.gz" \
#-mas "${fmriprep_dir}/sub-${subject}/anat/sub-${subject}_desc-brain_mask.nii.gz" \
#"${fmriprep_dir}/sub-${subject}/anat/sub-${subject}_T1w_ss.nii.gz"

# create transforms from anatomical space to MNI space
ANTS 3 -m CC[/home1/09123/ofriend/analysis/movie_scan/MNI152_T1_2mm_brain.nii.gz, \
"${fmriprep_dir}/sub-${subject}/anat/sub-${subject}_T1w_ss.nii.gz",1,5] \
-t SyN[0.25] -r Gauss[3,0] \
-o "${fmriprep_dir}/sub-${subject}/transforms/movie/movie_to_MNI_" \
-i 30x90x20 --use-Histogram-Matching \
--number-of-affine-iterations 10000x10000x10000x10000x10000 \
--MI-option 32x16000



