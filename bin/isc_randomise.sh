#!/bin/bash
#
# Run randomise to test z-statistic images.

if [[ $# -lt 2 ]]; then
    echo "randomise_new.sh roi comp"
    exit 1
fi

roi=$1
comp=$2

if [[ $roi == 'b_hip' ]]; then
  grp_mask_path=/corral-repl/utexas/prestonlab/temple/group_masks/hip_func/b_hip_func.nii.gz
elif [[ $roi == 'b_gray_func' ]]; then
  grp_mask_path=/scratch/09123/ofriend/movie_scan/gm_masks/group_mask.nii.gz
fi

mkdir -p /scratch/09123/ofriend/movie_scan/rando/randomise_out/

randomise -i /scratch/09123/ofriend/movie_scan/rando/group_z.nii.gz \
-o /scratch/09123/ofriend/movie_scan/rando/randomise_out/${roi}_cont_age \
-d /scratch/09123/ofriend/movie_scan/rando/age_cont.mat \
-t /scratch/09123/ofriend/movie_scan/rando/age_cont.con \
-m $grp_mask_path \
-n 5000 -x --uncorrp

randomise -i /scratch/09123/ofriend/movie_scan/rando/group_z.nii.gz \
-o /scratch/09123/ofriend/movie_scan/rando/randomise_out/${roi}_group_mean \
-m $grp_mask_path \
-1 \
-n 5000 -x  --uncorrp
