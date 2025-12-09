#!/usr/bin/env bash

#ANT_MASK=/scratch/09123/ofriend/movie_scan/to_adult_merge/ant_hip_func.nii.gz
#POS_MASK=/scratch/09123/ofriend/movie_scan/to_adult_merge/post_hip_func.nii.gz

ANT_MASK=/scratch/09123/ofriend/movie_scan/to_adult_merge/b_hip_ant_bin_new.nii.gz
POS_MASK=/scratch/09123/ofriend/movie_scan/to_adult_merge/b_hip_post_bin_new.nii.gz

echo "subject,anterior_mean,posterior_mean" > /scratch/09123/ofriend/movie_scan/new_ISC_test/merged/isc_values_hip_bin_new.csv

for f in /scratch/09123/ofriend/movie_scan/new_ISC_test/merged/sub-temple*_iscToAdult_z*.nii.gz; do
    subj=$(basename "$f" | cut -d_ -f1)

    ant=$(fslstats "$f" -k "$ANT_MASK" -M)
    pos=$(fslstats "$f" -k "$POS_MASK" -M)

    echo "$subj,$ant,$pos" >> /scratch/09123/ofriend/movie_scan/new_ISC_test/merged/isc_values_hip_bin_new.csv
done