#!/usr/bin/env bash

ANT_MASK=/scratch/09123/ofriend/movie_scan/to_adult_merge/ant_hip_func.nii.gz
POS_MASK=/scratch/09123/ofriend/movie_scan/to_adult_merge/post_hip_func.nii.gz

echo "subject,anterior_mean,posterior_mean" > isc_values_hip.csv

for f in /scratch/09123/ofriend/movie_scan/to_adult_merge/sub-temple*_iscToAdult_z*.nii.gz; do
    subj=$(echo $f | cut -d_ -f1)

    ant=$(fslstats "$f" -k "$ANT_MASK" -M)
    pos=$(fslstats "$f" -k "$POS_MASK" -M)

    echo "$subj,$ant,$pos" >> /scratch/09123/ofriend/movie_scan/to_adult_merge/isc_values_hip.csv
done