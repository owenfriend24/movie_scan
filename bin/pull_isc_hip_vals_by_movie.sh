#!/usr/bin/env bash

ANT_MASK=/scratch/09123/ofriend/movie_scan/to_adult_movies/ant_hip_func.nii.gz
POS_MASK=/scratch/09123/ofriend/movie_scan/to_adult_movies/post_hip_func.nii.gz

echo "subject,movie,anterior_mean,posterior_mean" > isc_values_hip.csv

for f in /scratch/09123/ofriend/movie_scan/to_adult_movies/sub-temple*_movie_coin_iscToAdult_z.nii.gz; do
    subj=$(basename "$f" | cut -d_ -f1)

    ant=$(fslstats "$f" -k "$ANT_MASK" -M)
    pos=$(fslstats "$f" -k "$POS_MASK" -M)
    mov="coin"
    echo "$subj,$mov,$ant,$pos" >> /scratch/09123/ofriend/movie_scan/to_adult_movies/isc_values_hip_by_movie.csv
done

for f in /scratch/09123/ofriend/movie_scan/to_adult_movies/sub-temple*_movie_jinx_iscToAdult_z.nii.gz; do
    subj=$(basename "$f" | cut -d_ -f1)

    ant=$(fslstats "$f" -k "$ANT_MASK" -M)
    pos=$(fslstats "$f" -k "$POS_MASK" -M)
    mov="jinx"
    echo "$subj,$mov,$ant,$pos" >> /scratch/09123/ofriend/movie_scan/to_adult_movies/isc_values_hip_by_movie.csv
done