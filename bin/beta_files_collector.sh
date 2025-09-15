#!/bin/bash

SUBNUM=$1
runNUM="1 2 3 4"

for sbj in $SUBNUM

do


    for run in $runNUM

    do


    feat_model /scratch/09123/ofriend/movie_scan/sub-${sbj}/betaseries/sub-${sbj}_betaL1_collector_run-${run} #fsf file

    rm /scratch/09123/ofriend/movie_scan/sub-${sbj}/betaseries/sub-${sbj}_betaL1_collector_run-${run}_cov.png
    rm /scratch/09123/ofriend/movie_scan/sub-${sbj}/betaseries/sub-${sbj}_betaL1_collector_run-${run}_cov.ppm
    rm /scratch/09123/ofriend/movie_scan/sub-${sbj}/betaseries/sub-${sbj}_betaL1_collector_run-${run}.con
    rm /scratch/09123/ofriend/movie_scan/sub-${sbj}/betaseries/sub-${sbj}_betaL1_collector_run-${run}.frf
    rm /scratch/09123/ofriend/movie_scan/sub-${sbj}/betaseries/sub-${sbj}_betaL1_collector_run-${run}.min
    rm /scratch/09123/ofriend/movie_scan/sub-${sbj}/betaseries/sub-${sbj}_betaL1_collector_run-${run}.png
    rm /scratch/09123/ofriend/movie_scan/sub-${sbj}/betaseries/sub-${sbj}_betaL1_collector_run-${run}.ppm
    rm /scratch/09123/ofriend/movie_scan/sub-${sbj}/betaseries/sub-${sbj}_betaL1_collector_run-${run}.trg

    done


done
