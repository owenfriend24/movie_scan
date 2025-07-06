#!/bin/env bash
#
# cluster simulations for preliminary age group RS analyses

if [[ $# -lt 1 ]]; then
    echo "Usage: clust_sim.sh fmriprep_dir"
    exit 1
fi

fmriprep_dir=$1

module load afni
export OMP_NUM_THREADS=None

mkdir -p ${fmriprep_dir}/clust_sim
cd ${fmriprep_dir}/clust_sim

#3dClustSim -mask /home1/09123/ofriend/analysis/movie_scan/b_hip_func.nii.gz -acf 0.620 2.485 7.502 -nodec -prefix bayes_hip
3dClustSim -mask /scratch/09123/ofriend/movie_scan/mni_masks/group_75_mask.nii.gz -acf 0.620 2.485 7.502 -nodec -prefix bayes_gm_new

#3dClustSim -mask /home1/09123/ofriend/analysis/movie_scan/b_hip_func.nii.gz -acf 0.620 2.485 7.495 -nodec -prefix ppl_hip
3dClustSim -mask /scratch/09123/ofriend/movie_scan/mni_masks/group_75_mask.nii.gz -acf 0.620 2.485 7.495 -nodec -prefix ppl_gm_new