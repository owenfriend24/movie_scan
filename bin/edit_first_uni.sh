#!/bin/bash

if [[ $# -lt 4 ]]; then
    echo "Usage: edit_first_fsf.sh template out_path subject fmriprep_dir"
    exit 1
fi

template=$1
out_path=$2
subject=$3
fm_dir=$4
first_movie=$5
second_movie=$6

# Load any necessary modules
module load python3/3.9.7

# Activate your virtual environment if you're using one
source /home1/09123/ofriend/analysis/temple/profile

mkdir -p /scratch/09123/ofriend/movie_scan/sub-${subject}/

nifti_file1=$fm_dir/sub-"$subject"/func/sub-"$subject"_task-movie_run-01_space-T1w_desc-preproc_bold_ss_4mm.nii.gz
# dimensions of functional images
d1=$(fslinfo "$nifti_file1" | awk '$1 == "dim1" {print $2}')
d2=$(fslinfo "$nifti_file1" | awk '$1 == "dim2" {print $2}')
d3=$(fslinfo "$nifti_file1" | awk '$1 == "dim3" {print $2}')

num_vols1=$(fslinfo "$nifti_file1" | awk '$1 == "dim4" {print $2}')
num_vox1=$((num_vols1*d1*d2*d3))

nifti_file2=$fm_dir/sub-"$subject"/func/sub-"$subject"_task-movie_run-02_space-T1w_desc-preproc_bold_ss_4mm.nii.gz
num_vols2=$(fslinfo "$nifti_file2" | awk '$1 == "dim4" {print $2}')
num_vox2=$((num_vols2*d1*d2*d3))

python /home1/09123/ofriend/analysis/movie_scan/bin/edit_first_uni.py $template $out_path $subject 1 $num_vols1 $num_vox1 $first_movie
python /home1/09123/ofriend/analysis/movie_scan/bin/edit_first_uni.py $template $out_path $subject 2 $num_vols2 $num_vox2 $second_movie

ppl_2=/home1/09123/analysis/movie_scan/perpelxity_2nd_level.fsf
bayes_2=/home1/09123/analysis/movie_scan/bayes_2nd_level.fsf

python /home1/09123/ofriend/analysis/movie_scan/bin/edit_first_uni.py $ppl_2 $out_path $subject 5 2222 $num_vox1 "movie"
python /home1/09123/ofriend/analysis/movie_scan/bin/edit_first_uni.py $bayes_2 $out_path $subject 5 2222 $num_vox2 "movie"