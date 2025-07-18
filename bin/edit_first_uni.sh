#!/bin/bash

if [[ $# -lt 4 ]]; then
    echo "Usage: edit_first_uni.sh out_path subject func_dir first_movie second_movie"
    exit 1
fi

out_path=$1
subject=$2
fm_dir=$3
first_movie=$4
second_movie=$5

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

parse_template="/home1/09123/ofriend/analysis/movie_scan/parse_first_template.fsf"

#ppl_1="/home1/09123/ofriend/analysis/movie_scan/perplexity_template.fsf"
#bayes_1="/home1/09123/ofriend/analysis/movie_scan/bayes_template.fsf"

python /home1/09123/ofriend/analysis/movie_scan/bin/edit_first_uni.py ${parse_template} $out_path $subject 1 $num_vols1 $num_vox1 $first_movie
python /home1/09123/ofriend/analysis/movie_scan/bin/edit_first_uni.py ${parse_template} $out_path $subject 2 $num_vols2 $num_vox2 $second_movie

#python /home1/09123/ofriend/analysis/movie_scan/bin/edit_first_uni.py ${bayes_1} $out_path $subject 1 $num_vols1 $num_vox1 $first_movie
#python /home1/09123/ofriend/analysis/movie_scan/bin/edit_first_uni.py ${bayes_1} $out_path $subject 2 $num_vols2 $num_vox2 $second_movie

#
#ppl_2="/home1/09123/ofriend/analysis/movie_scan/perplexity_2nd_level.fsf"
#bayes_2="/home1/09123/ofriend/analysis/movie_scan/bayes_2nd_level.fsf"
parse_2="/home1/09123/ofriend/analysis/movie_scan/parse_2nd_level.fsf"
#
#python /home1/09123/ofriend/analysis/movie_scan/bin/edit_first_uni.py ${ppl_2} $out_path $subject 5 2222 $num_vox1 "movie"
#python /home1/09123/ofriend/analysis/movie_scan/bin/edit_first_uni.py ${bayes_2} $out_path $subject 5 2222 $num_vox2 "movie"
python /home1/09123/ofriend/analysis/movie_scan/bin/edit_first_uni.py ${parse_2} $out_path $subject 5 2222 $num_vox2 "movie"