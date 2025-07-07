#!/bin/bash
#
# Run randomise to test z-statistic images.

if [[ $# -lt 1 ]]; then
    echo "batch_uni_first.sh $FM SUB"
    exit 1
fi

fmriprep_dir=$1
sub=$2

corr=/corral-repl/utexas/prestonlab/temple

source $HOME/analysis/temple/profile
#

/home1/09123/ofriend/analysis/movie_scan/bin/run_first_movie_models.sh \
${fmriprep_dir} ${sub} /corral-repl/utexas/prestonlab/temple/ parse

#/home1/09123/ofriend/analysis/movie_scan/bin/run_first_movie_models.sh \
#${fmriprep_dir} ${sub} /corral-repl/utexas/prestonlab/temple/ ppl
#
#/home1/09123/ofriend/analysis/movie_scan/bin/run_first_movie_models.sh \
#${fmriprep_dir} ${sub} /corral-repl/utexas/prestonlab/temple/ bayes


#ppl_2="/home1/09123/ofriend/analysis/movie_scan/perplexity_2nd_level.fsf"
#bayes_2="/home1/09123/ofriend/analysis/movie_scan/bayes_2nd_level.fsf"

#out_path=${fmriprep_dir}/sub-${sub}/
#subject=$sub

#python /home1/09123/ofriend/analysis/movie_scan/bin/edit_first_uni.py ${ppl_2} $out_path $subject 5 2222 999 "movie"
#python /home1/09123/ofriend/analysis/movie_scan/bin/edit_first_uni.py ${bayes_2} $out_path $subject 5 2222 999 "movie"

#feat /scratch/09123/ofriend/movie_scan/sub-${sub}/sub-${sub}-uni_ppl_second_level.fsf
#feat /scratch/09123/ofriend/movie_scan/sub-${sub}/sub-${sub}-uni_bayes_second_level.fsf


feat /scratch/09123/ofriend/movie_scan/sub-${sub}/sub-${sub}-uni_parse_second_level.fsf
