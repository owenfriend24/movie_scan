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

/home1/09123/ofriend/analysis/movie_scan/bin/run_first_movie_models.sh \
${fmriprep_dir} ${sub} /corral-repl/utexas/prestonlab/temple/ ppl

/home1/09123/ofriend/analysis/movie_scan/bin/run_first_movie_models.sh \
${fmriprep_dir} ${sub} /corral-repl/utexas/prestonlab/temple/ bayes
