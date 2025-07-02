#!/bin/bash

if [[ $# -lt 4 ]]; then
    echo "Usage: edit_first_fsf.sh template out_path subject fmriprep_dir"
    exit 1
fi

template=$1
out_path=$2
subject=$3
fm_dir=$4



# Load any necessary modules
module load python3/3.9.7

# Activate your virtual environment if you're using one
source /home1/09123/ofriend/analysis/temple/profile

# Move to the directory containing your Python script
cd /home1/09123/ofriend/analysis/temple/bin

# Run your Python script
python edit_first_uni.py $template $out_path $subject 5 2222 "movie"


