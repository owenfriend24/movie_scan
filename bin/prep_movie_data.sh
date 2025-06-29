#!/bin/env bash

if [[ $# -lt 3 ]]; then
    echo "Usage: prep_func_data.sh freesurfer_dir fmriprep_dir subject"
    exit 1
fi

fsdir=$1
fmdir=$2
subject=$3

# Load any necessary modules
module load python3/3.9.7

# Activate your virtual environment if you're using one
source /home1/09123/ofriend/analysis/temple/profile

# Run your Python script

python /home1/09123/ofriend/analysis/movie_scan/bin/prep_movie_data.py "${fsdir}" "${fmdir}" "${subject}"

temple_smooth.sh $fmdir $fsdir $subject "movie"
# move movie files back up to the func directory, get rid of intermediate
mv ${fmdir}/sub-${subject}/func/skullstripped_T1/*4mm.nii.gz ${fmdir}/sub-${subject}/func/
rm -R ${fmdir}/sub-${subject}/func/skullstripped_T1/