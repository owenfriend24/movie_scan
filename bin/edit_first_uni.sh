#!/bin/bash

if [[ $# -lt 4 ]]; then
    echo "Usage: edit_first_fsf.sh template out_path subject fmriprep_dir"
    exit 1
fi

template=$1
out_path=$2
subject=$3
fm_dir=$4

module load python3/3.9.7

source /home1/09123/ofriend/analysis/temple/profile


mkdir /scratch/09123/ofriend/temple/prepro_data/derivatives/fmriprep/sub-$subject/univ

nifti_file1=$fm_dir/sub-"$subject"/func/skullstripped_T1/sub-"$subject"_task-collector_run-01_space-T1w_desc-preproc_bold_ss_4mm.nii.gz
num_vols1=$(fslinfo "$nifti_file1" | awk '$1 == "dim4" {print $2}')

nifti_file2=$fm_dir/sub-"$subject"/func/skullstripped_T1/sub-"$subject"_task-collector_run-02_space-T1w_desc-preproc_bold_ss_4mm.nii.gz
num_vols2=$(fslinfo "$nifti_file2" | awk '$1 == "dim4" {print $2}')

python edit_first_uni.py $template $out_path $subject 1 $num_vols1
python edit_first_uni.py $template $out_path $subject 2 $num_vols2



