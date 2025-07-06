#!/bin/bash

if [[ $# -lt 2 ]]; then
    echo "Usage: temple_acf.sh fmriprep_dir subject"
    exit 1
fi

fmriprep_dir="$1"
subject="$2"
module load afni

out_file="${fmriprep_dir}/bayes_residuals/by_subject_acf.txt"
mask_path="/corral-repl/utexas/prestonlab/temple/freesurfer/sub-${subject}/mri/b_gray_movie.nii.gz"

for run in {1..2}; do
    resid_path="${fmriprep_dir}/bayes_residuals/sub-${subject}_bayes_r${run}_resid.nii.gz"

    if [[ -f "$resid_path" ]]; then
        output=$(3dFWHMx -mask "$mask_path" -ACF NULL -input "$resid_path" -arith)
        acf_coefs=$(echo "$output" | tail -n 1 | awk '{print $(NF-3), $(NF-2), $(NF-1), $NF}')
        echo "$subject $run $acf_coefs" >> "$out_file"
    else
        echo "Missing residual file for $subject run $run"
    fi
done

out_file="${fmriprep_dir}/ppl_residuals/by_subject_acf.txt"
mask_path="/corral-repl/utexas/prestonlab/temple/freesurfer/sub-${subject}/mri/out/b_gray_movie.nii.gz"

for run in {1..2}; do
    resid_path="${fmriprep_dir}/ppl_residuals/sub-${subject}_ppl_r${run}_resid.nii.gz"

    if [[ -f "$resid_path" ]]; then
        output=$(3dFWHMx -mask "$mask_path" -ACF NULL -input "$resid_path" -arith)
        acf_coefs=$(echo "$output" | tail -n 1 | awk '{print $(NF-3), $(NF-2), $(NF-1), $NF}')
        echo "$subject $run $acf_coefs" >> "$out_file"
    else
        echo "Missing residual file for $subject run $run"
    fi
done

