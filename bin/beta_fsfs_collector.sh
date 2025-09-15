#!/bin/bash
subj=$1
template_file='/home1/09123/ofriend/analysis/temple/bin/templates/betaL1_template_collector.fsf'


# list of subjects
subjects=(
  $subj
)

# iterate through the subjects list
for subject in "${subjects[@]}"; do
  # subject_output_directory is set based on the subject directory
  subject_output_directory="/scratch/09123/ofriend/movie_scan/sub-${subject}/betaseries"

  if [[ ! -d "$subject_output_directory" ]]; then
    # create a directory if it doesn't exist
    mkdir -p "$subject_output_directory"
  fi

  for run in 1 2 3 4; do
    #fmri_file_path="/corral-repl/utexas/prestonlab/temple/sub-${subject}/func/sub-${subject}_task-arrow_run-0${run}_space-T1w_desc-preproc_bold_ss_4mm.nii.gz"
    # Define the path to the subject's fMRI file
    fmri_file_path="/corral-repl/utexas/prestonlab/temple/sub-${subject}/func/sub-${subject}_task-collector_run-0${run}_space-T1w_desc-preproc_bold_ss_4mm.nii.gz"

    # Use fslnvols to get the number of TRs in the fMRI file
    num_trs=$(fslnvols "$fmri_file_path")

    # pull the number of voxels for the given functional run
    # this isn't actually necessary since we're not running a univariate analysis in feat, just creating the design matrix
    dims=($(fslinfo "$fmri_file_path" | awk '/^dim[1234]/ {print $2}'))
    total_voxels=$(( dims[0] * dims[1] * dims[2] * dims[3] ))

    # output files are stored in the subject_output_directory
    output_file="${subject_output_directory}/sub-${subject}_betaL1_collector_run-${run}.fsf"

    # Use sed to replace the subject, run, and number of TRs in the template
    sed -e "s/sub-999/sub-${subject}/g" \
        -e "s/run-1/run-${run}/g" \
        -e "s/69056000/${total_voxels}/g" \
	      -e "s/run-01/run-0${run}/g" \
        -e "s/222/${num_trs}/g" \
        "$template_file" > "$output_file"

  done
done
