#!/bin/env bash

if [[ $# -lt 2 ]]; then
    echo "Usage: tsnr_maps data_dir subject"
    exit 1
fi

fmdir=$1
subject=$2

mkdir -p ${fmdir}/sub-${subject}/func/tsnr/
func_dir=${fmdir}/sub-${subject}/func

for run in 1 2 3 4; do
  fslmaths ${func_dir}/sub-${subject}_task-collector_run-0${run}_space-T1w_desc-preproc_bold.nii.gz -mas \
  ${fmdir}/freesurfer/sub-${subject}/mri/out/brainmask_func_dilated.nii.gz \
  ${func_dir}/tsnr/collector_run_${run}_stripped.nii.gz

  fslmaths ${func_dir}/tsnr/collector_run_${run}_stripped.nii.gz -Tmean ${func_dir}/tsnr/collector_run_${run}_meanfunc.nii.gz
  fslmaths ${func_dir}/tsnr/collector_run_${run}_stripped.nii.gz -Tstd ${func_dir}/tsnr/collector_run_${run}_stdfunc.nii.gz
  fslmaths ${func_dir}/tsnr/collector_run_${run}_meanfunc.nii.gz -div ${func_dir}/tsnr/collector_run_${run}_stdfunc.nii.gz \
  ${func_dir}/tsnr/collector_run_${run}_tsnr_map.nii.gz

  fslstats ${func_dir}/tsnr/collector_run_${run}_tsnr_map.nii.gz -M >> ${func_dir}/tsnr/collector_tsnr_values.txt

  rm ${func_dir}/tsnr/collector_run_${run}_stripped.nii.gz
  rm ${func_dir}/tsnr/collector_run_${run}_meanfunc.nii.gz
  rm ${func_dir}/tsnr/collector_run_${run}_stdfunc.nii.gz

done