#!/usr/bin/env bash

if [[ $# -lt 1 ]]; then
  echo "Usage: sl_to_mni.sh <subID>"
  exit 1
fi

sub="$1"

in_dir="/scratch/09123/ofriend/movie_scan/sub-${sub}/betaseries"
ref="/home1/09123/ofriend/analysis/temple/bin/templates/MNI152_T1_func_brain.nii.gz"
warp="/corral-repl/utexas/prestonlab/temple/sub-${sub}/transforms/native_to_MNI_Warp.nii.gz"
affn="/corral-repl/utexas/prestonlab/temple/sub-${sub}/transforms/native_to_MNI_Affine.txt"


for file in ${in_dir}/*.nii.gz; do
  base="$(basename "$file" .nii.gz)"
  out="${in_dir}/${base}_MNI.nii.gz"

  antsApplyTransforms -d 3 \
    -i "$file" \
    -o "$out" \
    -r "$ref" \
    -t "$warp" \
    -t "$affn" \
    -n NearestNeighbor
done
