#!/bin/env bash

if [[ $# -lt 2 ]]; then
    echo "Usage: tsnr_by_roi data_dir subject masktype.  MAKE SURE YOU HAVE ALREADY RUN tsnr_maps.sh"
    exit 1
fi

fmdir=$1
subject=$2
masktype=$3
type=$4

func_dir=${fmdir}/sub-${subject}/func
tsnr_dir=${func_dir}/tsnr
output_csv="${func_dir}/tsnr/tsnr_values_${masktype}_${type}.csv"

# Create CSV file with headers
echo "run,mask,tsnr,nvoxs" > ${output_csv}

# Define mask paths
if [[ "$type" == "movie" ]]; then
    masks=(
        "${fmdir}/sub-${subject}/masks/hip_masks/func-b_hip_ant.nii.gz"
        "${fmdir}/sub-${subject}/masks/hip_masks/func-b_hip_post.nii.gz"
    )
elif [[ "$type" == "collector" ]]; then
    masks=(
        "/scratch/09123/ofriend/movie_scan/sub-${subject}/ant_hip_func.nii.gz"
        "/scratch/09123/ofriend/movie_scan/sub-${subject}/post_hip_func.nii.gz"
    )

fi

# subject missing one run of arrow
if [ "$type" = "movie" ]; then
    runs="1 2"

else
    runs="1 2 3 4"

fi

for run in $runs; do
    tsnr_file="${tsnr_dir}/${type}_run_${run}_tsnr_map.nii.gz"

    for mask in "${masks[@]}"; do
        tsnr_value=$(fslstats ${tsnr_file} -k ${mask} -M)
        nvoxs=$(fslstats ${mask} -V | awk '{print $1}')  # voxel count
        echo "${run},${mask},${tsnr_value},${nvoxs}" >> ${output_csv}
    done
done

echo "Saved tSNR values to ${output_csv}"