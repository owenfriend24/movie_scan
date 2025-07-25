#!/bin/bash

mask="hip_mask.nii.gz"  # your binary 2mm hippocampus mask in MNI space
output_dir="hip_slices"
mkdir -p $output_dir

# Get Y bounds
read xmin xsize ymin ysize zmin zsize <<< $(fslstats $mask -w)

declare -A slice_voxels
declare -A slice_files

# Step 1: Extract each Y slice
for (( i=0; i<ysize; i++ )); do
    y=$((ymin + i))
    slice_file="${output_dir}/slice_y${y}.nii.gz"
    fslroi $mask $slice_file 0 -1 $y 1 0 -1
    
    # Count voxels
    voxels=$(fslstats $slice_file -V | awk '{print $1}')
    slice_voxels[$y]=$voxels
    slice_files[$y]=$slice_file
done

# Step 2: Merge low-voxel slices with closest valid slice
valid_slices=()
merged_dir="${output_dir}/final_slices"
mkdir -p $merged_dir

for y in "${!slice_voxels[@]}"; do
    voxels=${slice_voxels[$y]}
    file=${slice_files[$y]}
    
    if (( voxels >= 2 )); then
        cp "$file" "${merged_dir}/slice_y${y}.nii.gz"
        valid_slices+=($y)
    fi
done

# Now deal with small slices
for y in "${!slice_voxels[@]}"; do
    voxels=${slice_voxels[$y]}
    file=${slice_files[$y]}
    
    if (( voxels < 2 && voxels > 0 )); then
        # Find nearest valid slice
        min_dist=999
        nearest=""
        for vy in "${valid_slices[@]}"; do
            dist=$(( abs = y > vy ? y - vy : vy - y ))
            if (( dist < min_dist )); then
                min_dist=$dist
                nearest=$vy
            fi
        done

        # Merge with nearest
        target="${merged_dir}/slice_y${nearest}.nii.gz"
        fslmaths $target -add $file $target
    fi
done

