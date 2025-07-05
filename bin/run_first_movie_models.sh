#!/bin/bash
#
# use feat to run first level analyses and transform into mni space


if [[ $# -lt 2 ]]; then
    echo "Usage: run_first_levels.sh fmriprep_dir subject corral"
    exit 1
fi

fmriprep_dir=$1
subject=$2
corral=$3
measure=$4

for run in 1 2; do
#    echo "running first level analysis for sub ${subject}..."
#    feat "${fmriprep_dir}/sub-${subject}/sub-${subject}-uni_level1_${measure}_run-0${run}.fsf"
#    chmod 775 -R "${corral}/sub-${subject}/transforms/movie"
    
    echo "saving first level output to native directory"
    mkdir "${fmriprep_dir}/sub-${subject}/${measure}_out_run${run}.feat/native"
    cp -r "${fmriprep_dir}/sub-${subject}/${measure}_out_run${run}.feat/stats/"* "${fmriprep_dir}/sub-${subject}/${measure}_out_run${run}.feat/native"
    cp "${fmriprep_dir}/sub-${subject}/${measure}_out_run${run}.feat/example_func.nii.gz" "${fmriprep_dir}/sub-${subject}/${measure}_out_run${run}.feat/native/example_func.nii.gz"
    cp "${fmriprep_dir}/sub-${subject}/${measure}_out_run${run}.feat/mean_func.nii.gz" "${fmriprep_dir}/sub-${subject}/${measure}_out_run${run}.feat/native/mean_func.nii.gz"
    cp "${fmriprep_dir}/sub-${subject}/${measure}_out_run${run}.feat/mask.nii.gz" "${fmriprep_dir}/sub-${subject}/${measure}_out_run${run}.feat/native/mask.nii.gz"
    
    # cope images
    echo "transforming cope images"
    track=1
    for cope in ${fmriprep_dir}/sub-${subject}/"${measure}_out_run${run}.feat"/native/cope*; do
    fslreorient2std ${cope}
    antsApplyTransforms -d 3 -i "${cope}" \
    -o ${fmriprep_dir}/sub-${subject}/"${measure}_out_run${run}.feat"/stats/cope${track}.nii.gz \
    -n NearestNeighbor -r /home1/09123/ofriend/analysis/movie_scan/MNI152_T1_2mm_brain.nii.gz \
    -t "${corral}/sub-${subject}/transforms/movie/movie_to_MNI_Warp.nii.gz" \
    -t "${corral}/sub-${subject}/transforms/movie/movie_to_MNI_Affine.txt" \
    -t "${corral}/sub-${subject}/transforms/mask_to_func_ref_movie_Affine.txt"
    ((track=track+1))
    done
    
    # cope images
    echo "transforming varcope images"
    track=1
    for cope in ${fmriprep_dir}/sub-${subject}/"${measure}_out_run${run}.feat"/native/varcope*; do
    fslreorient2std ${cope}
    antsApplyTransforms -d 3 -i "${cope}" \
    -o ${fmriprep_dir}/sub-${subject}/"${measure}_out_run${run}.feat"/stats/varcope${track}.nii.gz \
    -n NearestNeighbor -r /home1/09123/ofriend/analysis/movie_scan/MNI152_T1_2mm_brain.nii.gz \
    -t "${corral}/sub-${subject}/transforms/movie/movie_to_MNI_Warp.nii.gz" \
    -t "${corral}/sub-${subject}/transforms/movie/movie_to_MNI_Affine.txt" \
    -t "${corral}/sub-${subject}/transforms/mask_to_func_ref_movie_Affine.txt"
    ((track=track+1))
    done
    
    
    # func data
    echo "transforming func data"
    
    fslreorient2std "${fmriprep_dir}/sub-${subject}/${measure}_out_run${run}.feat/native/example_func.nii.gz" 
    antsApplyTransforms -d 3 -i "${fmriprep_dir}/sub-${subject}/${measure}_out_run${run}.feat/native/example_func.nii.gz" \
    -o "${fmriprep_dir}/sub-${subject}/${measure}_out_run${run}.feat/example_func.nii.gz" \
    -n BSpline \
    -r /home1/09123/ofriend/analysis/movie_scan/MNI152_T1_2mm_brain.nii.gz \
    -t "${corral}/sub-${subject}/transforms/movie/movie_to_MNI_Warp.nii.gz" \
    -t "${corral}/sub-${subject}/transforms/movie/movie_to_MNI_Affine.txt" \
    -t "${corral}/sub-${subject}/transforms/mask_to_func_ref_movie_Affine.txt"
    
    fslreorient2std "${fmriprep_dir}/sub-${subject}/${measure}_out_run${run}.feat/native/mean_func.nii.gz"
    antsApplyTransforms -d 3 -i "${fmriprep_dir}/sub-${subject}/${measure}_out_run${run}.feat/native/mean_func.nii.gz" \
    -o "${fmriprep_dir}/sub-${subject}/${measure}_out_run${run}.feat/mean_func.nii.gz" \
    -n BSpline \
    -r /home1/09123/ofriend/analysis/movie_scan/MNI152_T1_2mm_brain.nii.gz \
    -t "${corral}/sub-${subject}/transforms/movie/movie_to_MNI_Warp.nii.gz" \
    -t "${corral}/sub-${subject}/transforms/movie/movie_to_MNI_Affine.txt" \
    -t "${corral}/sub-${subject}/transforms/mask_to_func_ref_movie_Affine.txt"


    # mask
    echo "transforming mask"
    fslreorient2std "${fmriprep_dir}/sub-${subject}/${measure}_out_run${run}.feat/mask.nii.gz"
    antsApplyTransforms -d 3 -i "${fmriprep_dir}/sub-${subject}/${measure}_out_run${run}.feat/native/mask.nii.gz"\
     -o "${fmriprep_dir}/sub-${subject}/${measure}_out_run${run}.feat/mask.nii.gz" \
     -n NearestNeighbor \
     -r /home1/09123/ofriend/analysis/movie_scan/MNI152_T1_2mm_brain.nii.gz \
     -t "${corral}/sub-${subject}/transforms/movie/movie_to_MNI_Warp.nii.gz" \
     -t "${corral}/sub-${subject}/transforms/movie/movie_to_MNI_Affine.txt" \
     -t "${corral}/sub-${subject}/transforms/mask_to_func_ref_movie_Affine.txt"
    
    echo "formatting reg folder"
    # set up reg folder
    mkdir "${fmriprep_dir}/sub-${subject}/${measure}_out_run${run}.feat/reg"
    cp /home1/09123/ofriend/analysis/movie_scan/MNI152_T1_2mm_brain.nii.gz \
    "${fmriprep_dir}/sub-${subject}/${measure}_out_run${run}.feat/reg/standard.nii.gz"

    cp "${corral}/sub-${subject}/anat/sub-${subject}_MNI_ss.nii.gz" \
    "${fmriprep_dir}/sub-${subject}/${measure}_out_run${run}.feat/reg/highres.nii.gz"

    cp "/home1/09123/ofriend/analysis/temple/univ/identity.mat" \
    "${fmriprep_dir}/sub-${subject}/${measure}_out_run${run}.feat/reg/example_func2standard.mat"

    updatefeatreg "${fmriprep_dir}/sub-${subject}/${measure}_out_run${run}.feat" -pngs
done

