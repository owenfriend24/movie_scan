# Analysis protocol for alignment to adult neural activity during statistical learning task

1) Run post-processing on functional data from fMRIPrep
* skullstrip and smooth functional data, create warp and affine files for resampling to template space
* requires development node; should complete in < 1 hour
```
/home1/09123/ofriend/analysis/temple/bin/prep_func_data.sh {freesurfer_dir} {fmriprep_dir} {subject}
```

2) Compute voxelwise betaseries images for each stimulus presented during learning task and resample to template space
* for each voxel, fit General Linear Model with 12 items as predictors of interest and 14 motion parameters; extract beta weights to derive item-level representation
* batch function below runs intermediate functions to create design matrices, fit models, and extract beta weights; see function for more details
```
/home1/09123/ofriend/analysis/movie_scan/batch_learning_betaseries.sh {subject}
```
* resample to MNI space for comparison across subjects; functional data must be aligned to a common space to compare functional data within the same ROI across subjects
* function below is essentially wrapper for antsApplyTransforms using file naming convention from step 1 above
```
collector_betas_to_mni.sh {subject}
```

3) Apply logistic classifier to item-level neural representations
* classify whether to neural representations come from items from the same triplet (within-triplet) or different triplets (across-triplet) from the statistical learning task (66 total pairs)
  * here, focus analysis on 4th (final) run of statistical learning where triplet learning should be most robust
* since we are interested in children's alignment to adult group, train on adults and test on children
  * script also allows for comparison within each age group using leave-one-out logic, though that is not the primary focus here
  * 
```




5) Extract classifier performance metrics within region of interest for children and adults
