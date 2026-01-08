# Analysis protocol for alignment to adult neural activity during statistical learning task

### 1) Run post-processing on functional data from fMRIPrep
* skullstrip and smooth functional data, create warp and affine files for resampling to template space
* requires development node; should complete in < 1 hour
```
prep_func_data.sh {freesurfer_dir} {fmriprep_dir} {subject}
```

### 2) Compute voxelwise betaseries images for each stimulus presented during learning task and resample to template space
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

### 3) Apply logistic classifier to item-level neural representations
* classify whether to neural representations come from items from the same triplet (within-triplet) or different triplets (across-triplet) from the statistical learning task (66 total pairs)
  * here, focus analysis on 4th (final) run of statistical learning where triplet learning should be most robust
* since we are interested in children's alignment to adult group, train on adults and test on children
  * script also allows for comparison within each age group using leave-one-out logic, though that is not the primary focus here
* Primary analysis uses following parametrs:
  * L2 Logistic CLF - binary decision (within- or across- triplet) with L2 regularization term (svm also an option, but doesn't perform as well as logreg)
  * 1.0 Inverse regularization strength (scikit-learn's default, but can be changed if wanted)
  * Z-score beta images by voxel (within subject, run) - controls for baseline activation differences and normalizes feature scaling so that classifier is sensitive to triplet strucutre, not baseline or mean activation (i.e., raw signal differences which we're not interested in here)
  * Meta CSV (subject,age,age_group,run,item_id,triplet_id,beta_path) - make sure beta path is to MNI-transformed images, not native
  * Mask - binary nifti in functional MNI space to match betaseries images (1.7mm isotropic-ish for this protocol)
```
child_classifier.py {meta csv} {mask} {output_dir} --clf=logreg --zscore_items
```


### 4) Extract classifier performance metrics within region of interest for children and adults

