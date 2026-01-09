### Note
end to end

### 1. Pre- and post-process raw fMRI data for analyses
* Pre- and post-processing of raw data including registration, slice-time correction, motion estimation, smoothing, etc. are carried out identically to the report which focuses solely on the stat learning task to probe underlying representation and described in detail [here](https://github.com/owenfriend24/temple/tree/main/1_process_raw_data)
---
### 2. Preparing neural data for classification analyses
* Training data is neural representations of experimental stimuli, as classifier will distinguish whether any pair of stimuli are from the same group or different groups
* This requires item-level estimation of individual stimuli within each run of scanning
* Compute voxelwise betaseries images via least-squares approach (fit GLM at each voxel to distinguish stimuli and regress out motion, extract beta weights for each stimulus; Mumford et al., 2012)
  * Importantly, betaseries images are computed on per-stimulus basis, allowing for comparison of activity patterns across subjects even when exact stimulus presentation (and subsequent neural timeseries) are not exactly aligned due to randomization and stimulus jitter
  * This is notably different from movie-viewing in which stimulus presentation is time-locked, allowing for a simpler comparison of subject-level timeseries (see step 3 below)
* Betaseries images are derived in participant-specific functional space and then warped to a group template for comparison across subjects
* Additional normalization of item-level patterns is carried out within main classification analyses (see step 5 below)

* wrapper function to create design matrix, run GLM, extract betaseries, concatenate into single 4D image
```
batch_learning_betaseries.sh
```
---
### 3. Residualizing neural data for timeseries analyses
* Movies do not contain discrete stimuli so we cannot use an item-level approach to quantify stimulus-specific activity patterns
* Instead, movies are time-locked across subjects (i.e., all subjects watch the same exact movie; no randomization or jitter in stimulus presentation) allowing for direct comparison of time series (inter-subject timeseries correlation)
* However, this means we cannot regress out confounds like motion in the same way as with the betaseries (where they are included in GLMs)
* To regress out motion, we residualize the timeseries, meaning we fit a GLM with time-locked confounds as explanatory variables and with a 128 Hz high-pass filter
* We then extract the residuals from this model as the task-related signal
* Like the item-level estimation above, movie data is warped into template space for comparison across subjects via the function below
* I recommend running subject's in parallel on HPC cluster due to size and computational demands of these operations
```
prep_isc_data.py $subject(s) $bids_dir $out_dir
```
e.g., 
```
slaunch -J "prep_isc_data.py {} /corral-repl/utexas/prestonlab/temple/ /corral-repl/utexas/prestonlab/temple/movie_processed" $subjects -N 1 -n 1 -n 01:00:00 -p development
```
---
### 4. Dual AI-human approach to quantifying memory for movies based on free recall
* Directly after viewing movies, participants were instructed to describe movies from beginning to end in as much detail as possible, providing naturalistic (but messy) data for what they remembered
* 

### 5. Behavioral comparison between movie recall and statistical learning tasks

### 6. ML classification approach to defining adult-like neural activity and out-of-sample generalization to children

### 7. AUC-based regression to capture age differences in functional maturity and figure generation

### 8. Timeseries comparison to quantify child-adult neural alignment during movie-viewing via inter-subject correlation

### 9. Assess behavioral correlates of adult-like activity during movie viewing on free recall and figure generation
