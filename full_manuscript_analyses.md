### Note
end to end

### 1. Pre- and post-process raw fMRI data for analyses
* Pre- and post-processing of raw data including registration, slice-time correction, motion estimation, smoothing, etc. are carried out identically to the report which focuses solely on the stat learning task to probe underlying representation and described in detail [here](https://github.com/owenfriend24/temple/tree/main/1_process_raw_data)

### 2. Preparing neural data for classification analyses
* Training data is neural representations of experimental stimuli, as classifier will distinguish whether any pair of stimuli are from the same group or different groups
* This requires item-level estimation of individual stimuli within each run of scanning
* Compute voxelwise betaseries images via least-squares approach (fit GLM at each voxel to distinguish stimuli and regress out motion, extract beta weights for each stimulus; Mumford et al., 2012)
* Normalization of item-level patterns is carried out within main classification analyses (see step 5 below)

### 3. Residualizing neural data for timeseries analyses

### 4. Dual AI-human approach to quantifying memory for movies based on free recall

### 5. Behavioral comparison between movie recall and statistical learning tasks

### 6. ML classification approach to defining adult-like neural activity and out-of-sample generalization to children

### 7. AUC-based regression to capture age differences in functional maturity and figure generation

### 8. Timeseries comparison to quantify child-adult neural alignment during movie-viewing via inter-subject correlation

### 9. Assess behavioral correlates of adult-like activity during movie viewing on free recall and figure generation
