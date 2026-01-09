## End-to-end analysis protocol
The below describes and links to the logic, preprocessing, and analyses of all manuscript analyses. Example calls to functions are provided as well as links to Jupyter and Rmd notebooks with additional code. Analyses and data processing steps are implemented in Python, Bash, and R and require several dependencies, listed in requirements.txt. In addition, fMRI preprocessing and primary analyses using neural data as input (classification and inter-subject correlation) should be run in high-performance computing environments, and these steps include example Slurm job launch calls. 

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
* To create standardized scores which allowed comparison of recall across subjects, we adapted in approach from adult imaging literature to define how many events within a movie a subject recalled (Chen et al., 2017)
* Briefly, several coders generated ground-truth transcripts for each movie, and participant's free recall was parsed into event units and aligned with ground truth labels to determine how many of the events in a movie they described
* Becuase this was extremely time-consuming, human coders completed 20% of the sample as training data, and we developed an approach which integrated GPT-4.1 via API and prompt engineering to code the remaining data, with human's supervising and reviewing all automated coding
 * To ensure automated coding aligned with human coding, we computed several agreement metrics, including ... 
* This approach reduced coding time from months to hours, and maintained precision of ???%
* see [the linked Jupyter notebook]() and associated packages/prompts for automated coding
---
### 5. Behavioral comparison between movie recall and statistical learning tasks
* While alignment between child and adult neural activity was my primary analytical focus, we also predicted children would demonstrate more age-related variation in the novel experimental task
* Further, we reasoned that neural alignment across the two tasks would be most meaningful if performance across the two tasks were related, suggesting simultaneously maturing memory mechanisms rather than two completely distinct and unrelated developmental trajectories
* [This linked Rmd file]() tests and confirms both behavioral hypotheses
---
### 6. ML classification approach to defining adult-like neural activity and out-of-sample generalization to children
* We used an out-of-sample generalization approach and AUC-based regression to compare adult-like neural activity at different ages
* First, we standardized (z-score) template-normalized betaseries images and restricted analyses to the final run of learning when item groups were best learned
* We then used L2 logistic regression to predict, for every possible pair of the 12 items (66 pairs; 12 same group, 54 different group) whether the neural representations reflected two items from the same or different groups
* After validating classifier accuracy within the adult group, we applied the learned weights to each child participant and computed the Area Under the Curve (AUC) to measure how well the adult classifier generalized to each child
 * Higher AUC --> more adult-like activity patterns
* This entire approach was implemented separately in anterior and posterior hippocampus to compare generalization of adult-like activity by subregionn

* classifier function for high-performance computing environment (recommend launching on Slurm with runtime at least 1 hour, betaseries images are large):
```
child_classifier.py $meta_csv $mask $out_dir --zscore_items
```
* optional parameter: --clf=logreg (default) or --clf=svm
* --zscore_items ensures betaseries images are normalized (i.e., coefficients reflect relative rather than absolute magnitude to adjust for subject-level variation in intensity)
---
### 7. AUC-based regression to capture age differences in functional maturity and figure generation
* With the AUC values we derived above, we regressed AUC by age to determine developmental trajectories, and compared AUC to behavior to test the effects of adult-like activity on memory
* Notably, we also included several quality-related covariates to control for individual variation in nuisance parameters like temporal signal to noise ratio (tSNR), number of voxels, attention, etc.
* [This linked Rmd file]() performs the AUC-based regression, comparison to behavioral performance, and figure generation
 ---
### 8. Timeseries comparison to quantify child-adult neural alignment during movie-viewing via inter-subject correlation (ISC)
* After residualizing the data, computing ISC is as simple as normalizing (z-scoring) the timeseries data for each subject and then computing correlation coefficients (normalizing to Fisher's Z)
* The below function creates ISC maps of the whole-brain for each subject in template space and runs in < 2 hours on an HPC development node:
 * the master script iteratively processes all subjects, but individual subjects can also be run using the --only_sub flag
```
isc_to_adult_group.py $bids_dir $mask $out_dir
```
* Using these whole-brain ISC maps, we can then extract average ISC in any template space mask
```
pull_isc_hip_vals.sh
```
---
### 9. Assess behavioral correlates of adult-like activity during movie viewing on free recall and figure generation
* Lastly, we compare the ISC values we derived above by age and memory performance
[This linked Rmd file]() performs those comparisons and generates corresponding figures


