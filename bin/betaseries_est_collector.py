#!/usr/bin/env python

import os
#import mvpa2
#from mvpa2 import *
from random import sample
from matplotlib import pylab
from numpy import *
from pylab import *
from scipy.io import *
from mvpa2.misc.fsl.base import *
from mvpa2.datasets.mri import fmri_dataset
#from mvpa2.measures.rsa import PDist
import numpy as N
import nibabel
import scipy.stats
from scipy.stats.mstats import zscore
from scipy.ndimage import convolve1d
from scipy.sparse import spdiags
from scipy.linalg import toeplitz
from mvpa2.datasets.mri import *
#import os
import sys
from copy import copy

### directory for experiment and design matrices ###
expdir = '/scratch/09123/ofriend/movie_scan'

### subjects to analyze as the input ###
sub = sys.argv[1]

### regressor indices and output ###
good_evs = range(0,12)
output_evs = good_evs
ntrials_total = len(good_evs)

### runs to model ###
runs = ['1','2', '3', '4']
### all of the directories for this subject ###
designdir =expdir+'/sub-%s/betaseries'%(sub)
subdir = expdir+'/sub-%s'%(sub)

#bolddir = f'/corral-repl/utexas/prestonlab/temple/sub-{sub}/func/'
#mask = f'/corral-repl/utexas/prestonlab/temple/freesurfer/sub-{sub}/mri/out/brainmask_func_dilated.nii.gz'

corr = '/corral-repl/utexas/prestonlab/temple/'

bolddir = corr+'/func/'
mask = corr + '/freesurfer/sub-%s/mri/out/brainmask_func_dilated.nii.gz'%(sub)

confounddir = subdir + '/collector_txt'
modeldir = designdir
betadir = designdir
os.makedirs(betadir, exist_ok=True)
### for each run ###
for run in runs:

        matfile = designdir+'/sub-%s_betaL1_collector_run-%s.mat'%(sub,run)
        desmat = FslGLMDesign(matfile)
        nevs = desmat.mat.shape[1]
        ntp = desmat.mat.shape[0]

        # load in the data and the design matrix
        bolddata = bolddir+'/sub-%s_task-collector_run-0%s_space-T1w_desc-preproc_bold_ss_4mm.nii.gz'%(sub,run)
        confoundfile = confounddir+'/sub-%s_task-collector_run-0%s_formatted_confounds.txt'%(sub,run)


        # load data
        data = fmri_dataset(bolddata,mask=mask)

        # below here is a bunch of Jeannette stuff
        dm_nuisance1 = N.loadtxt(confoundfile)


        trial_ctr = 0
        all_conds = []
        beta_maker = N.zeros((ntrials_total,ntp))

        # for all of the "good" evs
        for e in range(len(good_evs)):

            ev = good_evs[e]

            dm_toi = desmat.mat[:,ev]

            other_good_evs = [x for x in good_evs if x != ev]
            dm_otherevs = desmat.mat[:,other_good_evs]
            dm_otherevs = N.sum(dm_otherevs[:,:,N.newaxis],axis=1)

            # Put together the design matrix
            dm_full = N.hstack((dm_toi[:,N.newaxis],dm_otherevs,dm_nuisance1)) #,dm_nuisance2[:,N.newaxis],dm_nuisance3[:,N.newaxis],dm_nuisance4[:,N.newaxis],dm_nuisance5[:,N.newaxis],dm_nuisance6[:,N.newaxis]))

            # making betas
            dm_full = dm_full - N.kron(N.ones((dm_full.shape[0],dm_full.shape[1])), \
                    	N.mean(dm_full,0))[0:dm_full.shape[0],0:dm_full.shape[1]]
            dm_full=N.hstack((dm_full,N.ones((ntp,1))))
            beta_maker_loop=N.linalg.pinv(dm_full)
            beta_maker[trial_ctr,:]=beta_maker_loop[0,:]
            trial_ctr+=1

        # this uses Jeanette's trick of extracting the beta-forming vector for each
        # trial and putting them together, which allows estimation for all trials
        # at once
        glm_res_full = N.dot(beta_maker,data.samples)
        #residuals = data.samples - glm_res_full
        # map the data into images and save to betaseries directory
        for e in output_evs:
            outdata = zscore(glm_res_full[e])
            #out_res = zscore(residuals[e])
            ni = map2nifti(data,data=outdata)
            #ni_res = map2nifti(data,data=out_res)
            ni.to_filename(betadir+'/betaOUT_collector_run-%s_ev-%03d.nii.gz'%(run,(e+1)))
            #ni_res.to_filename(betadir+'/beta_residuals_run-%s_ev-%03d.nii.gz'%(run,(e+1)))
