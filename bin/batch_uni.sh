#!/bin/bash
#
# Run randomise to test z-statistic images.

if [[ $# -lt 1 ]]; then
    echo "batch_uni_first.sh $FM SUB"
    exit 1
fi

fmriprep_dir=$1
sub=$2

sub_dir=${fmriprep_dir}/ppi/sub-${sub}/
corr=/corral-repl/utexas/prestonlab/temple

source $HOME/analysis/temple/profile

# create behavioral files for both analyses
/home1/09123/ofriend/analysis/temple/bin/ppi_hip_new.sh ${fmriprep_dir}/ppi/ ${sub} collector
/home1/09123/ofriend/analysis/temple/bin/ppi_txt_behav.py ${corr} both ${sub} ${sub_dir}/ppi/


# create .fsf files for increased connectivity at triplet boundaries
/home1/09123/ofriend/analysis/temple/bin/edit_first_ppi.sh /home1/09123/ofriend/analysis/temple/univ/ppi_first_template.fsf ${sub_dir}/ppi/ ${sub} ${corr}
/home1/09123/ofriend/analysis/temple/bin/run_first_ppi.sh ${fmriprep_dir} ${sub} ${corr}
/home1/09123/ofriend/analysis/temple/bin/edit_second_ppi.sh /home1/09123/ofriend/analysis/temple/univ/2nd_level_ppi_template.fsf ${sub_dir}/ppi/ ${sub} ppi
/home1/09123/ofriend/analysis/temple/bin/run_second_ppis.sh ${fmriprep_dir}/ppi/ ${sub} ppi

# create .fsf files for DECREASED connectivity at triplet boundaries
/home1/09123/ofriend/analysis/temple/bin/edit_first_ppi.sh /home1/09123/ofriend/analysis/temple/univ/ppi_first_inverse.fsf ${sub_dir}/ppi_inverse/ ${sub} ${corr}
/home1/09123/ofriend/analysis/temple/bin/run_first_ppi_inverse.sh ${fmriprep_dir} ${sub} ${corr}
/home1/09123/ofriend/analysis/temple/bin/edit_second_ppi.sh /home1/09123/ofriend/analysis/temple/univ/2nd_level_ppi_template.fsf ${sub_dir}/ppi_inverse/ ${sub} inverse
/home1/09123/ofriend/analysis/temple/bin/run_second_ppis.sh ${fmriprep_dir}/ppi/ ${sub} inverse


