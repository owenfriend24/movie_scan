#!/usr/bin/env python
#
# generate .txt files without headers for motion confounds or for collector behavioral data

from pathlib import Path
import pandas as pd
import os
import argparse
import os
import subprocess

def run_com(command):
    subprocess.run(command, shell=True)

def main(data_dir, file_type, sub, first_mov, skip_second):
    func_dir = data_dir + f'/sub-{sub}/func/'
    confs = []
    if file_type == 'motion' or file_type == 'all':
        conf1 = pd.read_table(func_dir + f'sub-{sub}_task-movie_run-01_desc-confounds_timeseries.tsv')
        confs.append(conf1)
        if not skip_second:
            conf2 = pd.read_table(func_dir + f'sub-{sub}_task-movie_run-02_desc-confounds_timeseries.tsv')
            confs.append(conf2)

        col_names = ['csf', 'white_matter', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        for c in range(8):
            col_names.append(col_names[c] + '_derivative1')

        for run, conf in enumerate(confs):
            u_conf = conf[col_names]
            u_conf = u_conf.fillna(0)
            out = (func_dir + f'/sub-{sub}_task-movie_run-0{run+1}_formatted_confounds.txt')
            u_conf.to_csv(out, sep='\t', header=False, index=False)


    if file_type == 'surprise' or file_type == 'all':
        if skip_second:
            if first_mov == 'coin':
                run_com(f"cp /home1/09123/ofriend/analysis/movie_scan/coin_ppl_events.txt {func_dir}/sub-{sub}_task-movie_run-01_events.tsv")
                run_com(
                    f"cp /home1/09123/ofriend/analysis/movie_scan/coin_bayes_events.txt {func_dir}/sub-{sub}_task-movie_run-01_events.tsv")
            else:
                run_com(f"cp /home1/09123/ofriend/analysis/movie_scan/jinx_lou_ppl_events.txt {func_dir}/sub-{sub}_task-movie_run-01_events.tsv")
                run_com(
                    f"cp /home1/09123/ofriend/analysis/movie_scan/jinx_lou_bayes_events.txt {func_dir}/sub-{sub}_task-movie_run-01_events.tsv")
        else:
            if first_mov == 'coin':
                run_com(f"cp /home1/09123/ofriend/analysis/movie_scan/coin_ppl_events.txt {func_dir}/sub-{sub}_task-movie_run-01_events.tsv")
                run_com(f"cp /home1/09123/ofriend/analysis/movie_scan/jinx_lou_ppl_events.txt {func_dir}/sub-{sub}_task-movie_run-02_events.tsv")
                run_com(
                    f"cp /home1/09123/ofriend/analysis/movie_scan/coin_bayes_events.txt {func_dir}/sub-{sub}_task-movie_run-01_events.tsv")
                run_com(
                    f"cp /home1/09123/ofriend/analysis/movie_scan/jinx_lou_bayes_events.txt {func_dir}/sub-{sub}_task-movie_run-02_events.tsv")
            else:
                run_com(f"cp /home1/09123/ofriend/analysis/movie_scan/jinx_lou_ppl_events.txt {func_dir}/sub-{sub}_task-movie_run-01_events.tsv")
                run_com(f"cp /home1/09123/ofriend/analysis/movie_scan/coin_ppl_events.txt {func_dir}/sub-{sub}_task-movie_run-02_events.tsv")
                run_com(f"cp /home1/09123/ofriend/analysis/movie_scan/jinx_lou_bayes_events.txt {func_dir}/sub-{sub}_task-movie_run-01_events.tsv")
                run_com(f"cp /home1/09123/ofriend/analysis/movie_scan/coin_bayes_events.txt {func_dir}/sub-{sub}_task-movie_run-02_events.tsv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="main directory where subjects are located (e.g., derivatives/fmriprep/)")
    parser.add_argument("file_type", help="motion, perplexity, bayes, all")
    parser.add_argument("sub", help="subject number e.g. temple001")
    parser.add_argument("first_mov", help="the first movie the participant watched in the counterbalanced order")
    parser.add_argument("--skip_second", action=argparse.BooleanOptionalAction,
                        default=False, help="only process the first movie - boolean")
    args = parser.parse_args()
    main(args.data_dir, args.file_type, args.sub, args.first_mov, args.skip_second)
