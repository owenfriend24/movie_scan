#!/usr/bin/env python
#
# generate .txt files without headers for motion confounds or for collector (learning) beta series

from pathlib import Path
import pandas as pd
import os
import argparse


def main(data_dir, file_type, sub):
    out_dir = f'/scratch/09123/ofriend/movie_scan/sub-{sub}/collector_txt'
    os.makedirs(out_dir, exist_ok=True)
    func_dir = data_dir + f'/sub-{sub}/func/'

    if file_type == 'motion' or file_type == 'both':
        conf1 = pd.read_table(func_dir + f'/sub-{sub}_task-collector_run-01_desc-confounds_timeseries.tsv')
        conf2 = pd.read_table(func_dir + f'/sub-{sub}_task-collector_run-02_desc-confounds_timeseries.tsv')
        conf3 = pd.read_table(func_dir + f'/sub-{sub}_task-collector_run-03_desc-confounds_timeseries.tsv')
        conf4 = pd.read_table(func_dir + f'/sub-{sub}_task-collector_run-04_desc-confounds_timeseries.tsv')


        col_names = ['csf', 'white_matter', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                     'framewise_displacement', 'dvars']
        print(f'confounds: {col_names}')
        for c in range(8):
            col_names.append(col_names[c] + '_derivative1')

        confs = [(conf1, 'conf1'), (conf2, 'conf2'), (conf3, 'conf3'), (conf4, 'conf4')]

        for conf, name in confs:
            run = name[-1]
            u_conf = conf[col_names]
            u_conf = u_conf.fillna(0)
            out = (out_dir + f'/sub-{sub}_task-collector_run-0{run}_formatted_confounds.txt')
            u_conf.to_csv(out, sep='\t', header=False, index=False)
            # run += 1

    if file_type == 'collector' or file_type == 'both':
        c1 = pd.read_table(func_dir + f'sub-{sub}_task-collector_run-01_events_fixed.tsv')
        c2 = pd.read_table(func_dir + f'sub-{sub}_task-collector_run-02_events_fixed.tsv')
        c3 = pd.read_table(func_dir + f'sub-{sub}_task-collector_run-03_events_fixed.tsv')
        c4 = pd.read_table(func_dir + f'sub-{sub}_task-collector_run-04_events_fixed.tsv')


        arrs = [(c1, 'c1'), (c2, 'c2'), (c3, 'c3'), (c4, 'c4')]
        for arr, name in arrs:
            run = name[-1]
            for item in range(1, 13, 1):
                triad = (item - 1) // 3 + 1
                position = (item - 1) % 3 + 1
                items = pd.DataFrame(columns=['onset', 'duration', 'weight'])
                ref = arr[(arr['triad'] == triad) & (arr['position'] == position)]
                for index, row in ref.iterrows():
                    items.loc[len(items)] = [float(row['onset']), float(row['duration']), float(1.0)]
                out = out_dir + f'/sub-{sub}_task-collector_run-{run}_item-{item}.txt'
                items.to_csv(out, sep='\t', header=False, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="main directory where subjects are located (e.g., derivatives/fmriprep/)")
    parser.add_argument("file_type", help="motion, collector, or both")
    parser.add_argument("sub", help="subject number e.g. temple001")
    args = parser.parse_args()
    main(args.data_dir, args.file_type, args.sub)
