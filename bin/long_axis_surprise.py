#!/usr/bin/env python

import os
import subprocess
from pathlib import Path
import argparse
import pandas as pd


def run_com(command):
    subprocess.run(command, check=True)


def get_mean_pe(img):
    result = subprocess.run(
        ["fslstats", img, "-M"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return float(result.stdout.strip())


def main(sub):
    output_csv = "/scratch/09123/ofriend/movie_scan/surprise_long_axis_new.csv"

    masks = [f"/scratch/09123/ofriend/movie_scan/hip_slices/final_slices/slice_y{mask}.nii.gz"
             for mask in range(43, 60)]

    one_movie_subs = ["temple056", "temple063", "temple107", "temple113", "temple116", "temple123"]

    rows = []

    for model in ["ppl", "bayes"]:
        if sub == 'temple123':
            r = 2
        else:
            r = 1
        for run in [1, 2] if sub not in one_movie_subs else [r]:
            base_pe = f"/scratch/09123/ofriend/movie_scan/sub-{sub}/{model}_out_run{run}.feat/stats/cope1"

            for mask in masks:
                masked_pe = f"{base_pe}_{os.path.basename(mask)}"

                run_com([
                    "fslmaths",
                    f"{base_pe}.nii.gz",
                    "-mas",
                    mask,
                    masked_pe
                ])
                mean_pe = get_mean_pe(masked_pe)
                rows.append([sub, run, model, os.path.basename(mask), mean_pe])

    df = pd.DataFrame(rows, columns=["subject", "run", "model", "mask", "mean_pe"])

    if os.path.exists(output_csv):
        df.to_csv(output_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sub", help="subject number; include full templeXXX")
    args = parser.parse_args()
    main(args.sub)
