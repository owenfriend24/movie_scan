#!/usr/bin/env python

import os
import pandas as pd
import argparse
from temple_utils import get_age_groups

def aggregate_tsnr(data_dir, subjects, masktype, task):

    all_data = []  # List to store per-subject data

    for subject in subjects:
        print(subject)
        tsnr_dir = os.path.join(data_dir, f"sub-{subject}", "func", "tsnr")

        # Load tSNR by ROI
        roi_csv_path = os.path.join(tsnr_dir, f"tsnr_values_{masktype}_{task}.csv")
        if os.path.exists(roi_csv_path):
            df_roi = pd.read_csv(roi_csv_path)
            df_roi["subject"] = subject  # Add subject column

            # Clean up mask names by keeping only the filename (last part of the path)
            df_roi["mask"] = df_roi["mask"].apply(lambda x: os.path.basename(x).replace(".nii.gz", ""))
        else:
            print(f"Warning: Missing ROI CSV for subject {subject}")
            continue

        # Ensure "run" is an integer in both DataFrames for merging
        df_roi["run"] = df_roi["run"].astype(int)

        # Reorder columns to have "subject" first
        df_combined = df_roi[["subject", "run", "mask", "tsnr", "nvoxs"]]

        all_data.append(df_combined)

    # Concatenate all subject data
    if all_data:
        df_final = pd.concat(all_data, ignore_index=True)
        output_csv = os.path.join(data_dir, f"tsnr_aggregated_{masktype}_{task}.csv")
        df_final.to_csv(output_csv, index=False)
        print(f"Saved aggregated tSNR data to {output_csv}")

        return df_final
    else:
        print("No valid data found for aggregation.")
        return None

def main(data_dir, masktype, task):
    subjects = get_age_groups.get_all_subjects()
    aggregate_tsnr(data_dir, subjects, masktype, task)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="where folders containing .txt/.csv files are stored (i.e. $CORR)")
    parser.add_argument("masktype", help="mask name e.g., b_hip_subregions, hip_subfields, lat_hip_subregions, etc.")
    parser.add_argument("task", help="collector or movie")
    args = parser.parse_args()
    main(args.data_dir, args.masktype, args.task)
