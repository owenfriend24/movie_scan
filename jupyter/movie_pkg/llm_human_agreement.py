import pandas as pd
from difflib import SequenceMatcher


def get_reliability_subjects():
    subjects = ['057', '059', '060', '061', '063', '064', '065',
            '066', '068', '082', '085', '090', '091', '094']
    return subjects

def get_rel_subject_movies(subject):
    movies = ['coin', 'jinx_lou']
    if subject in ['059']: # only one set of codes
        movies.remove('jinx_lou')
    if subject in ['066', '082']:
        movies.remove('coin')

    return movies

def get_rel_subject_ages():
    return [20.61, 20.26, 10.07, 11.29, 10.72, 7.36, 9.45, 9.97, 9.76, 11.37, 11.46, 12.52, 11.28, 8.38]


def _fuzzy_match(detail, comparison_set, threshold=0.8):
    for other in comparison_set:
        ratio = SequenceMatcher(None, detail, other).ratio()
        if ratio >= threshold:
            return True
    return False

def extract_recall_sets_csv(csv_path, sep=",", detail_prefix="detail_recall_", single_column=False, verbose=False):
    df = pd.read_csv(csv_path, sep=sep)
    
    if single_column:
        detail_cols = ["detail_recall"]
    else:
        detail_cols = [col for col in df.columns if col.startswith(detail_prefix)]

    all_recall_sets = []
    for i, row in df.iterrows():
        event_recall = set()
        for col in detail_cols:
            val = row.get(col, "")
            if pd.notna(val):
                val_clean = str(val).strip()
                if val_clean:
                    # Split on semicolon for TSV (single column) inputs
                    parts = val_clean.split(";") if single_column else [val_clean]
                    for p in parts:
                        part = p.strip()
                        if part:
                            event_recall.add(part)
        all_recall_sets.append(event_recall)
    return all_recall_sets



def compute_proportion_agreement_csv_tsv(file1, file2, verbose=False, flexible_match=False, fuzzy=False):
    # file1 = CSV with multiple detail_recall columns
    # file2 = TSV with single detail_recall column
    recalls1 = extract_recall_sets_csv(file1, sep=",", single_column=False, verbose=verbose)
    recalls2 = extract_recall_sets_csv(file2, sep="\t", single_column=True, verbose=verbose)

    total_events = max(len(recalls1), len(recalls2))
    agreements = 0

    for i in range(total_events):
        r1 = recalls1[i] if i < len(recalls1) else set()
        r2_current = recalls2[i] if i < len(recalls2) else set()

        if flexible_match:
            r2_prev = recalls2[i - 1] if i > 0 else set()
            r2_next = recalls2[i + 1] if i + 1 < len(recalls2) else set()
            r2_combined = r2_current | r2_prev | r2_next
        else:
            r2_combined = r2_current

        if verbose:
            print(f"Event {i}:")
            print(f"  File1: {r1}")
            print(f"  File2: {r2_combined} {'(including prev/next)' if flexible_match else ''}")

        if not r1 and not r2_current:
            agreements += 1
            if verbose: print("  → Agreement (both empty)")
        elif any((_fuzzy_match(d, r2_combined) if fuzzy else d in r2_combined) for d in r1):
            agreements += 1
            if verbose: print("  → Agreement (shared recall)")
        else:
            if verbose: print("  → Disagreement")

    if verbose:
        print(f"\nAgreement count: {agreements} / {total_events}")

    return agreements / total_events if total_events > 0 else 1.0

def compute_proportion_agreement_csv(file1, file2, verbose=False, flexible_match=False, fuzzy=False):
    recalls1 = extract_recall_sets_csv(file1, verbose=verbose)
    recalls2 = extract_recall_sets_csv(file2, verbose=verbose)

    total_events = max(len(recalls1), len(recalls2))
    agreements = 0

    for i in range(total_events):
        r1 = recalls1[i] if i < len(recalls1) else set()
        r2_current = recalls2[i] if i < len(recalls2) else set()

        # include previous and next if flexible match is on
        if flexible_match:
            r2_prev = recalls2[i - 1] if i > 0 else set()
            r2_next = recalls2[i + 1] if i + 1 < len(recalls2) else set()
            r2_combined = r2_current | r2_prev | r2_next
        else:
            r2_combined = r2_current

        if verbose:
            print(f"Event {i}:")
            print(f"  File1: {r1}")
            print(f"  File2: {r2_combined} {'(including prev/next)' if flexible_match else ''}")

        # Agreement criteria
        if not r1 and not r2_current:
            agreements += 1
            if verbose: print("  → Agreement (both empty)")
        elif any((_fuzzy_match(d, r2_combined) if fuzzy else d in r2_combined) for d in r1):
            agreements += 1
            if verbose: print("  → Agreement (shared recall)")
        else:
            if verbose: print("  → Disagreement")

    if verbose:
        print(f"\nAgreement count: {agreements} / {total_events}")

    return agreements / total_events if total_events > 0 else 1.0

import pandas as pd
import numpy as np
import json
import re
from scipy.stats import pearsonr, ttest_1samp
from math import atanh, tanh
from pathlib import Path

# Function to clean recall text
def clean(text):
    return re.sub(r'[.,\"“”\'’]', '', str(text)).strip().lower()

# Updated correlation function
def mismatch_correlation(human_file1, human_file2, llm_file, verbose=False):
    def extract_human_recall_sets(csv_path):
        df = pd.read_csv(csv_path)
        recall_sets = []
        for _, row in df.iterrows():
            recalls = set()
            for col in df.columns:
                if col.startswith("detail_recall_") and pd.notna(row[col]):
                    cleaned = clean(row[col])
                    if cleaned:
                        recalls.add(cleaned)
            recall_sets.append(recalls)
        return recall_sets

    def extract_llm_recall_sets(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        recall_sets = []
        for event in data:
            recalls = set()
            for item in event.get("detail_recall", []):
                cleaned = clean(item.get("recall", ""))
                if cleaned:
                    recalls.add(cleaned)
            recall_sets.append(recalls)
        return recall_sets

    def get_mismatch_vector(sets_a, sets_b):
        n = max(len(sets_a), len(sets_b))
        vector = np.zeros(n, dtype=int)
        for i in range(n):
            r1 = sets_a[i] if i < len(sets_a) else set()
            r2 = sets_b[i] if i < len(sets_b) else set()
            if not r1 and not r2:
                continue
            if r1 & r2:
                continue
            vector[i] = 1
        return vector

    # Extract recall sets
    recall_human1 = extract_human_recall_sets(human_file1)
    recall_human2 = extract_human_recall_sets(human_file2)
    recall_llm    = extract_llm_recall_sets(llm_file)

    # Compute mismatch vectors
    v_human = get_mismatch_vector(recall_human1, recall_human2)
    v_llm   = get_mismatch_vector(recall_human1, recall_llm)

    min_len = min(len(v_human), len(v_llm))
    if min_len == 0:
        return {"correlation": None, "p_value": None, "n_events": 0, "fisher_z": None}

    v_human = v_human[:min_len]
    v_llm   = v_llm[:min_len]

    # Skip if either vector is constant
    if np.all(v_human == v_human[0]) or np.all(v_llm == v_llm[0]):
        if verbose:
            print("One of the vectors is constant — skipping correlation.")
        return {"correlation": None, "p_value": None, "n_events": min_len, "fisher_z": None}

    # Compute correlation
    corr, pval = pearsonr(v_human, v_llm)
    fisher_z = atanh(corr) if abs(corr) < 1 else np.inf

    return {
        "correlation": corr,
        "p_value": pval,
        "n_events": min_len,
        "fisher_z": fisher_z
    }


def compute_human_human_mismatch_distance(csv1_path, csv2_path, verbose=False):
    def clean(text):
        return re.sub(r'[.,\"“”\'’]', '', str(text)).strip().lower()

    def extract_utterance_to_event_index(csv_path):
        df = pd.read_csv(csv_path)
        utt_map = {}
        for idx, row in df.iterrows():
            for col in df.columns:
                if col.startswith("detail_recall_") and pd.notna(row[col]):
                    utt = clean(row[col])
                    if utt:
                        utt_map[utt] = idx
        return utt_map

    coder1_map = extract_utterance_to_event_index(csv1_path)
    coder2_map = extract_utterance_to_event_index(csv2_path)

    shared_utts = set(coder1_map.keys()) & set(coder2_map.keys())
    mismatches = []
    for utt in shared_utts:
        idx1 = coder1_map[utt]
        idx2 = coder2_map[utt]
        if idx1 != idx2:
            distance = abs(idx1 - idx2)
            mismatches.append((utt, idx1, idx2, distance))
            if verbose:
                print(f"Mismatch: '{utt}' → coder1: {idx1}, coder2: {idx2}, distance: {distance}")

    avg_distance = sum(d for _, _, _, d in mismatches) / len(mismatches) if mismatches else 0.0

    if verbose:
        print(f"Matched utterances: {len(shared_utts)}")
        print(f"Mismatches: {len(mismatches)}")
        print(f"Average mismatch distance: {avg_distance:.2f}")

    return avg_distance

def compute_human_llm_mismatch_distance(human_csv_path, llm_json_path, verbose=False):
    def clean(text):
        # Remove punctuation and normalize whitespace
        return re.sub(r'[.,\"“”\'’]', '', str(text)).strip().lower()

    # Load human utterances
    human_df = pd.read_csv(human_csv_path)
    human_map = {}
    for idx, row in human_df.iterrows():
        for col in human_df.columns:
            if col.startswith("detail_recall_") and pd.notna(row[col]):
                utt = clean(row[col])
                if utt:
                    human_map[utt] = idx  # event index

    # Load LLM utterances
    with open(llm_json_path, 'r') as f:
        llm_data = json.load(f)

    llm_map = {}
    for idx, event in enumerate(llm_data):
        for item in event.get("detail_recall", []):
            utt = clean(item.get("recall", ""))
            if utt:
                llm_map[utt] = idx

    # Compare shared utterances
    shared_utts = set(human_map.keys()) & set(llm_map.keys())
    mismatches = []
    for utt in shared_utts:
        idx1 = human_map[utt]
        idx2 = llm_map[utt]
        if idx1 != idx2:
            distance = abs(idx1 - idx2)
            mismatches.append((utt, idx1, idx2, distance))
            if verbose:
                print(f"Mismatch: '{utt}' → human: {idx1}, llm: {idx2}, distance: {distance}")

    avg_distance = sum(d for _, _, _, d in mismatches) / len(mismatches) if mismatches else 0.0

    if verbose:
        print(f"Matched utterances: {len(shared_utts)}")
        print(f"Mismatches: {len(mismatches)}")
        print(f"Average mismatch distance: {avg_distance:.2f}")

    return avg_distance
