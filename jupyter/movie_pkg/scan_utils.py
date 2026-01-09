import json
import pandas as pd
import os
import re
import html



base = "/Users/owenfriend/Documents/movie_scan_local/movie_pkg"
study_dir = "/Users/owenfriend/Documents/movie_scan_local"
parse_dir  = "/Users/owenfriend/Documents/movie_scan_local/llm_coding/scan_parsing/"

def get_scan_subs():
    return ['temple056', 'temple057', 'temple058', 'temple059', 'temple060', 'temple061', 'temple063', 'temple064', 'temple065', 'temple066', 'temple068', 'temple069', 'temple070', 'temple071', 'temple072', 'temple073', 'temple074', 'temple075', 'temple076', 'temple079', 'temple082', 'temple083', 'temple084', 'temple085', 'temple087', 'temple088', 'temple089', 'temple090', 'temple091', 'temple092', 'temple093', 'temple094', 'temple095', 'temple096', 'temple097', 'temple098', 'temple099', 'temple103', 'temple105', 'temple106', 'temple107', 'temple108', 'temple109', 'temple110', 'temple111', 'temple112', 'temple113', 'temple114', 'temple115', 'temple116', 'temple120', 'temple122',  'temple124', 'temple117', 'temple119', 'temple121', 'temple123', 'temple125', 'temple126', 'temple128', 'temple129', 'temple130', 'temple131', 'temple132', 'temple135', 'temple136']

def get_subject_movies(subject):
    movies = ['coin', 'jinx_lou']
    if subject in ['temple056', 'temple116', 'temple135']:
        movies.remove('jinx_lou')
    if subject in ['temple107', 'temple113', 'temple130']:
        movies.remove('coin')
    return movies

def get_blank_recall_json(movie):
    with open(f"{base}/blanks/blank_recall_{movie}.json", "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    return loaded_data


def pull_training_event_coding(movie):
    with open(f"{base}/training/example_4_{movie}.json", "r", encoding="utf-8") as f:
        example_1 = json.load(f)
    with open(f"{base}/training/example_5_{movie}.json", "r", encoding="utf-8") as f:
        example_2 = json.load(f)
    with open(f"{base}/training/example_6_{movie}.json", "r", encoding="utf-8") as f:
        example_3 = json.load(f)
    return [example_1, example_2, example_3]


def load_recall_txt(subject, movie):
    s_id = subject[-3:]
    m_id = movie[0:4]
    target_prefix = f"{s_id}_{m_id}"
    for filename in os.listdir(f"{study_dir}/recall_txt/scan"):
        if filename.startswith(target_prefix) and filename.endswith(".txt"):
            full_path = os.path.join(f"{study_dir}/recall_txt/scan", filename)
            with open(full_path, "r", encoding="utf-8", errors = "ignore") as f:
                return f.read()
    raise FileNotFoundError(f"no file found for: {target_prefix}")

def clean_recall_txt(text):
    # Remove bracketed numbers and specific tags
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[(inaudible|exp prompt)\]', '', text, flags=re.IGNORECASE)
    # Remove standalone filler words "um" and "uh"
    text = re.sub(r'\b(um|uh)\b', '', text, flags=re.IGNORECASE)
    # Remove brackets but keep their contents
    text = text.replace('[', '').replace(']', '')
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text




def recall_tsv_to_json(subject, movie, list_column="utterance"):
    df = pd.read_csv(f'{parse_dir}/{subject}/subj-{subject}_{movie}_parsed.tsv', sep="\t")
    result = []
    for index, row in df.iterrows():
        # Clean the utterance
        cleaned = re.sub(r'[.,\"“”\'’]', '', str(row[list_column]))
        result.append({
            "movie": movie,
            "index": index + 1,
            "transcript": cleaned
        })
    return json.dumps(result, indent=4)




def generate_parsing_report(movie_data_list, output_html_path):
    """
    Generate a combined HTML report for multiple movies.

    Parameters:
    - movie_data_list: a list of dicts, one per movie, each with keys:
        'movie', 'recall_segments', 'original_recall_text'
    - output_html_path: where to save the HTML file
    """

    def alternating_color_paragraph(segments):
        colors = ['#3883eb', '#8334eb']
        colored_segments = [
            f"<span style='color:{colors[i % 2]};'>{html.escape(seg.strip())}</span>"
            for i, seg in enumerate(segments) if seg.strip()
        ]
        return " ".join(colored_segments)

    # Build content for all movies
    all_sections = ""
    for i, movie_data in enumerate(movie_data_list):
        movie = movie_data["movie"]
        recall_segments = movie_data["recall_segments"]

        paragraph_text = alternating_color_paragraph(recall_segments)

        section = f"""
        <h1>{html.escape(movie.capitalize())}</h1>

        <h2>Parsed Recall Segments</h2>
        <ol>
            {''.join(f'<li>{html.escape(seg)}</li>' for seg in recall_segments)}
        </ol>

        <h2>Parsed recall - continuous paragraph </h2>
        <div class="highlighted">{paragraph_text}</div>
        """

        if i < len(movie_data_list) - 1:
            section += "<hr style='margin: 40px 0;'>"

        all_sections += section

    # Full HTML wrapper
    html_report = f"""
    <html>
    <head>
        <title>Parsing Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #ffffff;
                color: #000000;
            }}
            h1 {{ color: #005f86; margin-top: 60px; }}
            h2 {{ color: #333; margin-top: 30px; }}
            .highlighted {{
                margin-top: 20px;
                padding: 10px;
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                white-space: pre-wrap;
            }}
            ol li {{ margin-bottom: 8px; }}
        </style>
    </head>
    <body>
        {all_sections}
    </body>
    </html>
    """

    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_report)

    print(f"HTML report saved to: {output_html_path}")

def json_to_csv(data, csv_file_path, subject):
    if isinstance(data, list) and isinstance(data[0], list):
        data = data[0]

    used_detail_recalls = set()
    rows = []

    for entry in data:
        # Deduplicate detail recalls across all rows
        raw_details = [item['recall'] for item in entry.get('detail_recall', [])]
        new_details = [r for r in raw_details if r not in used_detail_recalls]
        used_detail_recalls.update(new_details)

        # Include all gist recalls
        raw_gists = [item['recall'] for item in entry.get('gist_recall', [])]

        row = {
            'subject': subject,
            'movie': entry.get('movie', ''),
            'ground_truth_index': entry.get('index', ''),
            'start_time': entry.get('start_time', ''),
            'end_time': entry.get('end_time', ''),
            'transcript': entry.get('ground_truth_event', ''),
            'detail_recall': '; '.join(new_details),
            'gist_recall': '; '.join(raw_gists)
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df['ground_truth_index'] = pd.to_numeric(df['ground_truth_index'], errors='coerce')
    df['ground_truth_index'] = df['ground_truth_index'] - (df['ground_truth_index'].values[0] - 1)

    # Write to TSV (tab-separated)
    df.to_csv(csv_file_path, sep='\t', index=False)



import pandas as pd
import html

def generate_coding_report(movie_list, output_html_path):
    def get_coded_mappings(csv_path):
        df = pd.read_csv(csv_path, sep="\t")
        detail_map = {}
        gist_map = {}

        for _, row in df.iterrows():
            if pd.isna(row['transcript']):
                continue
            gt = row['transcript'].strip().lower()
            if pd.notna(row.get('detail_recall')):
                for part in row['detail_recall'].split(";"):
                    for frag in part.split("|"):
                        cleaned = frag.strip().lower()
                        if cleaned:
                            detail_map.setdefault(cleaned, []).append(gt)
            if pd.notna(row.get('gist_recall')):
                for part in row['gist_recall'].split(";"):
                    for frag in part.split("|"):
                        cleaned = frag.strip().lower()
                        if cleaned:
                            gist_map.setdefault(cleaned, []).append(gt)

        ground_truth_events = df['transcript'].dropna().tolist()
        return detail_map, gist_map, ground_truth_events

    def match_gt_to_recall(gt_events, recall_list, detail_map, gist_map):
        matched_rows = []
        used_detail_utterances = set()
        matched_gist_utterances = set()

        # Invert gist_map: gt -> list of phrases
        inverted_gist_map = {}
        for phrase, gt_list in gist_map.items():
            for gt in gt_list:
                inverted_gist_map.setdefault(gt, []).append(phrase)

        # Invert detail_map: gt -> list of phrases
        inverted_detail_map = {}
        for phrase, gt_list in detail_map.items():
            for gt in gt_list:
                inverted_detail_map.setdefault(gt, []).append(phrase)

        for idx, gt in enumerate(gt_events, 1):
            gt_lower = gt.strip().lower()
            detail_matches = []
            gist_matches = []

            # Match detail (can be multiple utterances)
            if gt_lower in inverted_detail_map:
                for phrase in inverted_detail_map[gt_lower]:
                    for i, utt in enumerate(recall_list):
                        if utt in used_detail_utterances:
                            continue
                        if phrase in utt.lower():
                            detail_matches.append((utt, 'green', i + 1))
                            used_detail_utterances.add(utt)

            # Match gist (can be multiple utterances)
            if gt_lower in inverted_gist_map:
                for phrase in inverted_gist_map[gt_lower]:
                    for i, utt in enumerate(recall_list):
                        if phrase in utt.lower():
                            gist_matches.append((utt, i + 1))
                            matched_gist_utterances.add(utt)

            # Add matches
            if detail_matches:
                utts = [f"({idx_}) {utt}" for utt, _, idx_ in detail_matches]
                combined = ", ".join(utts)
                matched_rows.append((idx, gt, combined, 'green'))
            elif gist_matches:
                gist_strs = [f"({i}) {u}" for u, i in gist_matches]
                combined = ", ".join(gist_strs)
                matched_rows.append((idx, gt, combined, 'purple'))
            else:
                matched_rows.append((idx, gt, "", None))

        # Exclude used utterances from uncoded list
        used_all = used_detail_utterances.union(matched_gist_utterances)
        uncoded = [(i + 1, utt) for i, utt in enumerate(recall_list) if utt not in used_all]

        return matched_rows, uncoded

    all_sections = ""
    for movie_data in movie_list:
        movie = movie_data["movie"]
        recall_list = [utt.strip() for utt in movie_data["recall_list"] if utt.strip()]
        csv_path = movie_data["coded_csv_path"]

        detail_map, gist_map, gt_events = get_coded_mappings(csv_path)
        matched, uncoded = match_gt_to_recall(gt_events, recall_list, detail_map, gist_map)

        rows_html = ""
        for gt_idx, gt, utt, color in matched:
            if utt:
                utt_html = f"<span style='color:{color};'>{html.escape(utt, quote=False)}</span>"
            else:
                utt_html = ""

            rows_html += f"""
                <tr style="border-bottom: 1px dotted #ccc;">
                    <td style="padding:6px 4px; vertical-align:top; text-align:left; width:50%;"><strong>{gt_idx}.</strong> {html.escape(gt)}</td>
                    <td style="padding:6px 4px; vertical-align:top; text-align:left; width:50%;">{utt_html}</td>
                </tr>
            """

        uncoded_html = ""
        for idx, utt in uncoded:
            uncoded_html += f"<p style='color:red; margin-bottom:4px;'>{html.escape(f'({idx}) {utt}')}</p>"

        section = f"""
        <h2 style="font-family:Arial; font-size:18px;">{html.escape(movie.capitalize())}</h2>
        <table style="width:100%; table-layout:fixed; font-family:Arial; font-size:14px; border-collapse:collapse; margin-bottom:20px;">
            <thead>
                <tr style="border-bottom:2px solid #888;">
                    <th style="text-align:left; width:50%;">Ground Truth Event</th>
                    <th style="text-align:left; width:50%;">Matched Utterance (Green-Detail; Purple-Gist)</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        <div style="font-family:Arial; font-size:14px; margin-top:10px;">
            <strong>Uncoded Utterances:</strong>
            {uncoded_html}
        </div>
        <hr style="margin: 40px 0;">
        """

        all_sections += section

    html_content = f"""
    <html>
    <head>
        <title>Event Coding Report</title>
        <meta charset="UTF-8">
    </head>
    <body style="padding: 20px;">
        {all_sections}
    </body>
    </html>
    """
    html_content = html_content.replace('&#x27;', "'")
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML report written to: {output_html_path}")
