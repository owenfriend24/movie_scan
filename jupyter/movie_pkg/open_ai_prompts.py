import json
import pandas as pd
from openai import OpenAI
from argparse import ArgumentParser
import os.path as osp
import os
import json
import random
import re
import numpy as np
import ast

def call_openai_parse(client, model, example_txt, example_out, recall_txt):
    system_prompt = f"""
Your job is to parse a continuous recall transcript into individual events. Your parsing will be scored by *proportion agreement* on held-out data, and must reach **≥ 90%**.  
Below are *complete* example files with recall text (the **input**), and a list of parsed events in order (**the required output**). Use them as your *sole* blueprint for formatting, ordering, and logic.

=== EXAMPLE RECALL TEXT ===
{example_txt}

=== EXAMPLE OUTPUT ===
{example_out}

=== NEW PARSING TASK ===
You must follow these rules exactly:

1. **Identify unique events based on action units**  
      - An `"action unit"` is the smallest meaningful phrase or sentence that captures a subject and an action.
      - Often, an `"action unit"` will include a noun and at least one verb.
      - The only exception to the above rule is when an action is described and a subject is not mentioned but **is** implied. For example, '"he gets up and goes home"` should be parsed into two units: 1) `"he gets up"` 2) `"and goes home"`
      - `"Action units"` can include descriptions of the action. For example, `"he goes home quickly after work ends"` is still one `"action unit`".
      - Do **NOT** removes instances of `"and"`, `"so"`, `"then"`, etc. which precede action units, those are important to organization and thus perfectly suitable starting words to action units.
      
2. **Revise and refine**  
   - After drafting, compare your output to the example’s structure. Revise by tightening confirming that nothing important was removed and every unique action unit has its own line. 

Return all results as a **single valid Python array** using double quotes, e.g.: ["event 1", "event 2", "event 3"]. Do **NOT** return multiple separate objects.

Now map these new recalls:
{recall_txt}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Ready to parse the recalls."}
        ],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    try:
        result = ast.literal_eval(content)
        return result
        
    except Exception as e:
        print("Failed to parse model output. Raw content below:")
        print(content)
        raise e



def call_openai_event_coding(client, model, ground_truth, examples, recall):
    """
    examples: list of dicts, each with keys "ground_truth", "recalls", and "recall_codes"
    """
    examples_str = "\n\n".join(
        f"=== EXAMPLE {i+1} ===\n{json.dumps(example, indent=4)}"
        for i, example in enumerate(examples)
    )
    
    system_prompt = f"""
Your mapping will be scored by *proportion agreement* on held-out data, and must reach **≥ 90%**.  
Below are *complete* example files with ground-truth segments (denoted as 'transcript' in the example file), a set of recalls (the text within the 'recall' fields), and the exact, fully-coded output from human coders (the mapping of 'recalls' to detail_recall or gist_recall for each ground-truth 'transcript'). Use them as your *sole* blueprint for formatting, ordering, and logic.

=== EXAMPLES ===
{examples_str}

=== NEW MAPPING TASK ===
GROUND TRUTH SEGMENTS:
{json.dumps(ground_truth, indent=4)}

You must follow these rules exactly:

1. **Match segments first**  
   • A “match” = the segment’s transcript is literally contained in, or semantically captured by, the recall utterance.
   
2. **Detail-first mapping**  
   a. For each utterance, **attempt** to map it to **one** best-fitting segment:  
      - If the utterance exactly matches, paraphrases, or clearly refers to a single segment → assign to `"detail_recall"` for that segment.
      - A “detail_recall” must be exact. You must be confident that the recall utterance refers **directly** to the event captured within the segment's transcript and **no other event**.
      - If you can confidently choose one segment, **do not** consider gist at all.  
      - Any utterance mapped as a "detail_recall" **cannot** be mapped to any other segment as either a detail_recall or gist_recall

3. **Gist fallback**  
   • Only if **no single** segment can fully capture the utterance, and **two or more contiguous** segments together do → assign to `"gist_recall"` for *all* those contiguous segments.  
   • Gist may **never** span non-adjacent segments. 
   • If matches are non-contiguous, you must choose which contiguous events better fit the utterance or map as individual details.

4. **Unmapped**  
   • Try to map all segments, but if *no* segment (or contiguous block) matches, omit the recall entirely.

5. **Output format & order**  
   • Emit one object per segment, **in the same order** as input.  
   • If a segment has no recalls, both `"detail_recall"` and `"gist_recall"` must be `[]`.  
   • Field names, nesting, and JSON schema must mirror the full example exactly.

6. **Proportion ≥ 90%**  
   - After drafting, compare your output to the example’s structure. Revise by tightening detail matches and removing spurious gist spans.
   - Assign each mapping a confidence score between 0 and 1, 1 meaning you are sure that you have identified the exactly correct mapping. For any utterances where your confidence score is below 0.8, revisit to ensure the utterance can **only** be mapped to the event you mapped it to if it is a "detail_recall".

7. **Double-checking the output**
   - Remember that utterances can only be mapped to **one** ground truth event as a detail recall, so correct detail mapping is **essential**.
   - Iterate through the full JSON file you have created, making **100% sure** that all recalls have been properly mapped. 
   - Validate all detail matches by asking whether the utterance exactly matches and/or paraphrases the ground truth segment. If they are **not** proper matches, remap the utterance to the ground truth segment that results in a proper mapping. Do **not** remap detail recalls to gist recalls, simply ensure that utterances that are coded as detail recalls do not match better with other ground truth events.
   
Return all results as a **single valid JSON array** (e.g., `[ {{...}}, {{...}} ]`). Do **NOT** return multiple separate objects.
Make sure that **every single ground truth annotation is included in your output** in the same order as it appears in the input, whether or not it has a corresponding human recall utterance that has been mapped to it.
Now map these new recalls:
{json.dumps(recall, indent=4)}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Ready to map the recalls."}
        ],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    # Try normal parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Strip code fence if present
        if content.startswith("```"):
            content = re.sub(r"^```[a-z]*\n?", "", content)
            content = re.sub(r"```$", "", content.strip())
    
        # Basic object match — flat-level only
        objects = re.findall(r'\{[^{}]*\}', content)
        if not objects:
            raise ValueError("No JSON objects found in model output.")
    
        wrapped = "[" + ",\n".join(objects) + "]"
        try:
            return json.loads(wrapped)
        except Exception as e:
            raise ValueError(f"Failed to parse wrapped LLM output:\n\n{wrapped}\n\nError: {e}")






def call_openai_event_coding_PRESERVE(client, model, ground_truth, examples, recall):
    """
    examples: list of dicts, each with keys "ground_truth", "recalls", and "recall_codes"
    """
    examples_str = "\n\n".join(
        f"=== EXAMPLE {i+1} ===\n{json.dumps(example, indent=4)}"
        for i, example in enumerate(examples)
    )
    
    system_prompt = f"""
Your mapping will be scored by *proportion agreement* on held-out data, and must reach **≥ 90%**.  
Below are *complete* example files with ground-truth segments (denoted as 'transcript' in the example file), a set of recalls (the text within the 'recall' fields), and the exact, fully-coded output from human coders (the mapping of 'recalls' to detail_recall or gist_recall for each ground-truth 'transcript'). Use them as your *sole* blueprint for formatting, ordering, and logic.

=== EXAMPLES ===
{examples_str}

=== NEW MAPPING TASK ===
GROUND TRUTH SEGMENTS:
{json.dumps(ground_truth, indent=4)}

You must follow these rules exactly:

1. **Match segments first**  
   • A “match” = the segment’s transcript is literally contained in, or semantically captured by, the recall utterance.
   
2. **Detail-first mapping**  
   a. For each utterance, **attempt** to map it to **one** best-fitting segment:  
      - If the utterance exactly matches, paraphrases, or clearly refers to a single segment → assign to `"detail_recall"` for that segment.
      - A “detail_recall” must be exact. You must be confident that the recall utterance refers **directly** to the event captured within the segment's transcript and **no other event**.
      - If you can confidently choose one segment, **do not** consider gist at all.  
      - Any utterance mapped as a "detail_recall" **cannot** be mapped to any other segment as either a detail_recall or gist_recall
      - **PRESERVE THE ORIGINAL ORDER THAT THE RECALLS WERE PROVIDED WHENEVER POSSIBLE, UNLESS IT IS CLEAR THAT SOMEONE IS GOING BACK IN THE STORY IN THEIR RECALL (e.g., 'oh and one thing I forgot from earlier was...')

3. **Gist fallback**  
   • Only if **no single** segment can fully capture the utterance, and **two or more contiguous** segments together do → assign to `"gist_recall"` for *all* those contiguous segments.  
   • Gist may **never** span non-adjacent segments. 
   • If matches are non-contiguous, you must choose which contiguous events better fit the utterance or map as individual details.

4. **Unmapped**  
   • Try to map all segments, but if *no* segment (or contiguous block) matches, omit the recall entirely.

5. **Output format & order**  
   • Emit one object per segment, **in the same order** as input.  
   • If a segment has no recalls, both `"detail_recall"` and `"gist_recall"` must be `[]`.  
   • Field names, nesting, and JSON schema must mirror the full example exactly.

6. **Proportion ≥ 90%**  
   - After drafting, compare your output to the example’s structure. Revise by tightening detail matches and removing spurious gist spans.
   - Assign each mapping a confidence score between 0 and 1, 1 meaning you are sure that you have identified the exactly correct mapping. For any utterances where your confidence score is below 0.8, revisit to ensure the utterance can **only** be mapped to the event you mapped it to if it is a "detail_recall".

7. **Double-checking the output**
   - Remember that utterances can only be mapped to **one** ground truth event as a detail recall, so correct detail mapping is **essential**.
   - Iterate through the full JSON file you have created, making **100% sure** that all recalls have been properly mapped. 
   - Validate all detail matches by asking whether the utterance exactly matches and/or paraphrases the ground truth segment. If they are **not** proper matches, remap the utterance to the ground truth segment that results in a proper mapping. Do **not** remap detail recalls to gist recalls, simply ensure that utterances that are coded as detail recalls do not match better with other ground truth events.
   
Return all results as a **single valid JSON array** (e.g., `[ {{...}}, {{...}} ]`). Do **NOT** return multiple separate objects.
Make sure that **every single ground truth annotation is included in your output** in the same order as it appears in the input, whether or not it has a corresponding human recall utterance that has been mapped to it.
Now map these new recalls:
{json.dumps(recall, indent=4)}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Ready to map the recalls."}
        ],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    # Try normal parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Strip code fence if present
        if content.startswith("```"):
            content = re.sub(r"^```[a-z]*\n?", "", content)
            content = re.sub(r"```$", "", content.strip())
    
        # Basic object match — flat-level only
        objects = re.findall(r'\{[^{}]*\}', content)
        if not objects:
            raise ValueError("No JSON objects found in model output.")
    
        wrapped = "[" + ",\n".join(objects) + "]"
        try:
            return json.loads(wrapped)
        except Exception as e:
            raise ValueError(f"Failed to parse wrapped LLM output:\n\n{wrapped}\n\nError: {e}")

