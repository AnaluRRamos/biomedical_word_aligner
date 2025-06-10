# LinguAligner/biomedical_aligner.py

import pandas as pd
import ast 
from LinguAligner.pipeline import AlignmentPipeline
import spacy 
from transformers import logging 
from collections import defaultdict

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

import ast
from LinguAligner.pipeline import AlignmentPipeline
import spacy
from transformers import logging
from collections import defaultdict
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
CSV_FILE_PATH = "/Users/mac/Desktop/LinguAligner/your_data.csv"
print("🔗 Initializing AlignmentPipeline...")
alignment_config = {
    "pipeline": ["M_Trans", "lemma", "word_aligner", "gestalt", "leveinstein"],
    "spacy_model": "pt_core_news_lg",
    "WAligner_model": "bert-base-multilingual-uncased",
}
pipeline = AlignmentPipeline(config=alignment_config)
print("✅ AlignmentPipeline initialized.")
global_results_by_label = defaultdict(lambda: {
    'total_terms': 0,
    'found_in_ref': 0,
    'found_in_pred': 0
})
all_term_details = []
print(f"\n🚀 Starting alignment and checking translations from: {CSV_FILE_PATH}...\n")
try:
    df = pd.read_csv(CSV_FILE_PATH)
except FileNotFoundError:
    print(f"Error: CSV file not found at {CSV_FILE_PATH}. Please check the path.")
    exit()
for index, row in df.iterrows():
    en_sentence = row['English']
    pt_ref = row['Reference']
    pt_pred = row['Prediction']
    try:
        ner_spacy_str = row['NER_spacy']
        ner_kushtrim_str = row['NER_Kushtrim']
        spacy_labeled_ents = ast.literal_eval(ner_spacy_str) if pd.notna(ner_spacy_str) and ner_spacy_str.strip() else []
        hf_labeled_ents = ast.literal_eval(ner_kushtrim_str) if pd.notna(ner_kushtrim_str) and ner_kushtrim_str.strip() else []
    except (ValueError, SyntaxError) as e:
        print(f"Warning: Could not parse NER tags for row {index}. Skipping row. Error: {e}")
        continue
    combined_labeled_entities_dict = {}
    for term, label in spacy_labeled_ents:
        combined_labeled_entities_dict[term] = label
    for term, label in hf_labeled_ents:
        if term not in combined_labeled_entities_dict:
            combined_labeled_entities_dict[term] = label
    combined_labeled_entities = list(combined_labeled_entities_dict.items())
    if not combined_labeled_entities:
        continue
    for ent, label in combined_labeled_entities:
        global_results_by_label[label]['total_terms'] += 1
        res_ref, span_ref = pipeline.align_annotation(
            src_sent=en_sentence,
            src_ann=ent,
            tgt_sent=pt_ref,
            trans_ann=ent,
            lookupTable=None
        )
        res_pred, span_pred = pipeline.align_annotation(
            src_sent=en_sentence,
            src_ann=ent,
            tgt_sent=pt_pred,
            trans_ann=ent,
            lookupTable=None
        )
        found_in_ref = bool(res_ref)
        found_in_pred = bool(res_pred)
        if found_in_ref:
            global_results_by_label[label]['found_in_ref'] += 1
        if found_in_pred:
            global_results_by_label[label]['found_in_pred'] += 1
        all_term_details.append({
            "row_id": row.get('ID', index),
            "english_sentence": en_sentence,
            "portuguese_reference": pt_ref,
            "portuguese_prediction": pt_pred,
            "term": ent,
            "label": label,
            "reference_match": {
                "found": found_in_ref,
                "match_text": res_ref,
                "span": span_ref
            },
            "prediction_match": {
                "found": found_in_pred,
                "match_text": res_pred,
                "span": span_pred
            }
        })
    if (index + 1) % 100 == 0:
        print(f"Processed {index + 1} rows...")
print("\n All sentences processed.\n")
print("--- Final Alignment Statistics by NER Label ---")
print("{:<25} {:<15} {:<15} {:<15} {:<15}".format(
    "NER Label", "Total Terms", "Found in Ref", "Found in Pred", "Pred Success %"
))
print("-" * 85)
sorted_labels = sorted(global_results_by_label.keys())
for label in sorted_labels:
    data = global_results_by_label[label]
    total = data['total_terms']
    found_ref = data['found_in_ref']
    found_pred = data['found_in_pred']
    pred_success_percent = (found_pred / total * 100) if total > 0 else 0
    print("{:<25} {:<15} {:<15} {:<15} {:<14.1f}%".format(
        label, total, found_ref, found_pred, pred_success_percent
    ))
print("-" * 85)
# IMPORTANT: Update this path to your CSV file
CSV_FILE_PATH = "/Users/mac/Desktop/LinguAligner/your_data.csv" 
# Ensure 'your_data.csv' is replaced with the actual name and path of your file.

# --- Initialize Alignment Pipeline (only once) ---
print("🔗 Initializing AlignmentPipeline...")
alignment_config = {
    # M_Trans is still in the pipeline, but since lookupTable is None by default,
    # it won't trigger unless you specifically provide a lookupTable for a term.
    # The 'word_aligner' will be the main cross-lingual method here.
    "pipeline": ["M_Trans", "lemma", "word_aligner", "gestalt", "leveinstein"],
    "spacy_model": "pt_core_news_lg", # This model is used by the aligner itself
    "WAligner_model": "bert-base-multilingual-uncased",
}
pipeline = AlignmentPipeline(config=alignment_config)
print("✅ AlignmentPipeline initialized.")

# --- Global Accumulators for Statistics ---
# This defaultdict will store counts across all sentences
global_results_by_label = defaultdict(lambda: {
    'total_terms': 0,
    'found_in_ref': 0,
    'found_in_pred': 0
})

# List to store detailed results for inspection (optional, can be very large for many texts)
all_term_details = []

# --- Main Processing Logic ---
print(f"\n🚀 Starting alignment and checking translations from: {CSV_FILE_PATH}...\n")

try:
    df = pd.read_csv(CSV_FILE_PATH)
except FileNotFoundError:
    print(f"Error: CSV file not found at {CSV_FILE_PATH}. Please check the path.")
    exit()

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    en_sentence = row['English']
    pt_ref = row['Reference']
    pt_pred = row['Prediction']
    
    # Parse NER tags from string representation to actual Python lists of tuples
    try:
        ner_spacy_str = row['NER_spacy']
        ner_kushtrim_str = row['NER_Kushtrim']
        
        # Handle NaN values for NER columns
        spacy_labeled_ents = ast.literal_eval(ner_spacy_str) if pd.notna(ner_spacy_str) and ner_spacy_str.strip() else []
        hf_labeled_ents = ast.literal_eval(ner_kushtrim_str) if pd.notna(ner_kushtrim_str) and ner_kushtrim_str.strip() else []
        
    except (ValueError, SyntaxError) as e:
        print(f"Warning: Could not parse NER tags for row {index}. Skipping row. Error: {e}")
        continue # Skip this row if NER parsing fails

    # Combine and deduplicate entities, as before
    combined_labeled_entities_dict = {}
    for term, label in spacy_labeled_ents:
        combined_labeled_entities_dict[term] = label
    for term, label in hf_labeled_ents:
        if term not in combined_labeled_entities_dict:
            combined_labeled_entities_dict[term] = label

    combined_labeled_entities = list(combined_labeled_entities_dict.items())

    # Skip row if no entities found
    if not combined_labeled_entities:
        # print(f"No biomedical entities found for row {index}. Skipping alignment for this row.")
        continue

    # Process each identified term
    for ent, label in combined_labeled_entities:
        global_results_by_label[label]['total_terms'] += 1

        # Align in Reference sentence
        res_ref, span_ref = pipeline.align_annotation(
            src_sent=en_sentence,
            src_ann=ent,
            tgt_sent=pt_ref,
            trans_ann=ent, # Still pass English term, aligner finds Portuguese equivalent
            lookupTable=None
        )
        
        # Align in Prediction sentence
        res_pred, span_pred = pipeline.align_annotation(
            src_sent=en_sentence,
            src_ann=ent,
            tgt_sent=pt_pred,
            trans_ann=ent,
            lookupTable=None
        )

        found_in_ref = bool(res_ref)
        found_in_pred = bool(res_pred)

        if found_in_ref:
            global_results_by_label[label]['found_in_ref'] += 1
        if found_in_pred:
            global_results_by_label[label]['found_in_pred'] += 1

        # Store detailed results for this term (optional, for debugging/deep dive)
        all_term_details.append({
            "row_id": row.get('ID', index), # Use 'ID' column if present, else row index
            "english_sentence": en_sentence,
            "portuguese_reference": pt_ref,
            "portuguese_prediction": pt_pred,
            "term": ent,
            "label": label,
            "reference_match": {
                "found": found_in_ref,
                "match_text": res_ref,
                "span": span_ref
            },
            "prediction_match": {
                "found": found_in_pred,
                "match_text": res_pred,
                "span": span_pred
            }
        })
    
    # Optional: Print progress for long runs
    if (index + 1) % 100 == 0:
        print(f"Processed {index + 1} rows...")

print("\n✅ All sentences processed.\n")

# --- Final Statistics and Table ---
print("--- Final Alignment Statistics by NER Label ---")
print("{:<25} {:<15} {:<15} {:<15} {:<15}".format(
    "NER Label", "Total Terms", "Found in Ref", "Found in Pred", "Pred Success %"
))
print("-" * 85)

sorted_labels = sorted(global_results_by_label.keys())

for label in sorted_labels:
    data = global_results_by_label[label]
    total = data['total_terms']
    found_ref = data['found_in_ref']
    found_pred = data['found_in_pred']
    
    pred_success_percent = (found_pred / total * 100) if total > 0 else 0
    
    print("{:<25} {:<15} {:<15} {:<15} {:<14.1f}%".format(
        label, total, found_ref, found_pred, pred_success_percent
    ))

print("-" * 85)

# Optional: Save detailed results to a JSON file
# import json
# output_json_path = "/Users/mac/Desktop/LinguAligner/detailed_alignment_results.json"
# with open(output_json_path, 'w', encoding='utf-8') as f:
#     json.dump(all_term_details, f, ensure_ascii=False, indent=2)
# print(f"\nDetailed results saved to: {output_json_path}")