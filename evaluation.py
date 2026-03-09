import os
import json
import argparse
import re
import csv
from collections import defaultdict
import jiwer

def clean_phonemes(text):
    if not text:
        return ""

    #remove stress markers, length markers, punctuation, and special symbols
    cleaned = re.sub(r'[ˈˌː.,?!;:()\[\]]', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned)

    return cleaned.strip()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Phoneme Error Rate (PER).")
    parser.add_argument("--manifest", type=str, required=True, help="Path to the prediction results jsonl file")
    parser.add_argument("--out_csv", type=str, required=True, help="Path to the output evaluation report metrics.csv")
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"File not found: {args.manifest}")
        return

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # structure: {snr_db: {"references": [], "hypotheses": []}}
    snr_groups = defaultdict(lambda: {"references": [], "hypotheses": []})
    total_processed = 0

    with open(args.manifest, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            if not line.strip():
                continue
            
            record = json.loads(line)
            raw_ref = record.get('ref_pho', '')
            raw_hyp = record.get('hyp_pho', '')
            snr_db = record.get('snr_db', 'clean') 

            clean_ref = clean_phonemes(raw_ref)
            clean_hyp = clean_phonemes(raw_hyp)

            if not clean_ref:
                continue

            snr_groups[snr_db]["references"].append(clean_ref)
            snr_groups[snr_db]["hypotheses"].append(clean_hyp)
            total_processed += 1

    if total_processed == 0:
        print("No valid data found for evaluation.")
        return

    print(f"\nSuccessfully loaded {total_processed} items, starting PER calculation...\n")
    print("-" * 50)
    print(f"{'SNR (dB)':<15} | {'Samples':<10} | {'PER (%)':<10}")
    print("-" * 50)

    csv_rows = []

    sorted_snrs = sorted([k for k in snr_groups.keys() if isinstance(k, (int, float))], reverse=True)
    
    if 'clean' in snr_groups:
        sorted_snrs.insert(0, 'clean')

    for snr in sorted_snrs:
        refs = snr_groups[snr]["references"]
        hyps = snr_groups[snr]["hypotheses"]
        
        error_rate = jiwer.wer(refs, hyps)
        per_percentage = round(error_rate * 100, 2)
        
        sample_count = len(refs)
        print(f"{str(snr):<15} | {sample_count:<10} | {per_percentage:>6}%")
        
        csv_rows.append({
            "snr_db": snr,
            "sample_count": sample_count,
            "per_percentage": per_percentage
        })

    print("-" * 50)

    with open(args.out_csv, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["snr_db", "sample_count", "per_percentage"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nEvaluation report: {args.out_csv}")

if __name__ == "__main__":
    main()