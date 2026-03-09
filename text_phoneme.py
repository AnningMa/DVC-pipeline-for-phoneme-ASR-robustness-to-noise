import os
import json
import subprocess
import soundfile as sf
import argparse

def get_phonemes(text):
    result = subprocess.run(
        ['espeak-ng', '-q', '-x', text],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout.strip()

def main():
    parser = argparse.ArgumentParser(description="Prepare manifest for audio dataset.")
    parser.add_argument("--lang", type=str, required=True, help="Language code (e.g., 'en')")
    args = parser.parse_args()
    
    lang = args.lang
    wav_dir = f'data/raw/{lang}/wav'
    manifest_dir = f'data/manifests/{lang}'
    
    os.makedirs(manifest_dir, exist_ok=True)
    final_manifest_path = os.path.join(manifest_dir, 'clean.jsonl')
    temp_manifest_path = final_manifest_path + '.tmp'
    
    processed_id = set()
    
    if os.path.exists(final_manifest_path):
        with open(final_manifest_path, 'r', encoding='utf-8') as f_in:
            with open(temp_manifest_path, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    if not line.strip():
                        continue
                    f_out.write(line)
                    try:
                        record = json.loads(line)
                        processed_id.add(record['utt_id'])
                    except json.JSONDecodeError:
                        continue
    else:
        open(temp_manifest_path, 'w', encoding='utf-8').close()
    
    newly_processed_count = 0
    
    with open(temp_manifest_path, 'a', encoding='utf-8') as f_out:
        for filename in os.listdir(wav_dir):
            if not filename.endswith('.flac'):
                continue
                
            stem = os.path.splitext(filename)[0]
            utt_id = f"{lang}_{stem}"
            if utt_id in processed_id:
                continue
            
            wav_path = os.path.join(wav_dir, filename)
            meta_path = os.path.join(wav_dir, f"{stem}.metadata.json")
            
            if not os.path.exists(meta_path):
                print(f"Can't find metadata file for {filename}")
                continue

            with open(meta_path, 'r', encoding='utf-8') as f_meta:
                try:
                    metadata = json.load(f_meta)
                    ref_text = metadata.get('sentence', metadata.get('text', ''))
                    current_lang = metadata.get('locale', 'unknown')
                    
                    if not ref_text:
                        print(f"Skipping {meta_path}: No valid text found in metadata")
                        continue
                    
                except json.JSONDecodeError:
                    print(f"Skipping {meta_path}: Unable to parse as valid JSON")
                    continue

            data, sr = sf.read(wav_path)
            duration_s = round(len(data) / sr, 2)
            ref_pho = get_phonemes(ref_text)
            
            record = {
                "utt_id": f"{current_lang}_{stem}",
                "lang": current_lang,
                "wav_path": wav_path,
                "ref_text": ref_text,
                "sr": sr,
                "duration_s": duration_s,
                "snr_db": None,
                "ref_pho": ref_pho
            }
            
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
            newly_processed_count += 1
            print(f"Processed: {filename}")

    
    if newly_processed_count > 0:
        os.rename(temp_manifest_path, final_manifest_path)
        total = len(processed_id) + newly_processed_count
        print(f"\n{newly_processed_count} records have been processed, Total records: {total}. Manifest file created: {final_manifest_path}")
    else:
        print("\nNo matching data was processed.")
        if os.path.exists(temp_manifest_path):
            os.remove(temp_manifest_path)

if __name__ == "__main__":
    main()