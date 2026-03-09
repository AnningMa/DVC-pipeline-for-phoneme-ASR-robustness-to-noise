import os
import json
import argparse
import soundfile as sf
import numpy as np
import yaml

def add_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    noise = rng.normal(
        loc=0.0, 
        scale=np.sqrt(noise_power), 
        size=signal.shape
    )
    return signal + noise

def add_noise_to_file(input_wav: str, output_wav: str, snr_db: float, seed: int | None = None) -> None:
    signal, sr = sf.read(input_wav)
    
    if signal.ndim != 1:
        raise ValueError("Only mono audio is supported")
        
    rng = np.random.default_rng(seed)
    noisy_signal = add_noise(signal, snr_db, rng)
    
    sf.write(output_wav, noisy_signal, sr)

def main():
    parser = argparse.ArgumentParser(description="Add noise to audio dataset.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to the clean.jsonl manifest file")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for noisy audio files")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed, to ensure consistent noise addition")
    args = parser.parse_args()
    
    try:
        with open("params.yaml", "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
            noise_levels = params.get("noise_levels", [])
    except FileNotFoundError:
        print("Warning: params.yaml not found. Using default noise levels.")
        noise_levels = [0, 5, 10, 15, 20]
        

    wav_out_dir = os.path.join(args.out_dir, "wav")
    os.makedirs(wav_out_dir, exist_ok=True)
    
    out_manifest_path = os.path.join(args.out_dir, "noisy.jsonl")
    temp_manifest_path = out_manifest_path + ".tmp"

    processed_count = 0
    
    with open(args.manifest, 'r', encoding='utf-8') as f_in, \
         open(temp_manifest_path, 'w', encoding='utf-8') as f_out:
             
        for line in f_in:
            if not line.strip():
                continue
                
            record = json.loads(line)
            clean_wav_path = record['wav_path']
            original_utt_id = record['utt_id']
            stem = original_utt_id.split('_', 1)[-1]
            
            if not os.path.exists(clean_wav_path):
                print(f"Skipping {clean_wav_path}")
                continue

            for snr in noise_levels:
                noisy_wav_path = os.path.join(wav_out_dir, f"{stem}_noisy_{snr}db.wav")
                
                try:
                    add_noise_to_file(clean_wav_path, noisy_wav_path, snr, args.seed)
                except ValueError as e:
                    print(f"Skipping {clean_wav_path} (SNR: {snr}) - Error: {e}")
                    continue
                
                new_record = record.copy()
                
                new_record['utt_id'] = f"{original_utt_id}_snr{snr}"
                new_record['wav_path'] = noisy_wav_path
                new_record['snr_db'] = snr
                
                f_out.write(json.dumps(new_record, ensure_ascii=False) + '\n')
                processed_count += 1

    if processed_count > 0:
        os.rename(temp_manifest_path, out_manifest_path)
        print(f"\n{processed_count} noisy records have been generated. Manifest file created: {out_manifest_path}")
    else:
        print("\nNo noisy records were generated.")
        if os.path.exists(temp_manifest_path):
            os.remove(temp_manifest_path)

if __name__ == "__main__":
    main()