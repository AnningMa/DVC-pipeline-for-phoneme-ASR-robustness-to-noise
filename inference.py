import os
import json
import argparse
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

def main():
    parser = argparse.ArgumentParser(description="Run phoneme recognition inference.")
    parser.add_argument("--manifest", type=str, required=True, help="path to noisy.jsonl ")
    parser.add_argument("--out_manifest", type=str, required=True, help="output path for prediction results jsonl")
    parser.add_argument("--model_id", type=str, default="facebook/wav2vec2-lv-60-espeak-cv-ft", help="model name")
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"File not found: {args.manifest}")
        return

    os.makedirs(os.path.dirname(args.out_manifest), exist_ok=True)
    temp_manifest_path = args.out_manifest + ".tmp"

    print(f"Loading model {args.model_id}")
    
    processor = Wav2Vec2Processor.from_pretrained(args.model_id)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_id)
    
    target_sr = 16000
    processed_count = 0

    with open(args.manifest, 'r', encoding='utf-8') as f_in, \
         open(temp_manifest_path, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            if not line.strip():
                continue

            record = json.loads(line)
            wav_path = record['wav_path']

            if not os.path.exists(wav_path):
                print(f"Skipping: File not found {wav_path}")
                continue

            waveform, sr = torchaudio.load(wav_path)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)

            input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=target_sr).input_values
            
            with torch.no_grad():
                logits = model(input_values).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            hyp_pho = processor.batch_decode(predicted_ids)[0]

            record['hyp_pho'] = hyp_pho

            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
            processed_count += 1
            print(f"Processed {record.get('utt_id', 'unknown')} (SNR: {record.get('snr_db', 'N/A')})")

    if processed_count > 0:
        os.rename(temp_manifest_path, args.out_manifest)
        print(f"Inference pipeline completed! Total {processed_count} items processed, results saved to: {args.out_manifest}")
    else:
        print("\nNo data processed.")
        if os.path.exists(temp_manifest_path):
            os.remove(temp_manifest_path)

if __name__ == "__main__":
    main()