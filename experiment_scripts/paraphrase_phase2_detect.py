#!/usr/bin/env python3
"""
Phase 2: Run detection on paraphrased texts.
Only loads watermark detector model, reads paraphrased texts from disk.
"""

import os
import sys
import pickle
import argparse
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_detection(input_pickle: str, watermarker_name: str, model_name: str, output_dir: str):
    """Load paraphrased texts and run watermark detection."""
    import torch
    from watermarkers import get_watermarker
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load paraphrased texts
    print(f"Loading paraphrased texts from: {input_pickle}")
    with open(input_pickle, 'rb') as f:
        data = pickle.load(f)

    paraphrased_texts = data.get('paraphrased_texts', [])
    orig_model = data.get('model', model_name)
    orig_watermarker = data.get('watermarker', watermarker_name)

    print(f"Model: {model_name}")
    print(f"Watermarker: {watermarker_name}")
    print(f"Samples: {len(paraphrased_texts)}")

    # Load the model for detection (needed for GPW variants)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_paths = {
        "opt-1.3b": "facebook/opt-1.3b",
        "gpt2": "gpt2",
        "qwen-7b": "Qwen/Qwen2-7B"
    }
    model_path = model_paths.get(model_name, model_name)

    print(f"Loading model {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    model.eval()

    # Create detector
    print(f"Creating detector: {watermarker_name}...")
    detector = get_watermarker(
        watermarker_name,
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    # Run detection
    results = []
    detected_count = 0
    z_scores = []

    for i, item in enumerate(tqdm(paraphrased_texts, desc="Detecting")):
        text = item.get('paraphrased', item.get('original', ''))

        try:
            result = detector.detect(text)
            z_score = result.get('z_score', 0)
            is_detected = result.get('is_detected', z_score >= 4.0)

            results.append({
                'model': model_name,
                'watermarker': watermarker_name,
                'detector': watermarker_name,
                'attack': 'paraphrase',
                'sample_idx': i,
                'z_score': z_score,
                'p_value': result.get('p_value', -1),
                'is_detected': 1 if is_detected else 0
            })

            if is_detected:
                detected_count += 1
            z_scores.append(z_score)

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            results.append({
                'model': model_name,
                'watermarker': watermarker_name,
                'detector': watermarker_name,
                'attack': 'paraphrase',
                'sample_idx': i,
                'z_score': 0,
                'p_value': -1,
                'is_detected': 0
            })

    # Calculate summary
    n_samples = len(paraphrased_texts)
    detection_rate = (detected_count / n_samples * 100) if n_samples > 0 else 0
    mean_z = sum(z_scores) / len(z_scores) if z_scores else 0

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Detection Rate: {detection_rate:.1f}% ({detected_count}/{n_samples})")
    print(f"Mean Z-Score: {mean_z:.2f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(output_dir, f"paraphrase_{model_name}_{watermarker_name}_{timestamp}.csv")
    summary_file = os.path.join(output_dir, f"paraphrase_{model_name}_{watermarker_name}_{timestamp}_summary.json")

    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)

    summary = {
        'model': model_name,
        'watermarker': watermarker_name,
        'attack': 'paraphrase',
        'n_samples': n_samples,
        'detection_rate': detection_rate,
        'detected_count': detected_count,
        'mean_z_score': mean_z,
        'timestamp': timestamp
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {csv_file}")
    print(f"Summary saved to: {summary_file}")

    # Clear GPU memory
    del model
    del detector
    torch.cuda.empty_cache()

    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Run detection on paraphrased texts")
    parser.add_argument("--input", required=True, help="Input pickle file with paraphrased texts")
    parser.add_argument("--watermarker", required=True, help="Watermarker name for detection")
    parser.add_argument("--model", required=True, help="Model name (opt-1.3b, gpt2, qwen-7b)")
    parser.add_argument("--output_dir", default="results", help="Output directory")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_detection(args.input, args.watermarker, args.model, args.output_dir)


if __name__ == "__main__":
    main()
