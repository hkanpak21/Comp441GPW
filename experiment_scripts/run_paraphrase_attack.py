#!/usr/bin/env python3
"""
Paraphrase Attack Script

Handles paraphrase attacks separately to avoid CUDA OOM.
Two-phase approach:
1. Apply paraphrase to texts (loads Pegasus, unloads after)
2. Run detection (loads detector model)

Usage:
    python run_paraphrase_attack.py --input generated_texts/opt-1.3b_gpw_*.pkl --max_samples 50
"""

import os
import sys
import csv
import json
import glob
import pickle
import hashlib
import argparse
import gc
from datetime import datetime
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def clear_gpu_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def apply_paraphrase_to_texts(generated_data: List[Dict], max_samples: int = None) -> List[Dict]:
    """Phase 1: Apply paraphrase attack to texts."""
    from attacks.paraphrase import PegasusAttack

    print("\n" + "=" * 60)
    print("PHASE 1: Applying Paraphrase Attack")
    print("=" * 60)

    # Limit samples
    if max_samples and max_samples < len(generated_data):
        generated_data = generated_data[:max_samples]

    print(f"Processing {len(generated_data)} samples...")
    print("Loading Pegasus paraphraser...")

    attack = PegasusAttack()
    paraphrased_data = []

    for item in tqdm(generated_data, desc="Paraphrasing"):
        try:
            text = item["generated_text"]
            result = attack.attack(text)

            if hasattr(result, 'attacked_text'):
                paraphrased_text = result.attacked_text
            else:
                paraphrased_text = str(result) if result else text

            paraphrased_data.append({
                "sample_idx": item["sample_idx"],
                "original_text": text,
                "paraphrased_text": paraphrased_text,
                "prompt": item.get("prompt", ""),
                "human_text": item.get("human_text", ""),
            })
        except Exception as e:
            print(f"  Error on sample {item.get('sample_idx', '?')}: {e}")
            continue

    print(f"Successfully paraphrased {len(paraphrased_data)}/{len(generated_data)} samples")

    # Unload Pegasus to free memory
    del attack
    clear_gpu_memory()

    return paraphrased_data


def run_detection(paraphrased_data: List[Dict], model_name: str, watermarker_name: str, detector_name: str = None):
    """Phase 2: Run detection on paraphrased texts."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from watermarkers import UnigramWatermark, KGWWatermark
    from watermarkers.gpw import GPWWatermark, GPWConfig, SRConfig

    print("\n" + "=" * 60)
    print("PHASE 2: Running Detection")
    print("=" * 60)

    device = get_device()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Model configs
    MODEL_CONFIGS = {
        "gpt2": {"name": "gpt2", "dtype": torch.float32},
        "opt-1.3b": {"name": "facebook/opt-1.3b", "dtype": torch.float16},
        "qwen-7b": {"name": "Qwen/Qwen2.5-7B-Instruct", "dtype": torch.float16},
    }

    WATERMARKER_CONFIGS = {
        "unigram": {"type": "unigram", "gamma": 0.5, "delta": 2.0, "z_threshold": 4.0},
        "kgw": {"type": "kgw", "gamma": 0.5, "delta": 2.0, "z_threshold": 4.0, "seeding_scheme": "simple_1"},
        "gpw": {"type": "gpw", "variant": "GPW", "alpha": 3.0, "omega": 50.0, "z_threshold": 4.0},
        "gpw_sp": {"type": "gpw", "variant": "GPW-SP", "alpha": 3.0, "omega": 50.0, "z_threshold": 4.0},
        "gpw_sp_low": {"type": "gpw", "variant": "GPW-SP", "alpha": 3.0, "omega": 2.0, "z_threshold": 4.0},
        "gpw_sp_sr": {"type": "gpw", "variant": "GPW-SP+SR", "alpha": 3.0, "omega": 50.0, "z_threshold": 4.0, "sr_lambda": 0.1, "sr_rank": 16},
    }

    # Determine detector
    det_key = detector_name if detector_name else watermarker_name
    if det_key == "none":
        print("ERROR: Cannot detect on baseline without specifying --detector")
        return

    det_config = WATERMARKER_CONFIGS.get(det_key)
    if not det_config:
        print(f"ERROR: Unknown detector: {det_key}")
        return

    model_config = MODEL_CONFIGS.get(model_name)
    if not model_config:
        print(f"ERROR: Unknown model: {model_name}")
        return

    print(f"Loading model: {model_config['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        torch_dtype=model_config["dtype"] if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Creating detector: {det_key}...")

    # Create detector
    wm_type = det_config["type"]
    if wm_type == "unigram":
        detector = UnigramWatermark(model=model, tokenizer=tokenizer, gamma=det_config["gamma"],
                                    delta=det_config["delta"], z_threshold=det_config["z_threshold"], device=device)
    elif wm_type == "kgw":
        detector = KGWWatermark(model=model, tokenizer=tokenizer, gamma=det_config["gamma"],
                               delta=det_config["delta"], z_threshold=det_config["z_threshold"],
                               seeding_scheme=det_config["seeding_scheme"], device=device)
    elif wm_type == "gpw":
        variant = det_config["variant"]
        gpw_cfg = GPWConfig(alpha=det_config["alpha"], omega=det_config["omega"],
                           salted=(variant != "GPW"), ctx_mode="ngram", ngram=4)
        sr_enabled = "SR" in variant
        sr_cfg = SRConfig(enabled=sr_enabled,
                         lambda_couple=det_config.get("sr_lambda", 0.1) if sr_enabled else 0.09,
                         rank=det_config.get("sr_rank", 16) if sr_enabled else 32)
        detector = GPWWatermark(model=model, tokenizer=tokenizer, gpw_cfg=gpw_cfg, sr_cfg=sr_cfg,
                               hash_key=hashlib.sha256(b"gpw-final-exp").digest(), device=device)
        detector.z_threshold = det_config["z_threshold"]

    # Run detection
    print(f"\nRunning detection on {len(paraphrased_data)} paraphrased samples...")

    detected_count = 0
    z_scores = []
    results = []

    for item in tqdm(paraphrased_data, desc="Detecting"):
        try:
            text = item["paraphrased_text"]
            det_result = detector.detect(text)
            z_scores.append(det_result.z_score)

            if det_result.is_watermarked:
                detected_count += 1

            results.append({
                "model": model_name,
                "watermarker": watermarker_name,
                "detector": det_key,
                "attack": "paraphrase",
                "sample_idx": item["sample_idx"],
                "z_score": det_result.z_score,
                "p_value": det_result.p_value,
                "is_detected": 1 if det_result.is_watermarked else 0,
            })
        except Exception as e:
            continue

    # Summary
    n_valid = len(z_scores)
    det_rate = (detected_count / n_valid * 100) if n_valid > 0 else 0
    mean_z = (sum(z_scores) / n_valid) if n_valid > 0 else 0

    print(f"\n{'='*60}")
    print(f"RESULTS: Detection {det_rate:.1f}% ({detected_count}/{n_valid}), Mean Z: {mean_z:.2f}")
    print("=" * 60)

    # Save results
    output_file = os.path.join(RESULTS_DIR, f"attack_{model_name}_{watermarker_name}_paraphrase_{timestamp}.csv")
    if results:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    summary = {
        "model": model_name,
        "watermarker": watermarker_name,
        "detector": det_key,
        "attack": "paraphrase",
        "n_samples": n_valid,
        "detection_rate": det_rate,
        "mean_z_score": mean_z,
        "is_baseline": watermarker_name == "none",
        "timestamp": timestamp,
    }

    summary_file = os.path.join(RESULTS_DIR, f"attack_{model_name}_{watermarker_name}_paraphrase_{timestamp}_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to: {summary_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Paraphrase Attack")
    parser.add_argument("--input", type=str, required=True, help="Input pickle file with generated texts")
    parser.add_argument("--detector", type=str, default=None, help="Detector to use (for baseline)")
    parser.add_argument("--max_samples", type=int, default=50, help="Max samples (default 50 to avoid OOM)")
    args = parser.parse_args()

    # Find input files
    input_files = glob.glob(args.input)
    if not input_files:
        print(f"No files found matching: {args.input}")
        return

    for input_file in input_files:
        print(f"\n{'='*70}")
        print(f"Processing: {input_file}")
        print("=" * 70)

        # Load data
        with open(input_file, 'rb') as f:
            data = pickle.load(f)

        model_name = data["model"]
        watermarker_name = data["watermarker"]
        generated_data = data["data"]

        print(f"Model: {model_name}")
        print(f"Watermarker: {watermarker_name}")
        print(f"Total samples: {len(generated_data)}")

        # Phase 1: Paraphrase
        paraphrased = apply_paraphrase_to_texts(generated_data, args.max_samples)

        if not paraphrased:
            print("No paraphrased samples - skipping detection")
            continue

        # Phase 2: Detection
        run_detection(paraphrased, model_name, watermarker_name, args.detector)


if __name__ == "__main__":
    main()
