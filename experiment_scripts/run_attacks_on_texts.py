#!/usr/bin/env python3
"""
Run Attacks on Pre-Generated Texts

Applies attacks to previously generated texts and runs detection.
This is more efficient than regenerating texts for each attack.

Usage:
    # Run all attacks on a generated text file
    python run_attacks_on_texts.py --input generated_texts/opt-1.3b_gpw_*.pkl --attacks all

    # Run specific attacks
    python run_attacks_on_texts.py --input generated_texts/opt-1.3b_gpw_*.pkl --attacks clean synonym_30 swap_20

    # Run on baseline (no watermark) - measures FPR
    python run_attacks_on_texts.py --input generated_texts/opt-1.3b_none_*.pkl --attacks all --detector gpw
"""

import os
import sys
import csv
import json
import glob
import pickle
import hashlib
import argparse
from datetime import datetime
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer

# Watermarkers (for detection)
from watermarkers import UnigramWatermark, KGWWatermark
from watermarkers.gpw import GPWWatermark, GPWConfig, SRConfig

# Attacks
from attacks import SynonymAttack, SwapAttack, TypoAttack, CopyPasteAttack
from attacks.paraphrase import PegasusAttack

# ============================================================================
# CONFIGURATION
# ============================================================================

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

ATTACK_CONFIGS = {
    "clean": None,
    "synonym_30": {"type": "synonym", "edit_rate": 0.30},
    "swap_20": {"type": "swap", "edit_rate": 0.20},
    "typo_10": {"type": "typo", "edit_rate": 0.10},
    "copypaste_50": {"type": "copypaste", "n_segments": 3, "watermark_ratio": 0.50},
    "paraphrase": {"type": "paraphrase"},
}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_detector(config: Dict, model, tokenizer, device: str):
    """Create detector watermarker from config."""
    wm_type = config["type"]

    if wm_type == "unigram":
        return UnigramWatermark(model=model, tokenizer=tokenizer, gamma=config["gamma"],
                                delta=config["delta"], z_threshold=config["z_threshold"], device=device)

    elif wm_type == "kgw":
        return KGWWatermark(model=model, tokenizer=tokenizer, gamma=config["gamma"],
                           delta=config["delta"], z_threshold=config["z_threshold"],
                           seeding_scheme=config["seeding_scheme"], device=device)

    elif wm_type == "gpw":
        variant = config["variant"]
        gpw_cfg = GPWConfig(alpha=config["alpha"], omega=config["omega"],
                           salted=(variant != "GPW"), ctx_mode="ngram", ngram=4)
        sr_enabled = "SR" in variant
        sr_cfg = SRConfig(enabled=sr_enabled,
                         lambda_couple=config.get("sr_lambda", 0.1) if sr_enabled else 0.09,
                         rank=config.get("sr_rank", 16) if sr_enabled else 32)
        wm = GPWWatermark(model=model, tokenizer=tokenizer, gpw_cfg=gpw_cfg, sr_cfg=sr_cfg,
                         hash_key=hashlib.sha256(b"gpw-final-exp").digest(), device=device)
        wm.z_threshold = config["z_threshold"]
        return wm

    raise ValueError(f"Unknown watermarker type: {wm_type}")


def create_attack(config: Optional[Dict]):
    """Create attack from config."""
    if config is None:
        return None

    attack_type = config["type"]

    if attack_type == "synonym":
        return SynonymAttack(edit_rate=config["edit_rate"])
    elif attack_type == "swap":
        return SwapAttack(edit_rate=config["edit_rate"])
    elif attack_type == "typo":
        return TypoAttack(edit_rate=config["edit_rate"])
    elif attack_type == "copypaste":
        return CopyPasteAttack(n_segments=config["n_segments"], watermark_ratio=config["watermark_ratio"])
    elif attack_type == "paraphrase":
        return PegasusAttack()

    raise ValueError(f"Unknown attack type: {attack_type}")


def apply_attack(attack, text: str, human_text: Optional[str] = None) -> str:
    """Apply attack and return attacked text."""
    if attack is None:
        return text

    attack_name = attack.__class__.__name__

    if "CopyPaste" in attack_name:
        if human_text is None:
            return text
        result = attack.attack(text, human_text=human_text)
    else:
        result = attack.attack(text)

    if hasattr(result, 'attacked_text'):
        return result.attacked_text
    return str(result) if result else text


def run_attacks_on_texts(input_file: str, attacks: List[str], detector_name: Optional[str] = None, max_samples: Optional[int] = None):
    """Run attacks on pre-generated texts."""

    device = get_device()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load generated texts
    print(f"\nLoading texts from: {input_file}")
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    model_name = data["model"]
    watermarker_name = data["watermarker"]
    generated_data = data["data"]

    # Limit samples if specified
    if max_samples and max_samples < len(generated_data):
        generated_data = generated_data[:max_samples]
        print(f"Limited to {max_samples} samples (from {len(data['data'])})")

    print(f"Model: {model_name}")
    print(f"Watermarker: {watermarker_name}")
    print(f"Samples: {len(generated_data)}")

    # Determine detector
    if watermarker_name == "none":
        # Baseline - need to specify detector for FPR measurement
        if detector_name is None:
            print("ERROR: For baseline texts, you must specify --detector")
            return
        det_config = WATERMARKER_CONFIGS[detector_name]
        detector_key = detector_name
    else:
        # Use same watermarker for detection
        det_config = WATERMARKER_CONFIGS[watermarker_name]
        detector_key = watermarker_name

    # Load model for detection
    print(f"\nLoading model for detection...")
    model_config = MODEL_CONFIGS[model_name]
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

    # Create detector
    print(f"Creating detector: {detector_key}...")
    detector = create_detector(det_config, model, tokenizer, device)

    # Process each attack
    all_results = []

    for attack_name in attacks:
        print(f"\n{'='*60}")
        print(f"ATTACK: {attack_name}")
        print("=" * 60)

        attack_config = ATTACK_CONFIGS.get(attack_name)

        try:
            attack = create_attack(attack_config)
        except Exception as e:
            print(f"Error creating attack: {e}")
            continue

        detected_count = 0
        z_scores = []
        results = []

        for item in tqdm(generated_data, desc=f"Processing {attack_name}"):
            try:
                text = item["generated_text"]
                human_text = item.get("human_text")

                # Apply attack
                attacked_text = apply_attack(attack, text, human_text)

                if not attacked_text:
                    continue

                # Detect
                det_result = detector.detect(attacked_text)
                z_scores.append(det_result.z_score)

                if det_result.is_watermarked:
                    detected_count += 1

                results.append({
                    "model": model_name,
                    "watermarker": watermarker_name,
                    "detector": detector_key,
                    "attack": attack_name,
                    "sample_idx": item["sample_idx"],
                    "z_score": det_result.z_score,
                    "p_value": det_result.p_value,
                    "is_detected": 1 if det_result.is_watermarked else 0,
                })

            except Exception as e:
                continue

        # Summary for this attack
        n_valid = len(z_scores)
        det_rate = (detected_count / n_valid * 100) if n_valid > 0 else 0
        mean_z = (sum(z_scores) / n_valid) if n_valid > 0 else 0

        print(f"\nResults: Detection {det_rate:.1f}% ({detected_count}/{n_valid}), Mean Z: {mean_z:.2f}")

        all_results.extend(results)

        # Save individual attack results
        output_file = os.path.join(RESULTS_DIR, f"attack_{model_name}_{watermarker_name}_{attack_name}_{timestamp}.csv")
        if results:
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)

        # Save summary
        summary = {
            "model": model_name,
            "watermarker": watermarker_name,
            "detector": detector_key,
            "attack": attack_name,
            "n_samples": n_valid,
            "detection_rate": det_rate,
            "mean_z_score": mean_z,
            "is_baseline": watermarker_name == "none",
            "timestamp": timestamp,
        }

        summary_file = os.path.join(RESULTS_DIR, f"attack_{model_name}_{watermarker_name}_{attack_name}_{timestamp}_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All attacks complete! Results saved to: {RESULTS_DIR}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run Attacks on Pre-Generated Texts")
    parser.add_argument("--input", type=str, required=True, help="Input pickle file with generated texts")
    parser.add_argument("--attacks", type=str, nargs="+", default=["all"],
                        help="Attacks to run (or 'all' for all attacks)")
    parser.add_argument("--detector", type=str, default=None,
                        help="Detector to use (required for baseline texts)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples to process (for slow detectors)")
    args = parser.parse_args()

    # Expand 'all' attacks
    if args.attacks == ["all"]:
        attacks = list(ATTACK_CONFIGS.keys())
    else:
        attacks = args.attacks

    # Find input file(s)
    input_files = glob.glob(args.input)
    if not input_files:
        print(f"No files found matching: {args.input}")
        return

    for input_file in input_files:
        run_attacks_on_texts(input_file, attacks, args.detector, args.max_samples)


if __name__ == "__main__":
    main()
