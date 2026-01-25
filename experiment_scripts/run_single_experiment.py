#!/usr/bin/env python3
"""
Single Experiment Runner

Runs a single watermarker + attack combination for tracking individual experiments.

Usage:
    python run_single_experiment.py --model opt-1.3b --watermarker gpw --attack clean --n_samples 200
    python run_single_experiment.py --model gpt2 --watermarker unigram --attack synonym_30 --n_samples 200
"""

import os
import sys
import csv
import json
import hashlib
import argparse
from datetime import datetime
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Watermarkers
from watermarkers import UnigramWatermark, KGWWatermark
from watermarkers.gpw import GPWWatermark, GPWConfig, SRConfig
from watermarkers.semstamp import SEMSTAMPWatermark

# Attacks
from attacks import SynonymAttack, SwapAttack, TypoAttack, CopyPasteAttack
from attacks.paraphrase import PegasusAttack

# Data loaders
from data_loaders import load_c4

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_CONFIGS = {
    "gpt2": {"name": "gpt2", "dtype": torch.float32, "max_tokens": 200},
    "opt-1.3b": {"name": "facebook/opt-1.3b", "dtype": torch.float16, "max_tokens": 200},
    "qwen-7b": {"name": "Qwen/Qwen2.5-7B-Instruct", "dtype": torch.float16, "max_tokens": 200},
}

WATERMARKER_CONFIGS = {
    "unigram": {"type": "unigram", "gamma": 0.5, "delta": 2.0, "z_threshold": 4.0},
    "kgw": {"type": "kgw", "gamma": 0.5, "delta": 2.0, "z_threshold": 4.0, "seeding_scheme": "simple_1"},
    "gpw": {"type": "gpw", "variant": "GPW", "alpha": 3.0, "omega": 50.0, "z_threshold": 4.0},
    "gpw_sp": {"type": "gpw", "variant": "GPW-SP", "alpha": 3.0, "omega": 50.0, "z_threshold": 4.0},
    "gpw_sp_low": {"type": "gpw", "variant": "GPW-SP", "alpha": 3.0, "omega": 2.0, "z_threshold": 4.0},
    "gpw_sp_sr": {"type": "gpw", "variant": "GPW-SP+SR", "alpha": 3.0, "omega": 50.0, "z_threshold": 4.0, "sr_lambda": 0.1, "sr_rank": 16},
    "semstamp": {"type": "semstamp", "n_lsh_bits": 3, "lsh_margin": 0.0, "z_threshold": 4.0, "max_rejections": 2},
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


def compute_perplexity(text: str, model, tokenizer, device: str) -> float:
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            return float(torch.exp(outputs.loss).cpu())
    except:
        return -1.0


def create_watermarker(config: Dict, model, tokenizer, device: str):
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

    elif wm_type == "semstamp":
        embedder = SentenceTransformer("all-mpnet-base-v2")
        embedder.to(device)
        return SEMSTAMPWatermark(model=model, tokenizer=tokenizer, embedder=embedder,
                                lsh_dim=config["n_lsh_bits"], margin=config["lsh_margin"],
                                z_threshold=config["z_threshold"], max_rejections=config["max_rejections"],
                                device=device)

    raise ValueError(f"Unknown watermarker type: {wm_type}")


def create_attack(config: Optional[Dict]):
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


def run_experiment(model_name: str, watermarker_name: str, attack_name: str, n_samples: int):
    """Run a single experiment."""

    device = get_device()
    model_config = MODEL_CONFIGS[model_name]
    wm_config = WATERMARKER_CONFIGS[watermarker_name]
    attack_config = ATTACK_CONFIGS[attack_name]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("="*70)
    print(f"SINGLE EXPERIMENT: {watermarker_name} + {attack_name}")
    print("="*70)
    print(f"Model: {model_config['name']}")
    print(f"Watermarker: {watermarker_name}")
    print(f"Attack: {attack_name}")
    print(f"Samples: {n_samples}")
    print(f"Device: {device}")

    # Load model
    print("\nLoading model...")
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

    # Load data
    print("\nLoading dataset...")
    c4_data = load_c4(num_samples=n_samples)
    prompts = [d['prompt'] for d in c4_data]
    human_texts = [d['text'] for d in c4_data]
    print(f"Loaded {len(prompts)} samples")

    # Create watermarker
    print("\nCreating watermarker...")
    watermarker = create_watermarker(wm_config, model, tokenizer, device)

    # Create attack
    print(f"Creating attack: {attack_name}...")
    attack = create_attack(attack_config)

    # Generate and test
    print(f"\nRunning experiment...")
    results = []
    detected_count = 0
    z_scores = []
    perplexities = []

    for i, prompt in enumerate(tqdm(prompts, desc="Processing")):
        try:
            # Generate watermarked text
            wm_text = watermarker.generate(prompt, max_new_tokens=model_config["max_tokens"],
                                           temperature=1.0, top_k=50, top_p=0.95)

            if wm_text is None:
                continue

            # Compute perplexity (only for clean)
            if attack_name == "clean":
                generated_part = wm_text[len(prompt):].strip()
                if len(generated_part) > 20:
                    ppl = compute_perplexity(generated_part, model, tokenizer, device)
                    perplexities.append(ppl)

            # Apply attack
            attacked_text = apply_attack(attack, wm_text, human_texts[i])

            if not attacked_text:
                continue

            # Detect
            det_result = watermarker.detect(attacked_text)
            z_scores.append(det_result.z_score)

            if det_result.is_watermarked:
                detected_count += 1

            results.append({
                "model": model_name,
                "watermarker": watermarker_name,
                "attack": attack_name,
                "sample_idx": i,
                "z_score": det_result.z_score,
                "p_value": det_result.p_value,
                "is_detected": 1 if det_result.is_watermarked else 0,
                "perplexity": perplexities[-1] if attack_name == "clean" and perplexities else -1,
            })

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    # Summary
    n_valid = len(z_scores)
    det_rate = (detected_count / n_valid * 100) if n_valid > 0 else 0
    mean_z = (sum(z_scores) / n_valid) if n_valid > 0 else 0
    mean_ppl = (sum(p for p in perplexities if p > 0) / len([p for p in perplexities if p > 0])) if perplexities else -1

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Detection Rate: {det_rate:.1f}% ({detected_count}/{n_valid})")
    print(f"Mean Z-Score: {mean_z:.2f}")
    if mean_ppl > 0:
        print(f"Mean Perplexity: {mean_ppl:.2f}")

    # Save results
    output_file = os.path.join(RESULTS_DIR, f"exp_{model_name}_{watermarker_name}_{attack_name}_{timestamp}.csv")
    if results:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {output_file}")

    # Save summary
    summary = {
        "model": model_name,
        "watermarker": watermarker_name,
        "attack": attack_name,
        "n_samples": n_valid,
        "detection_rate": det_rate,
        "mean_z_score": mean_z,
        "mean_perplexity": mean_ppl,
        "timestamp": timestamp,
    }

    summary_file = os.path.join(RESULTS_DIR, f"exp_{model_name}_{watermarker_name}_{attack_name}_{timestamp}_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Single Watermarking Experiment")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--watermarker", type=str, required=True, choices=list(WATERMARKER_CONFIGS.keys()))
    parser.add_argument("--attack", type=str, required=True, choices=list(ATTACK_CONFIGS.keys()))
    parser.add_argument("--n_samples", type=int, default=200)
    args = parser.parse_args()

    run_experiment(args.model, args.watermarker, args.attack, args.n_samples)


if __name__ == "__main__":
    main()
