#!/usr/bin/env python3
"""
Text Generation Script

Generates texts ONCE and saves them for later attack/detection experiments.
Supports both watermarked and non-watermarked (baseline) generation.

Usage:
    # Generate watermarked texts
    python generate_texts.py --model opt-1.3b --watermarker gpw --n_samples 200

    # Generate non-watermarked baseline texts
    python generate_texts.py --model opt-1.3b --watermarker none --n_samples 200

    # Generate with all watermarkers
    python generate_texts.py --model opt-1.3b --watermarker all --n_samples 200
"""

import os
import sys
import json
import pickle
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
    "none": {"type": "none"},  # No watermark - baseline
    "unigram": {"type": "unigram", "gamma": 0.5, "delta": 2.0, "z_threshold": 4.0},
    "kgw": {"type": "kgw", "gamma": 0.5, "delta": 2.0, "z_threshold": 4.0, "seeding_scheme": "simple_1"},
    "gpw": {"type": "gpw", "variant": "GPW", "alpha": 3.0, "omega": 50.0, "z_threshold": 4.0},
    "gpw_sp": {"type": "gpw", "variant": "GPW-SP", "alpha": 3.0, "omega": 50.0, "z_threshold": 4.0},
    "gpw_sp_low": {"type": "gpw", "variant": "GPW-SP", "alpha": 3.0, "omega": 2.0, "z_threshold": 4.0},
    "gpw_sp_sr": {"type": "gpw", "variant": "GPW-SP+SR", "alpha": 3.0, "omega": 50.0, "z_threshold": 4.0, "sr_lambda": 0.1, "sr_rank": 16},
}

GENERATED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "generated_texts")
os.makedirs(GENERATED_DIR, exist_ok=True)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_watermarker(config: Dict, model, tokenizer, device: str):
    """Create watermarker from config. Returns None for baseline."""
    wm_type = config["type"]

    if wm_type == "none":
        return None  # No watermark

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


def generate_without_watermark(prompt: str, model, tokenizer, device: str, max_tokens: int) -> str:
    """Generate text without watermarking (baseline)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_texts(model_name: str, watermarker_name: str, n_samples: int):
    """Generate texts and save them."""

    device = get_device()
    model_config = MODEL_CONFIGS[model_name]
    wm_config = WATERMARKER_CONFIGS[watermarker_name]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print(f"TEXT GENERATION: {watermarker_name}")
    print("=" * 70)
    print(f"Model: {model_config['name']}")
    print(f"Watermarker: {watermarker_name}")
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

    # Create watermarker (or None for baseline)
    print(f"\nCreating watermarker: {watermarker_name}...")
    watermarker = create_watermarker(wm_config, model, tokenizer, device)

    # Generate texts
    print(f"\nGenerating texts...")
    generated_data = []

    for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
        try:
            if watermarker is None:
                # No watermark - baseline generation
                text = generate_without_watermark(prompt, model, tokenizer, device, model_config["max_tokens"])
            else:
                # Watermarked generation
                text = watermarker.generate(prompt, max_new_tokens=model_config["max_tokens"],
                                           temperature=1.0, top_k=50, top_p=0.95)

            if text is None:
                continue

            generated_data.append({
                "sample_idx": i,
                "prompt": prompt,
                "generated_text": text,
                "human_text": human_texts[i],
            })

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    print(f"\nSuccessfully generated {len(generated_data)}/{n_samples} texts")

    # Save generated texts
    output_file = os.path.join(GENERATED_DIR, f"{model_name}_{watermarker_name}_{timestamp}.pkl")

    save_data = {
        "model": model_name,
        "watermarker": watermarker_name,
        "config": wm_config,
        "n_samples": len(generated_data),
        "timestamp": timestamp,
        "data": generated_data,
    }

    with open(output_file, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"\nSaved to: {output_file}")

    # Also save as JSON for readability (without full texts to save space)
    meta_file = os.path.join(GENERATED_DIR, f"{model_name}_{watermarker_name}_{timestamp}_meta.json")
    meta_data = {
        "model": model_name,
        "watermarker": watermarker_name,
        "config": {k: str(v) for k, v in wm_config.items()},
        "n_samples": len(generated_data),
        "timestamp": timestamp,
        "pickle_file": output_file,
    }

    with open(meta_file, 'w') as f:
        json.dump(meta_data, f, indent=2)

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Generate Texts for Experiments")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--watermarker", type=str, required=True,
                        choices=list(WATERMARKER_CONFIGS.keys()) + ["all"])
    parser.add_argument("--n_samples", type=int, default=200)
    args = parser.parse_args()

    if args.watermarker == "all":
        # Generate for all watermarkers
        for wm_name in WATERMARKER_CONFIGS.keys():
            generate_texts(args.model, wm_name, args.n_samples)
    else:
        generate_texts(args.model, args.watermarker, args.n_samples)


if __name__ == "__main__":
    main()
