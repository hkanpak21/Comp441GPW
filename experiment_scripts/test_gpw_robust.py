#!/usr/bin/env python3
"""
Quick test to validate GPW robust detection improves attack resilience.

Tests GPW-SP standard vs GPW-SP-R (robust) on small subset (30 samples).
"""

import os
import sys
import hashlib

# Get project root and add to path
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir)
sys.path.insert(0, PROJECT_ROOT)

import importlib.util
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

dataset_loader = load_module("dataset_loader", os.path.join(script_dir, "dataset_loader.py"))
load_c4_dataset = dataset_loader.load_c4_dataset

# Import watermarkers and attacks
from watermarkers import create_gpw_variant
from watermarkers.gpw import GPWConfig, SRConfig, GPWWatermark
from attacks import SynonymAttack, SwapAttack, TypoAttack, CopyPasteAttack

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_NAME = "facebook/opt-1.3b"
NUM_SAMPLES = 30  # Small subset for quick testing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# GPW configurations to compare
GPW_CONFIGS = {
    "GPW-SP (standard)": {
        "variant": "GPW-SP",
        "alpha": 3.0,
        "omega": 1.0,
        "robust_detect": False,
    },
    "GPW-SP-R (robust 10%)": {
        "variant": "GPW-SP",
        "alpha": 3.0,
        "omega": 1.0,
        "robust_detect": True,
        "trim_fraction": 0.10,
    },
    "GPW-SP-R (robust 15%)": {
        "variant": "GPW-SP",
        "alpha": 3.0,
        "omega": 1.0,
        "robust_detect": True,
        "trim_fraction": 0.15,
    },
    "GPW-SP-R (robust 20%)": {
        "variant": "GPW-SP",
        "alpha": 3.0,
        "omega": 1.0,
        "robust_detect": True,
        "trim_fraction": 0.20,
    },
}

# Attacks to test
ATTACKS = {
    "clean": None,
    "synonym_30%": SynonymAttack(edit_rate=0.3),
    "swap_20%": SwapAttack(edit_rate=0.2),
    "typo_10%": TypoAttack(edit_rate=0.1),
    "copypaste_50%": CopyPasteAttack(n_segments=3, watermark_ratio=0.5),
}


def run_robust_test():
    print("=" * 80)
    print("GPW ROBUST DETECTION TEST")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Samples: {NUM_SAMPLES}")
    print(f"Device: {DEVICE}")
    print()
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print("Loading dataset...")
    c4_data = load_c4_dataset(num_samples=NUM_SAMPLES)
    prompts = [d['prompt'] for d in c4_data]
    human_texts = [d['text'] for d in c4_data]
    print(f"Loaded {len(prompts)} samples\n")
    
    # First, generate watermarked texts using standard GPW-SP
    # (same generation for all detection variants)
    print("Generating watermarked texts (GPW-SP, α=3, ω=1)...")
    
    gpw_cfg = GPWConfig(
        alpha=3.0,
        omega=1.0,
        salted=True,
        ctx_mode="ngram",
        ngram=4,
        robust_detect=False,
    )
    sr_cfg = SRConfig(enabled=False)
    
    generator = GPWWatermark(
        model=model,
        tokenizer=tokenizer,
        gpw_cfg=gpw_cfg,
        sr_cfg=sr_cfg,
        hash_key=hashlib.sha256(b"gpw-robust-test").digest(),
        device=DEVICE,
    )
    
    watermarked_texts = []
    for i, prompt in enumerate(prompts):
        try:
            wm_text = generator.generate(
                prompt,
                max_new_tokens=120,
                temperature=1.0,
                top_k=50,
                top_p=0.95,
            )
            if wm_text.startswith(prompt):
                wm_text = wm_text[len(prompt):].strip()
            watermarked_texts.append(wm_text)
            if (i + 1) % 10 == 0:
                print(f"  Generated {i+1}/{NUM_SAMPLES}")
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            watermarked_texts.append("")
    
    print(f"\nGenerated {sum(1 for t in watermarked_texts if t)} valid texts\n")
    
    # Now test each detection configuration
    results = {}
    
    for config_name, config in GPW_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Testing: {config_name}")
        print("=" * 60)
        
        # Create detector with this config
        gpw_cfg = GPWConfig(
            alpha=config["alpha"],
            omega=config["omega"],
            salted=True,
            ctx_mode="ngram",
            ngram=4,
            robust_detect=config["robust_detect"],
            trim_fraction=config.get("trim_fraction", 0.1),
        )
        
        detector = GPWWatermark(
            model=model,
            tokenizer=tokenizer,
            gpw_cfg=gpw_cfg,
            sr_cfg=SRConfig(enabled=False),
            hash_key=hashlib.sha256(b"gpw-robust-test").digest(),
            device=DEVICE,
        )
        
        results[config_name] = {}
        
        for attack_name, attack in ATTACKS.items():
            detected_count = 0
            total_z = 0.0
            valid_count = 0
            
            for i, wm_text in enumerate(watermarked_texts):
                if not wm_text:
                    continue
                
                # Apply attack
                if attack is None:
                    attacked_text = wm_text
                elif "copypaste" in attack_name:
                    attacked_text = attack(wm_text, human_text=human_texts[i])
                else:
                    attacked_text = attack(wm_text)
                
                if not attacked_text:
                    continue
                
                try:
                    det_result = detector.detect(attacked_text)
                    if det_result.is_watermarked:
                        detected_count += 1
                    total_z += det_result.z_score
                    valid_count += 1
                except Exception as e:
                    continue
            
            if valid_count > 0:
                det_rate = detected_count / valid_count
                mean_z = total_z / valid_count
                results[config_name][attack_name] = {
                    "detection_rate": det_rate,
                    "mean_z": mean_z,
                    "count": valid_count,
                }
                print(f"  {attack_name:15s}: {det_rate*100:5.1f}% detected, Z={mean_z:.2f}")
    
    # Print comparison summary
    print("\n" + "=" * 80)
    print("SUMMARY: Detection Rates (%)")
    print("=" * 80)
    
    # Header
    attacks = list(ATTACKS.keys())
    header = f"{'Config':<25} |"
    for attack in attacks:
        header += f" {attack:>12} |"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for config_name in GPW_CONFIGS.keys():
        row = f"{config_name:<25} |"
        for attack in attacks:
            if attack in results.get(config_name, {}):
                rate = results[config_name][attack]["detection_rate"] * 100
                row += f" {rate:11.1f}% |"
            else:
                row += f" {'N/A':>11} |"
        print(row)
    
    # Improvement analysis
    print("\n" + "=" * 80)
    print("IMPROVEMENT OVER STANDARD (percentage points)")
    print("=" * 80)
    
    baseline = results.get("GPW-SP (standard)", {})
    
    header = f"{'Config':<25} |"
    for attack in attacks[1:]:  # Skip clean
        header += f" {attack:>12} |"
    print(header)
    print("-" * len(header))
    
    for config_name in list(GPW_CONFIGS.keys())[1:]:  # Skip baseline
        row = f"{config_name:<25} |"
        for attack in attacks[1:]:
            if attack in results.get(config_name, {}) and attack in baseline:
                improvement = (results[config_name][attack]["detection_rate"] - 
                              baseline[attack]["detection_rate"]) * 100
                row += f" {improvement:+10.1f}pp |"
            else:
                row += f" {'N/A':>11} |"
        print(row)
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    # Find best robust config
    best_config = None
    best_avg_improvement = -float('inf')
    
    for config_name in list(GPW_CONFIGS.keys())[1:]:
        if config_name not in results:
            continue
        improvements = []
        for attack in attacks[1:]:
            if attack in results[config_name] and attack in baseline:
                imp = results[config_name][attack]["detection_rate"] - baseline[attack]["detection_rate"]
                improvements.append(imp)
        if improvements:
            avg_imp = sum(improvements) / len(improvements)
            if avg_imp > best_avg_improvement:
                best_avg_improvement = avg_imp
                best_config = config_name
    
    if best_config:
        print(f"Best robust config: {best_config}")
        print(f"Average improvement: {best_avg_improvement*100:+.1f}pp across attacks")
        
        # Check clean detection penalty
        if "clean" in results.get(best_config, {}) and "clean" in baseline:
            clean_diff = (results[best_config]["clean"]["detection_rate"] - 
                         baseline["clean"]["detection_rate"]) * 100
            print(f"Clean detection change: {clean_diff:+.1f}pp")
    else:
        print("No improvement found - standard detection may be optimal")


if __name__ == "__main__":
    run_robust_test()
