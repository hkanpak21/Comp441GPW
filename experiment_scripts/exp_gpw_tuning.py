#!/usr/bin/env python3
"""
GPW Parameter Tuning Experiments

Tests different GPW-SP configurations to find optimal attack robustness.
Compares:
- Original: alpha=2.0, omega=2.0
- Tuned: alpha=3.0, omega=1.0  
- Robust: alpha=4.0, omega=0.5

Key insight: Lower omega = wider "green bands" in cosine = more robust to token changes
"""

import os
import sys
import time
import hashlib
import csv
from datetime import datetime
import importlib.util

# Get project root and add to path
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir)
sys.path.insert(0, PROJECT_ROOT)

# Load modules using importlib
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

dataset_loader = load_module("dataset_loader", os.path.join(script_dir, "dataset_loader.py"))
load_c4_dataset = dataset_loader.load_c4_dataset

# Import watermarkers using package imports (they work with sys.path set)
from watermarkers import create_gpw_variant
from watermarkers.gpw import GPWConfig, SRConfig, GPWWatermark

# Import attacks
from attacks import SynonymAttack, SwapAttack, TypoAttack, CopyPasteAttack

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_NAME = "facebook/opt-1.3b"
NUM_SAMPLES = 50  # Smaller for quick tuning tests
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# GPW configurations to test
GPW_CONFIGS = {
    "gpw_sp_orig": {"alpha": 2.0, "omega": 2.0, "name": "Original (α=2, ω=2)"},
    "gpw_sp_tuned": {"alpha": 3.0, "omega": 1.0, "name": "Tuned (α=3, ω=1)"},
    "gpw_sp_robust": {"alpha": 4.0, "omega": 0.5, "name": "Robust (α=4, ω=0.5)"},
}

# Attacks to test
ATTACKS = {
    "clean": None,
    "synonym": SynonymAttack(edit_rate=0.3),
    "swap": SwapAttack(edit_rate=0.2),
    "typo": TypoAttack(edit_rate=0.1),
    "copypaste": CopyPasteAttack(n_segments=3, watermark_ratio=0.5),
}


def compute_perplexity(text, model, tokenizer, device):
    """Compute perplexity of text."""
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            return float(torch.exp(outputs.loss).cpu())
    except Exception as e:
        return -1.0


def run_tuning_experiments():
    print("=" * 80)
    print("GPW PARAMETER TUNING EXPERIMENTS")
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
    print(f"Loaded {len(prompts)} samples")
    
    # Results storage
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Test each GPW configuration
    for config_name, config in GPW_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"Alpha={config['alpha']}, Omega={config['omega']}")
        print("=" * 60)
        
        # Create watermarker with these params
        gpw_cfg = GPWConfig(
            alpha=config['alpha'],
            omega=config['omega'],
            salted=True,
            ctx_mode="ngram",
            ngram=4
        )
        sr_cfg = SRConfig(enabled=False)
        
        watermarker = GPWWatermark(
            model=model,
            tokenizer=tokenizer,
            gpw_cfg=gpw_cfg,
            sr_cfg=sr_cfg,
            hash_key=hashlib.sha256(b"gpw-tuning-test").digest(),
            device=DEVICE
        )
        
        # Generate watermarked texts
        print(f"Generating {NUM_SAMPLES} watermarked texts...")
        watermarked_texts = []
        for i, prompt in enumerate(prompts):
            try:
                wm_text = watermarker.generate(
                    prompt,
                    max_new_tokens=120,
                    temperature=1.0,
                    top_k=50,
                    top_p=0.95
                )
                # Remove prompt from output
                if wm_text.startswith(prompt):
                    wm_text = wm_text[len(prompt):].strip()
                watermarked_texts.append(wm_text)
                if (i + 1) % 10 == 0:
                    print(f"  Generated {i+1}/{NUM_SAMPLES}")
            except Exception as e:
                print(f"  Error on sample {i}: {e}")
                watermarked_texts.append("")
        
        # Test detection with and without attacks
        for attack_name, attack in ATTACKS.items():
            print(f"\n  Testing attack: {attack_name}")
            
            detected_count = 0
            total_z = 0.0
            valid_count = 0
            
            for i, wm_text in enumerate(watermarked_texts):
                if not wm_text:
                    continue
                    
                # Apply attack
                if attack is None:
                    attacked_text = wm_text
                elif attack_name == "copypaste":
                    attacked_text = attack(wm_text, human_text=human_texts[i])
                else:
                    attacked_text = attack(wm_text)
                
                if not attacked_text:
                    continue
                
                # Detect
                try:
                    result = watermarker.detect(attacked_text)
                    if result.is_watermarked:
                        detected_count += 1
                    total_z += result.z_score
                    valid_count += 1
                except Exception as e:
                    continue
            
            if valid_count > 0:
                det_rate = detected_count / valid_count
                mean_z = total_z / valid_count
                print(f"    Detection: {det_rate*100:.1f}% ({detected_count}/{valid_count})")
                print(f"    Mean Z-score: {mean_z:.2f}")
                
                results.append({
                    "config": config_name,
                    "config_name": config['name'],
                    "alpha": config['alpha'],
                    "omega": config['omega'],
                    "attack": attack_name,
                    "detection_rate": det_rate,
                    "mean_z_score": mean_z,
                    "num_samples": valid_count
                })
    
    # Save results
    results_file = os.path.join(RESULTS_DIR, f"gpw_tuning_{timestamp}.csv")
    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {results_file}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Detection Rates by Configuration and Attack")
    print("=" * 80)
    print(f"{'Config':<30} | {'Clean':>8} | {'Synonym':>8} | {'Swap':>8} | {'Typo':>8} | {'CopyPaste':>10}")
    print("-" * 90)
    
    for config_name, config in GPW_CONFIGS.items():
        row = f"{config['name']:<30} |"
        for attack in ['clean', 'synonym', 'swap', 'typo', 'copypaste']:
            matching = [r for r in results if r['config'] == config_name and r['attack'] == attack]
            if matching:
                row += f" {matching[0]['detection_rate']*100:7.1f}% |"
            else:
                row += f" {'N/A':>7} |"
        print(row)
    
    print("\n" + "=" * 80)
    print("ROBUSTNESS DROP (percentage points from clean)")
    print("=" * 80)
    print(f"{'Config':<30} | {'Synonym':>8} | {'Swap':>8} | {'Typo':>8} | {'CopyPaste':>10}")
    print("-" * 80)
    
    for config_name, config in GPW_CONFIGS.items():
        clean_results = [r for r in results if r['config'] == config_name and r['attack'] == 'clean']
        if not clean_results:
            continue
        clean_rate = clean_results[0]['detection_rate']
        
        row = f"{config['name']:<30} |"
        for attack in ['synonym', 'swap', 'typo', 'copypaste']:
            matching = [r for r in results if r['config'] == config_name and r['attack'] == attack]
            if matching:
                drop = (clean_rate - matching[0]['detection_rate']) * 100
                row += f" {drop:+7.1f}pp |"
            else:
                row += f" {'N/A':>7} |"
        print(row)


if __name__ == "__main__":
    run_tuning_experiments()
