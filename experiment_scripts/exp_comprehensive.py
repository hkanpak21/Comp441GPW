#!/usr/bin/env python3
"""
Comprehensive Watermarking Experiments

Runs all watermarkers (Unigram, KGW, GPW-SP) on a specified model.
Produces results for detection quality and attack robustness.
"""

import os
import sys
import time
import hashlib
import csv
import json
from datetime import datetime
import importlib.util

# Get project root and add to path
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir)
sys.path.insert(0, PROJECT_ROOT)

# Load dataset loader
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

dataset_loader = load_module("dataset_loader", os.path.join(script_dir, "dataset_loader.py"))
load_c4_dataset = dataset_loader.load_c4_dataset

# Import watermarkers and attacks using package imports
from watermarkers import UnigramWatermark, KGWWatermark, create_gpw_variant
from watermarkers.gpw import GPWConfig, SRConfig, GPWWatermark
from attacks import SynonymAttack, SwapAttack, TypoAttack, CopyPasteAttack

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 200
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Model configurations
MODEL_CONFIGS = {
    "gpt2": {
        "name": "gpt2",
        "dtype": torch.float32,
        "max_tokens": 120,
    },
    "opt-1.3b": {
        "name": "facebook/opt-1.3b",
        "dtype": torch.float16,
        "max_tokens": 120,
    },
    "qwen-7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "dtype": torch.float16,
        "max_tokens": 120,
    },
    "qwen-14b": {
        "name": "Qwen/Qwen2.5-14B-Instruct",
        "dtype": torch.float16,
        "max_tokens": 120,
    },
}

# Watermarker configurations
# Testing GPW with HIGH omega for robustness (user insight: higher omega = more robust)
# The grid search will help find optimal values
WATERMARKER_CONFIGS = {
    "unigram": {
        "gamma": 0.5,
        "delta": 2.0,
        "z_threshold": 4.0,
    },
    "kgw": {
        "gamma": 0.5,
        "delta": 2.0,
        "z_threshold": 4.0,
        "seeding_scheme": "simple_1",
    },
    # GPW baseline (no salting)
    "gpw": {
        "variant": "GPW",
        "alpha": 3.0,
        "omega": 20.0,  # HIGH omega for robustness
        "z_threshold": 4.0,
    },
    # GPW-SP (salted phase - context-dependent) - LOW omega baseline
    "gpw_sp_low": {
        "variant": "GPW-SP",
        "alpha": 3.0,
        "omega": 2.0,  # Original low omega for comparison
        "z_threshold": 4.0,
    },
    # GPW-SP with HIGH omega - expected to be MORE ROBUST
    "gpw_sp_high": {
        "variant": "GPW-SP",
        "alpha": 3.0,
        "omega": 50.0,  # HIGH omega for maximum robustness
        "z_threshold": 4.0,
    },
}

# Attacks
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
    except:
        return -1.0


def create_watermarker(wm_type, model, tokenizer, device):
    """Create a watermarker instance."""
    if wm_type == "unigram":
        config = WATERMARKER_CONFIGS["unigram"]
        return UnigramWatermark(
            model=model,
            tokenizer=tokenizer,
            gamma=config["gamma"],
            delta=config["delta"],
            z_threshold=config["z_threshold"],
            device=device,
        )
    elif wm_type == "kgw":
        config = WATERMARKER_CONFIGS["kgw"]
        return KGWWatermark(
            model=model,
            tokenizer=tokenizer,
            gamma=config["gamma"],
            delta=config["delta"],
            z_threshold=config["z_threshold"],
            seeding_scheme=config["seeding_scheme"],
            device=device,
        )
    elif wm_type.startswith("gpw"):
        # Handle all GPW variants: gpw, gpw_sp_low, gpw_sp_high, etc.
        config = WATERMARKER_CONFIGS[wm_type]
        gpw_cfg = GPWConfig(
            alpha=config["alpha"],
            omega=config["omega"],
            salted=config["variant"] != "GPW",  # GPW-SP variants have salted=True
            ctx_mode="ngram",
            ngram=4,
        )
        sr_cfg = SRConfig(enabled="SR" in config["variant"])
        return GPWWatermark(
            model=model,
            tokenizer=tokenizer,
            gpw_cfg=gpw_cfg,
            sr_cfg=sr_cfg,
            hash_key=hashlib.sha256(b"gpw-comprehensive-exp").digest(),
            device=device,
        )
    else:
        raise ValueError(f"Unknown watermarker type: {wm_type}")


def run_watermarker_experiments(model_key, wm_type, prompts, human_texts, model, tokenizer):
    """Run experiments for a single watermarker."""
    
    print(f"\n{'='*60}")
    print(f"Testing: {wm_type.upper()}")
    print("=" * 60)
    
    # Create watermarker
    watermarker = create_watermarker(wm_type, model, tokenizer, DEVICE)
    config = WATERMARKER_CONFIGS.get(wm_type, {})
    config_str = ", ".join(f"{k}={v}" for k, v in config.items())
    
    results = []
    model_config = MODEL_CONFIGS[model_key]
    
    # Generate watermarked texts
    print(f"Generating {len(prompts)} watermarked texts...")
    watermarked_texts = []
    perplexities = []
    
    for i, prompt in enumerate(prompts):
        try:
            wm_text = watermarker.generate(
                prompt,
                max_new_tokens=model_config["max_tokens"],
                temperature=1.0,
                top_k=50,
                top_p=0.95,
            )
            # Remove prompt if present
            if wm_text.startswith(prompt):
                wm_text = wm_text[len(prompt):].strip()
            watermarked_texts.append(wm_text)
            
            # Compute perplexity
            ppl = compute_perplexity(wm_text, model, tokenizer, DEVICE)
            perplexities.append(ppl)
            
            if (i + 1) % 20 == 0:
                print(f"  Generated {i+1}/{len(prompts)}")
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            watermarked_texts.append("")
            perplexities.append(-1.0)
    
    # Test each attack
    for attack_name, attack in ATTACKS.items():
        print(f"\n  Attack: {attack_name}")
        
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
                det_result = watermarker.detect(attacked_text)
                
                results.append({
                    "experiment_id": hashlib.md5(f"{wm_type}_{i}_{attack_name}".encode()).hexdigest()[:12],
                    "model_family": model_config["name"],
                    "watermark_method": wm_type.upper(),
                    "watermark_params": config_str,
                    "text_type": "generated_watermarked",
                    "attack_name": attack_name.capitalize() if attack else "Clean",
                    "attack_param": "N/A",
                    "sample_length": len(attacked_text.split()),
                    "perplexity_score": perplexities[i] if attack is None else -1.0,
                    "detection_z_score": det_result.z_score,
                    "detection_p_value": det_result.p_value,
                    "is_detected": 1 if det_result.is_watermarked else 0,
                    "prompt": prompts[i][:200],
                    "generated_text": wm_text[:500] if attack is None else "",
                    "attacked_text": attacked_text[:500] if attack else "",
                })
                
                if det_result.is_watermarked:
                    detected_count += 1
                total_z += det_result.z_score
                valid_count += 1
                
            except Exception as e:
                continue
        
        if valid_count > 0:
            det_rate = detected_count / valid_count
            mean_z = total_z / valid_count
            print(f"    Detection: {det_rate*100:.1f}% ({detected_count}/{valid_count}), Mean Z: {mean_z:.2f}")
    
    return results


def run_baseline_experiments(model_key, prompts, human_texts, model, tokenizer):
    """Run baseline experiments (human + unwatermarked AI)."""
    
    print(f"\n{'='*60}")
    print("Testing: BASELINES (Human + Unwatermarked AI)")
    print("=" * 60)
    
    results = []
    model_config = MODEL_CONFIGS[model_key]
    
    # Create detectors for false positive testing
    detectors = {
        "unigram": create_watermarker("unigram", model, tokenizer, DEVICE),
        "kgw": create_watermarker("kgw", model, tokenizer, DEVICE),
        "gpw_sp": create_watermarker("gpw_sp", model, tokenizer, DEVICE),
    }
    
    # Test human texts
    print(f"\nTesting {len(human_texts)} human texts...")
    for i, human_text in enumerate(human_texts):
        if not human_text:
            continue
            
        ppl = compute_perplexity(human_text, model, tokenizer, DEVICE)
        
        for det_name, detector in detectors.items():
            try:
                det_result = detector.detect(human_text)
                results.append({
                    "experiment_id": hashlib.md5(f"human_{i}_{det_name}".encode()).hexdigest()[:12],
                    "model_family": model_config["name"],
                    "watermark_method": det_name.upper(),
                    "watermark_params": "detector",
                    "text_type": "human_gold",
                    "attack_name": "N/A",
                    "attack_param": "N/A",
                    "sample_length": len(human_text.split()),
                    "perplexity_score": ppl,
                    "detection_z_score": det_result.z_score,
                    "detection_p_value": det_result.p_value,
                    "is_detected": 1 if det_result.is_watermarked else 0,
                    "prompt": "",
                    "generated_text": "",
                    "attacked_text": "",
                })
            except:
                continue
        
        if (i + 1) % 50 == 0:
            print(f"  Tested {i+1}/{len(human_texts)} human texts")
    
    # Generate and test unwatermarked AI texts
    print(f"\nGenerating {len(prompts)} unwatermarked AI texts...")
    for i, prompt in enumerate(prompts):
        try:
            # Generate without watermark
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=model_config["max_tokens"],
                    temperature=1.0,
                    top_k=50,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            ai_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if ai_text.startswith(prompt):
                ai_text = ai_text[len(prompt):].strip()
            
            if not ai_text:
                continue
            
            ppl = compute_perplexity(ai_text, model, tokenizer, DEVICE)
            
            for det_name, detector in detectors.items():
                try:
                    det_result = detector.detect(ai_text)
                    results.append({
                        "experiment_id": hashlib.md5(f"ai_{i}_{det_name}".encode()).hexdigest()[:12],
                        "model_family": model_config["name"],
                        "watermark_method": det_name.upper(),
                        "watermark_params": "detector",
                        "text_type": "generated_clean",
                        "attack_name": "N/A",
                        "attack_param": "N/A",
                        "sample_length": len(ai_text.split()),
                        "perplexity_score": ppl,
                        "detection_z_score": det_result.z_score,
                        "detection_p_value": det_result.p_value,
                        "is_detected": 1 if det_result.is_watermarked else 0,
                        "prompt": prompt[:200],
                        "generated_text": ai_text[:500],
                        "attacked_text": "",
                    })
                except:
                    continue
            
            if (i + 1) % 50 == 0:
                print(f"  Generated {i+1}/{len(prompts)} AI texts")
        except Exception as e:
            continue
    
    return results


def run_comprehensive_experiments(model_key="opt-1.3b"):
    """Run all experiments for a given model."""
    
    model_config = MODEL_CONFIGS.get(model_key)
    if not model_config:
        print(f"Unknown model: {model_key}")
        print(f"Available: {list(MODEL_CONFIGS.keys())}")
        return
    
    print("=" * 80)
    print(f"COMPREHENSIVE WATERMARKING EXPERIMENTS - {model_key}")
    print("=" * 80)
    print(f"Model: {model_config['name']}")
    print(f"Samples: {NUM_SAMPLES}")
    print(f"Watermarkers: {list(WATERMARKER_CONFIGS.keys())}")
    print(f"Device: {DEVICE}")
    print()
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        torch_dtype=model_config["dtype"] if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print("Loading dataset...")
    c4_data = load_c4_dataset(num_samples=NUM_SAMPLES)
    prompts = [d['prompt'] for d in c4_data]
    human_texts = [d['text'] for d in c4_data]
    print(f"Loaded {len(prompts)} samples")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    
    # Run experiments for each watermarker
    for wm_type in WATERMARKER_CONFIGS.keys():
        try:
            results = run_watermarker_experiments(
                model_key, wm_type, prompts, human_texts, model, tokenizer
            )
            all_results.extend(results)
            
            # Save intermediate results
            wm_file = os.path.join(RESULTS_DIR, f"results_{wm_type}_{model_key}_{timestamp}.csv")
            if results:
                with open(wm_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
                print(f"  Saved: {wm_file}")
        except Exception as e:
            print(f"  Error with {wm_type}: {e}")
    
    # Run baseline experiments
    try:
        baseline_results = run_baseline_experiments(
            model_key, prompts, human_texts, model, tokenizer
        )
        all_results.extend(baseline_results)
        
        baseline_file = os.path.join(RESULTS_DIR, f"results_baselines_{model_key}_{timestamp}.csv")
        if baseline_results:
            with open(baseline_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=baseline_results[0].keys())
                writer.writeheader()
                writer.writerows(baseline_results)
            print(f"  Saved: {baseline_file}")
    except Exception as e:
        print(f"  Error with baselines: {e}")
    
    # Save comprehensive results
    all_file = os.path.join(RESULTS_DIR, f"results_all_{model_key}_{timestamp}.csv")
    if all_results:
        with open(all_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nAll results saved to: {all_file}")
    
    # Generate summary
    print_summary(all_results, model_key)


def print_summary(results, model_key):
    """Print summary of results."""
    print("\n" + "=" * 80)
    print(f"SUMMARY - {model_key}")
    print("=" * 80)
    
    # Group by watermarker
    watermarkers = set(r['watermark_method'] for r in results if r['text_type'] == 'generated_watermarked')
    
    print(f"\n{'Watermarker':<15} | {'Clean':>8} | {'Synonym':>8} | {'Swap':>8} | {'Typo':>8} | {'CopyPaste':>10} | {'PPL':>6}")
    print("-" * 85)
    
    for wm in sorted(watermarkers):
        wm_results = [r for r in results if r['watermark_method'] == wm and r['text_type'] == 'generated_watermarked']
        
        row = f"{wm:<15} |"
        for attack in ['Clean', 'Synonym', 'Swap', 'Typo', 'Copypaste']:
            attack_results = [r for r in wm_results if r['attack_name'] == attack]
            if attack_results:
                det_rate = sum(r['is_detected'] for r in attack_results) / len(attack_results)
                row += f" {det_rate*100:7.1f}% |"
            else:
                row += f" {'N/A':>7} |"
        
        # Perplexity
        clean_results = [r for r in wm_results if r['attack_name'] == 'Clean' and r['perplexity_score'] > 0]
        if clean_results:
            mean_ppl = sum(r['perplexity_score'] for r in clean_results) / len(clean_results)
            row += f" {mean_ppl:5.1f}"
        else:
            row += f" {'N/A':>5}"
        
        print(row)
    
    # False positive rates
    print(f"\n{'False Positive Rates (Human/AI)':<15}")
    print("-" * 50)
    
    for wm in sorted(watermarkers):
        human_results = [r for r in results if r['watermark_method'] == wm and r['text_type'] == 'human_gold']
        ai_results = [r for r in results if r['watermark_method'] == wm and r['text_type'] == 'generated_clean']
        
        human_fp = sum(r['is_detected'] for r in human_results) / len(human_results) if human_results else 0
        ai_fp = sum(r['is_detected'] for r in ai_results) / len(ai_results) if ai_results else 0
        
        print(f"{wm:<15} | Human: {human_fp*100:5.1f}% | AI: {ai_fp*100:5.1f}%")


if __name__ == "__main__":
    model_key = sys.argv[1] if len(sys.argv) > 1 else "opt-1.3b"
    run_comprehensive_experiments(model_key)
