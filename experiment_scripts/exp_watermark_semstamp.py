#!/usr/bin/env python3
"""
SEMSTAMP Watermark Experiments

Runs SemStamp (Semantic Watermarking) experiments with max_rejections=1.
Tests clean detection and attack robustness.
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
from watermarkers import SEMSTAMPWatermark
from attacks import SynonymAttack, SwapAttack, TypoAttack, CopyPasteAttack

import torch
import nltk
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# Configuration  
NUM_SAMPLES = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# SEMSTAMP config - max_rejections=1 per user requirement
SEMSTAMP_CONFIG = {
    "lsh_dim": 3,
    "z_threshold": 2.0,  # Lower threshold for sentence-level detection
    "max_rejections": 1,  # User requirement: only 1 attempt for speed
    "margin": 0.0,  # Disabled for speed
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
    except Exception as e:
        return -1.0


def run_semstamp_experiments(model_key="opt-1.3b"):
    """Run SEMSTAMP experiments for a given model."""
    
    # Model configurations
    MODEL_CONFIGS = {
        "gpt2": {"name": "gpt2", "dtype": torch.float32},
        "opt-1.3b": {"name": "facebook/opt-1.3b", "dtype": torch.float16},
        "qwen-7b": {"name": "Qwen/Qwen2.5-7B-Instruct", "dtype": torch.float16},
        "qwen-14b": {"name": "Qwen/Qwen2.5-14B-Instruct", "dtype": torch.float16},
    }
    
    model_config = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS["opt-1.3b"])
    model_name = model_config["name"]
    model_dtype = model_config["dtype"]
    
    print("=" * 80)
    print(f"SEMSTAMP EXPERIMENTS - {model_key}")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Samples: {NUM_SAMPLES}")
    print(f"Max Rejections: {SEMSTAMP_CONFIG['max_rejections']}")
    print(f"Device: {DEVICE}")
    print()
    
    # Load model
    print("Loading language model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=model_dtype if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load sentence embedder
    print("Loading sentence embedder...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embedder = embedder.to(DEVICE)
    
    # Create SEMSTAMP watermarker
    print("Creating SEMSTAMP watermarker...")
    watermarker = SEMSTAMPWatermark(
        model=model,
        tokenizer=tokenizer,
        embedder=embedder,
        lsh_dim=SEMSTAMP_CONFIG["lsh_dim"],
        z_threshold=SEMSTAMP_CONFIG["z_threshold"],
        max_rejections=SEMSTAMP_CONFIG["max_rejections"],
        margin=SEMSTAMP_CONFIG["margin"],
        device=DEVICE,
    )
    
    # Load dataset
    print("Loading dataset...")
    c4_data = load_c4_dataset(num_samples=NUM_SAMPLES)
    prompts = [d['prompt'] for d in c4_data]
    human_texts = [d['text'] for d in c4_data]
    print(f"Loaded {len(prompts)} samples")
    
    # Results storage
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate watermarked texts
    print(f"\nGenerating {NUM_SAMPLES} watermarked texts...")
    watermarked_texts = []
    generation_times = []
    
    for i, prompt in enumerate(prompts):
        try:
            start_time = time.time()
            wm_text = watermarker.generate(
                prompt,
                max_new_tokens=120,
                temperature=1.0,
                top_p=0.95,
            )
            gen_time = time.time() - start_time
            watermarked_texts.append(wm_text)
            generation_times.append(gen_time)
            
            if (i + 1) % 10 == 0:
                avg_time = sum(generation_times) / len(generation_times)
                print(f"  Generated {i+1}/{NUM_SAMPLES} (avg {avg_time:.2f}s/sample)")
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            watermarked_texts.append("")
            generation_times.append(0)
    
    # Compute perplexity for clean watermarked texts
    print("\nComputing perplexities...")
    perplexities = []
    for i, wm_text in enumerate(watermarked_texts):
        if wm_text:
            ppl = compute_perplexity(wm_text, model, tokenizer, DEVICE)
            perplexities.append(ppl)
        else:
            perplexities.append(-1.0)
    
    valid_perplexities = [p for p in perplexities if p > 0]
    mean_perplexity = sum(valid_perplexities) / len(valid_perplexities) if valid_perplexities else -1.0
    print(f"Mean perplexity: {mean_perplexity:.2f}")
    
    # Test detection with and without attacks
    for attack_name, attack in ATTACKS.items():
        print(f"\nTesting attack: {attack_name}")
        
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
                
                # Store individual result
                results.append({
                    "experiment_id": hashlib.md5(f"{i}_{attack_name}".encode()).hexdigest()[:12],
                    "model_family": model_name,
                    "watermark_method": "SEMSTAMP",
                    "watermark_params": f"lsh_dim={SEMSTAMP_CONFIG['lsh_dim']}, max_rej={SEMSTAMP_CONFIG['max_rejections']}",
                    "text_type": "generated_watermarked",
                    "attack_name": attack_name.capitalize() if attack else "Clean",
                    "attack_param": "N/A" if attack is None else str(ATTACKS[attack_name].__dict__ if hasattr(ATTACKS[attack_name], '__dict__') else {}),
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
                print(f"  Detection error on sample {i}: {e}")
                continue
        
        if valid_count > 0:
            det_rate = detected_count / valid_count
            mean_z = total_z / valid_count
            print(f"  Detection: {det_rate*100:.1f}% ({detected_count}/{valid_count})")
            print(f"  Mean Z-score: {mean_z:.2f}")
    
    # Save results
    results_file = os.path.join(RESULTS_DIR, f"results_semstamp_{model_key}_{timestamp}.csv")
    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {results_file}")
    
    # Save summary
    summary = {
        "watermark": "SEMSTAMP",
        "model": model_key,
        "params": f"lsh_dim={SEMSTAMP_CONFIG['lsh_dim']}, max_rejections={SEMSTAMP_CONFIG['max_rejections']}",
        "num_samples": NUM_SAMPLES,
        "total_experiments": len(results),
        "mean_generation_time": sum(generation_times) / len(generation_times) if generation_times else 0,
        "mean_perplexity": mean_perplexity,
    }
    
    # Add attack-specific stats
    for attack_name in ATTACKS.keys():
        attack_results = [r for r in results if r['attack_name'].lower() == attack_name.lower() or 
                         (attack_name == 'clean' and r['attack_name'] == 'Clean')]
        if attack_results:
            summary[f"{attack_name}_detection_rate"] = sum(r['is_detected'] for r in attack_results) / len(attack_results)
            summary[f"{attack_name}_mean_z"] = sum(r['detection_z_score'] for r in attack_results) / len(attack_results)
    
    summary_file = results_file.replace('.csv', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")
    
    return results


if __name__ == "__main__":
    # Parse model argument
    model_key = sys.argv[1] if len(sys.argv) > 1 else "opt-1.3b"
    run_semstamp_experiments(model_key)
