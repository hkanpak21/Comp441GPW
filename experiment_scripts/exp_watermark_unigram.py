#!/usr/bin/env python3
"""
Unigram Watermark Experiments - All attacks

Usage:
    python exp_watermark_unigram.py --model opt-1.3b --num_samples 200
"""

import sys
import os
import time
import csv
import json
import hashlib
from datetime import datetime
import importlib.util

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Load modules
config_path = os.path.join(script_dir, 'config.py')
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

utils_path = os.path.join(script_dir, 'utils.py')
spec = importlib.util.spec_from_file_location("utils", utils_path)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

dataset_loader_path = os.path.join(script_dir, 'dataset_loader.py')
spec = importlib.util.spec_from_file_location("dataset_loader", dataset_loader_path)
dataset_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_loader)

import torch
import numpy as np
from scipy import stats

MODEL_CONFIGS = config.MODEL_CONFIGS
WATERMARK_PARAMS = config.WATERMARK_PARAMS
GENERATION_PARAMS = config.GENERATION_PARAMS
RESULTS_DIR = config.RESULTS_DIR

load_model_and_tokenizer = utils.load_model_and_tokenizer
load_watermarker = utils.load_watermarker
load_attack = utils.load_attack
load_c4_dataset = dataset_loader.load_c4_dataset


def compute_perplexity(text, model, tokenizer, device):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            return min(torch.exp(outputs.loss).item(), 1000.0)
    except:
        return -1.0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="opt-1.3b")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    wm_name = "unigram"
    attacks = ["clean", "synonym", "swap", "typo", "copypaste"]
    
    print("=" * 70)
    print(f"UNIGRAM WATERMARK EXPERIMENTS")
    print("=" * 70)
    print(f"Model: {args.model} | Samples: {args.num_samples}")
    print(f"Attacks: {attacks}")
    print("=" * 70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(args.output_dir, f"results_unigram_{args.model}_{timestamp}.csv")
    
    fieldnames = [
        'experiment_id', 'model_family', 'watermark_method', 'watermark_params',
        'text_type', 'attack_name', 'attack_param', 'sample_length',
        'perplexity_score', 'detection_z_score', 'detection_p_value', 'is_detected',
        'prompt', 'generated_text', 'attacked_text'
    ]
    
    # Load data
    print("\n[1/4] Loading dataset...")
    c4_data = load_c4_dataset(args.num_samples)
    prompts = [d['prompt'] for d in c4_data]
    human_texts = [d['text'] for d in c4_data]  # Use full text as human reference
    print(f"✓ Loaded {len(prompts)} samples")
    
    # Load model
    print(f"\n[2/4] Loading model...")
    model_config = MODEL_CONFIGS[args.model]
    model, tokenizer, device = load_model_and_tokenizer(model_config, args.device)
    print(f"✓ Model loaded")
    
    # Load watermarker
    print(f"\n[3/4] Loading Unigram watermarker...")
    wm_params = WATERMARK_PARAMS[wm_name]
    watermarker = load_watermarker(wm_name, model, tokenizer, wm_params, device)
    params_str = f"gamma={wm_params['gamma']}, delta={wm_params['delta']}"
    print(f"✓ Unigram loaded: {params_str}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    csvfile = open(output_csv, 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    # Generate watermarked texts
    print(f"\n[4/4] Running experiments...")
    watermarked_texts = []
    print("  Generating watermarked texts...")
    for idx, prompt in enumerate(prompts):
        if idx % 20 == 0:
            print(f"    Gen: {idx}/{args.num_samples}")
        try:
            text = watermarker.generate(prompt, **GENERATION_PARAMS)
            watermarked_texts.append(text)
        except Exception as e:
            watermarked_texts.append("")
            print(f"    ✗ Gen error {idx}: {str(e)[:40]}")
    
    total = 0
    results = {'z_scores': [], 'detected': [], 'perplexity': []}
    
    for attack_name in attacks:
        print(f"\n  Attack: {attack_name}")
        
        attack = None
        attack_param_str = "N/A"
        
        if attack_name == "synonym":
            attack = load_attack("synonym", {"edit_rate": 0.3})
            attack_param_str = "edit_rate=0.3"
        elif attack_name == "swap":
            attack = load_attack("swap", {"edit_rate": 0.2})
            attack_param_str = "edit_rate=0.2"
        elif attack_name == "typo":
            attack = load_attack("typo", {"edit_rate": 0.1})
            attack_param_str = "edit_rate=0.1"
        elif attack_name == "copypaste":
            attack = load_attack("copypaste", {"n_segments": 3, "watermark_ratio": 0.5})
            attack_param_str = "n_segments=3, ratio=0.5"
        
        for idx, (prompt, wm_text) in enumerate(zip(prompts, watermarked_texts)):
            if idx % 50 == 0:
                print(f"    Progress: {idx}/{args.num_samples}")
            
            if not wm_text:
                continue
            
            try:
                if attack_name == "clean":
                    attacked_text = wm_text
                elif attack_name == "copypaste" and idx < len(human_texts):
                    attacked_text = attack(wm_text, human_text=human_texts[idx])
                elif attack:
                    attacked_text = attack(wm_text)
                else:
                    attacked_text = wm_text
                
                result = watermarker.detect(attacked_text)
                z_score = result.z_score
                p_value = 1 - stats.norm.cdf(z_score)
                is_detected = 1 if p_value < 0.01 else 0
                
                ppl = compute_perplexity(attacked_text, model, tokenizer, device) if attack_name == "clean" else -1
                
                results['z_scores'].append(z_score)
                results['detected'].append(is_detected)
                if ppl > 0:
                    results['perplexity'].append(ppl)
                
                token_count = len(tokenizer.encode(attacked_text))
                
                row = {
                    'experiment_id': hashlib.md5(f"unigram_{attack_name}_{idx}".encode()).hexdigest()[:12],
                    'model_family': model_config['name'],
                    'watermark_method': 'UNIGRAM',
                    'watermark_params': params_str,
                    'text_type': 'generated_watermarked',
                    'attack_name': attack_name.capitalize() if attack_name != 'copypaste' else 'CopyPaste',
                    'attack_param': attack_param_str,
                    'sample_length': token_count,
                    'perplexity_score': round(ppl, 2) if ppl > 0 else -1,
                    'detection_z_score': round(z_score, 4),
                    'detection_p_value': round(p_value, 6),
                    'is_detected': is_detected,
                    'prompt': prompt[:200],
                    'generated_text': wm_text[:300],
                    'attacked_text': attacked_text[:300] if attack_name != "clean" else ""
                }
                writer.writerow(row)
                total += 1
                
            except Exception as e:
                print(f"    ✗ Error {idx}: {str(e)[:40]}")
    
    csvfile.close()
    
    # Summary
    summary = {
        'watermark': 'UNIGRAM',
        'model': args.model,
        'params': params_str,
        'num_samples': args.num_samples,
        'total_experiments': total,
        'mean_z_score': round(np.mean(results['z_scores']), 3),
        'detection_rate': round(np.mean(results['detected']), 4),
        'mean_perplexity': round(np.mean(results['perplexity']), 2) if results['perplexity'] else -1
    }
    
    with open(output_csv.replace('.csv', '_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("UNIGRAM EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"Total: {total} | Z-score: {summary['mean_z_score']} | Det: {summary['detection_rate']*100:.1f}%")
    print(f"Results: {output_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
