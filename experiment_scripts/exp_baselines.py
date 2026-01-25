#!/usr/bin/env python3
"""
Human + Unwatermarked AI Baselines

Generates baseline detection results for:
- Human text (should have low z-scores, ~0% detection)
- AI-generated text without watermark (should have low z-scores)

Usage:
    python exp_baselines.py --model opt-1.3b --num_samples 200
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
    
    print("=" * 70)
    print(f"BASELINE EXPERIMENTS (Human + Unwatermarked AI)")
    print("=" * 70)
    print(f"Model: {args.model} | Samples: {args.num_samples}")
    print("=" * 70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(args.output_dir, f"results_baselines_{args.model}_{timestamp}.csv")
    
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
    print(f"✓ Loaded {len(prompts)} prompts, {len(human_texts)} human texts")
    
    # Load model
    print(f"\n[2/4] Loading model...")
    model_config = MODEL_CONFIGS[args.model]
    model, tokenizer, device = load_model_and_tokenizer(model_config, args.device)
    print(f"✓ Model loaded")
    
    # Load a watermarker just for detection (to measure false positive rate)
    print(f"\n[3/4] Loading watermarkers for detection...")
    watermarkers = {}
    for wm_name in ['unigram', 'kgw']:
        wm_params = WATERMARK_PARAMS[wm_name]
        watermarkers[wm_name] = load_watermarker(wm_name, model, tokenizer, wm_params, device)
    # GPW-SP
    wm_params = WATERMARK_PARAMS['gpw_sp']
    watermarkers['gpw_sp'] = load_watermarker("gpw", model, tokenizer, wm_params, device)
    print(f"✓ Loaded detectors: {list(watermarkers.keys())}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    csvfile = open(output_csv, 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    total = 0
    results = {
        'human': {'perplexity': [], 'z_scores': {k: [] for k in watermarkers}, 'detected': {k: [] for k in watermarkers}},
        'ai_clean': {'perplexity': [], 'z_scores': {k: [] for k in watermarkers}, 'detected': {k: [] for k in watermarkers}}
    }
    
    # Part 1: Human text baselines
    print(f"\n[4/4] Running experiments...")
    print("\n  === Human Text Baselines ===")
    for idx, human_text in enumerate(human_texts[:args.num_samples]):
        if idx % 50 == 0:
            print(f"    Human: {idx}/{args.num_samples}")
        
        ppl = compute_perplexity(human_text, model, tokenizer, device)
        results['human']['perplexity'].append(ppl)
        token_count = len(tokenizer.encode(human_text))
        
        # Test against each watermarker
        for wm_name, watermarker in watermarkers.items():
            try:
                result = watermarker.detect(human_text)
                z_score = result.z_score
                p_value = 1 - stats.norm.cdf(z_score)
                is_detected = 1 if p_value < 0.01 else 0
                
                results['human']['z_scores'][wm_name].append(z_score)
                results['human']['detected'][wm_name].append(is_detected)
                
                row = {
                    'experiment_id': hashlib.md5(f"human_{wm_name}_{idx}".encode()).hexdigest()[:12],
                    'model_family': model_config['name'],
                    'watermark_method': wm_name.upper().replace('_', '-'),
                    'watermark_params': 'N/A (detection only)',
                    'text_type': 'human_gold',
                    'attack_name': 'Clean',
                    'attack_param': 'N/A',
                    'sample_length': token_count,
                    'perplexity_score': round(ppl, 2) if ppl > 0 else -1,
                    'detection_z_score': round(z_score, 4),
                    'detection_p_value': round(p_value, 6),
                    'is_detected': is_detected,
                    'prompt': '',
                    'generated_text': human_text[:300],
                    'attacked_text': ''
                }
                writer.writerow(row)
                total += 1
            except Exception as e:
                print(f"    ✗ Human detect error {wm_name}/{idx}: {str(e)[:30]}")
    
    # Part 2: Unwatermarked AI text
    print("\n  === Unwatermarked AI Text ===")
    print("    Generating unwatermarked texts...")
    
    unwatermarked_texts = []
    for idx, prompt in enumerate(prompts):
        if idx % 20 == 0:
            print(f"    Gen: {idx}/{args.num_samples}")
        try:
            # Generate without watermark using raw model
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=GENERATION_PARAMS.get('max_new_tokens', 120),
                    temperature=GENERATION_PARAMS.get('temperature', 1.0),
                    top_k=GENERATION_PARAMS.get('top_k', 50),
                    top_p=GENERATION_PARAMS.get('top_p', 0.95),
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            unwatermarked_texts.append(text)
        except Exception as e:
            unwatermarked_texts.append("")
            print(f"    ✗ Gen error {idx}: {str(e)[:30]}")
    
    print("    Running detection on unwatermarked texts...")
    for idx, (prompt, ai_text) in enumerate(zip(prompts, unwatermarked_texts)):
        if idx % 50 == 0:
            print(f"    AI Clean: {idx}/{args.num_samples}")
        
        if not ai_text:
            continue
        
        ppl = compute_perplexity(ai_text, model, tokenizer, device)
        results['ai_clean']['perplexity'].append(ppl)
        token_count = len(tokenizer.encode(ai_text))
        
        for wm_name, watermarker in watermarkers.items():
            try:
                result = watermarker.detect(ai_text)
                z_score = result.z_score
                p_value = 1 - stats.norm.cdf(z_score)
                is_detected = 1 if p_value < 0.01 else 0
                
                results['ai_clean']['z_scores'][wm_name].append(z_score)
                results['ai_clean']['detected'][wm_name].append(is_detected)
                
                row = {
                    'experiment_id': hashlib.md5(f"ai_clean_{wm_name}_{idx}".encode()).hexdigest()[:12],
                    'model_family': model_config['name'],
                    'watermark_method': wm_name.upper().replace('_', '-'),
                    'watermark_params': 'N/A (no watermark)',
                    'text_type': 'generated_clean',
                    'attack_name': 'Clean',
                    'attack_param': 'N/A',
                    'sample_length': token_count,
                    'perplexity_score': round(ppl, 2) if ppl > 0 else -1,
                    'detection_z_score': round(z_score, 4),
                    'detection_p_value': round(p_value, 6),
                    'is_detected': is_detected,
                    'prompt': prompt[:200],
                    'generated_text': ai_text[:300],
                    'attacked_text': ''
                }
                writer.writerow(row)
                total += 1
            except Exception as e:
                print(f"    ✗ AI detect error {wm_name}/{idx}: {str(e)[:30]}")
    
    csvfile.close()
    
    # Summary
    summary = {
        'model': args.model,
        'num_samples': args.num_samples,
        'total_experiments': total,
        'human': {
            'mean_perplexity': round(np.mean([p for p in results['human']['perplexity'] if p > 0]), 2),
            'false_positive_rates': {
                wm: round(np.mean(results['human']['detected'][wm]), 4) 
                for wm in watermarkers
            },
            'mean_z_scores': {
                wm: round(np.mean(results['human']['z_scores'][wm]), 3) 
                for wm in watermarkers
            }
        },
        'ai_clean': {
            'mean_perplexity': round(np.mean([p for p in results['ai_clean']['perplexity'] if p > 0]), 2),
            'false_positive_rates': {
                wm: round(np.mean(results['ai_clean']['detected'][wm]), 4) 
                for wm in watermarkers
            },
            'mean_z_scores': {
                wm: round(np.mean(results['ai_clean']['z_scores'][wm]), 3) 
                for wm in watermarkers
            }
        }
    }
    
    with open(output_csv.replace('.csv', '_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("BASELINE EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"Total experiments: {total}")
    print(f"\nHuman Text (should be ~0% FPR):")
    for wm in watermarkers:
        print(f"  {wm}: Z={summary['human']['mean_z_scores'][wm]:.2f}, FPR={summary['human']['false_positive_rates'][wm]*100:.1f}%")
    print(f"\nAI Clean (should be ~0% FPR):")
    for wm in watermarkers:
        print(f"  {wm}: Z={summary['ai_clean']['mean_z_scores'][wm]:.2f}, FPR={summary['ai_clean']['false_positive_rates'][wm]*100:.1f}%")
    print(f"\nResults: {output_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
