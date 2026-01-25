#!/usr/bin/env python3
"""
Comprehensive Watermarking Experiment - All Models, All Methods

This script runs:
1. Baseline (no watermark) - for comparison
2. GPW (non-salted, omega=50) - OPTIMAL
3. GPW-SP (salted, omega=50)
4. GPW-SP-LOW (salted, omega=2)
5. Unigram
6. KGW

Models: OPT-1.3B, GPT-2, Qwen2.5-7B, Pythia variants

Metrics: Detection Rate, Perplexity
Attacks: clean, synonym_30, swap_20, typo_10, copypaste_50
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Set offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, '/scratch/hkanpak21/Comp441GPW')

from transformers import AutoModelForCausalLM, AutoTokenizer
from watermarkers.gpw import GPWWatermark, GPWConfig, SRConfig
from watermarkers.unigram import UnigramWatermark
from watermarkers.kgw import KGWWatermark
from attacks.lexical import SynonymAttack, SwapAttack, TypoAttack

# Results directory
RESULTS_DIR = Path("/scratch/hkanpak21/Comp441GPW/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Attack configurations
ATTACKS = {
    "clean": None,
    "synonym_30": {"type": "synonym", "ratio": 0.3},
    "swap_20": {"type": "swap", "ratio": 0.2},
    "typo_10": {"type": "typo", "ratio": 0.1},
    "copypaste_50": {"type": "copypaste", "ratio": 0.5},
}

# GPW optimal parameters
GPW_ALPHA = 3.0
GPW_OMEGA = 50.0
GPW_Z_THRESHOLD = 4.0


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Found {n_gpus} GPU(s)")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return "cuda"
    return "cpu"


def load_model_and_tokenizer(model_path, num_params=None, device="cuda"):
    """Load model and tokenizer from path."""
    print(f"\nLoading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    load_kwargs = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    
    # For 7B+ models, use device_map
    if num_params and num_params >= 6_000_000_000:
        load_kwargs["device_map"] = "auto"
        print(f"  Using device_map='auto' for large model")
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    
    if "device_map" not in load_kwargs:
        model = model.to(device)
    
    model.eval()
    actual_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {actual_params/1e6:.1f}M parameters")
    
    return model, tokenizer


def load_prompts(num_samples=100):
    """Load prompts for generation."""
    base_prompts = [
        "The history of artificial intelligence began in",
        "In recent years, climate change has become",
        "The development of modern computing started when",
        "Scientists have discovered that the human brain",
        "The economic impact of technology on society",
        "Education systems around the world are",
        "The future of renewable energy looks",
        "Medical research has shown that regular exercise",
        "The role of social media in modern communication",
        "Space exploration has advanced significantly since",
        "The importance of biodiversity cannot be",
        "Modern agriculture relies heavily on",
        "The psychology of decision making involves",
        "Transportation infrastructure in major cities",
        "The evolution of language throughout human history",
        "Digital privacy concerns have grown as",
        "The relationship between art and technology",
        "Global trade patterns have shifted due to",
        "The science of nutrition has evolved to",
        "Urban planning faces new challenges as",
    ]
    
    prompts = [base_prompts[i % len(base_prompts)] for i in range(num_samples)]
    return prompts


def apply_attack(text, attack_config):
    """Apply attack to text."""
    if attack_config is None:
        return text
    
    attack_type = attack_config["type"]
    ratio = attack_config["ratio"]
    
    if attack_type == "synonym":
        attacker = SynonymAttack(edit_rate=ratio)
        result = attacker.attack(text)
        return result.attacked_text
    elif attack_type == "swap":
        attacker = SwapAttack(edit_rate=ratio)
        result = attacker.attack(text)
        return result.attacked_text
    elif attack_type == "typo":
        attacker = TypoAttack(edit_rate=ratio)
        result = attacker.attack(text)
        return result.attacked_text
    elif attack_type == "copypaste":
        # For copypaste, take first (1-ratio) of text as "human" portion
        words = text.split()
        n_words = len(words)
        human_portion = int(n_words * ratio)  # ratio of text to replace with "human"
        # Just keep the first portion as watermarked
        keep_words = words[:n_words - human_portion]
        return ' '.join(keep_words)
    else:
        return text


def calculate_perplexity(model, tokenizer, text, device):
    """Calculate perplexity."""
    try:
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encodings.input_ids.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        
        return torch.exp(loss).item()
    except Exception as e:
        return float('nan')


def run_baseline(model, tokenizer, prompts, device, model_name, num_samples=100):
    """Run baseline generation without watermarking."""
    print(f"\n{'='*70}")
    print(f"BASELINE (No Watermark) - {model_name}")
    print(f"{'='*70}")
    
    results = []
    ppl_device = next(model.parameters()).device
    
    for i, prompt in enumerate(tqdm(prompts[:num_samples], desc="Generating baseline")):
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(ppl_device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=1.0,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate perplexity
            ppl = calculate_perplexity(model, tokenizer, text, ppl_device)
            
            results.append({
                "model": model_name,
                "method": "Baseline",
                "sample_id": i,
                "text_length": len(text.split()),
                "perplexity": ppl,
            })
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue
    
    # Summarize
    if results:
        df = pd.DataFrame(results)
        avg_ppl = df["perplexity"].mean()
        print(f"  Generated: {len(results)} samples")
        print(f"  Avg Perplexity: {avg_ppl:.2f}")
        return {
            "model": model_name,
            "method": "Baseline",
            "num_samples": len(results),
            "avg_perplexity": avg_ppl,
            "detection_clean": "N/A",
            "detection_synonym": "N/A",
            "detection_swap": "N/A",
            "detection_typo": "N/A",
            "detection_copypaste": "N/A",
        }
    return None


def create_watermarker(method, model, tokenizer, device):
    """Create watermarker for the given method."""
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = device
    
    if method == "GPW":
        gpw_cfg = GPWConfig(alpha=GPW_ALPHA, omega=GPW_OMEGA, salted=False)
        return GPWWatermark(model=model, tokenizer=tokenizer, gpw_cfg=gpw_cfg, device=str(model_device))
    elif method == "GPW-SP":
        gpw_cfg = GPWConfig(alpha=GPW_ALPHA, omega=GPW_OMEGA, salted=True)
        return GPWWatermark(model=model, tokenizer=tokenizer, gpw_cfg=gpw_cfg, device=str(model_device))
    elif method == "GPW-SP-LOW":
        gpw_cfg = GPWConfig(alpha=GPW_ALPHA, omega=2.0, salted=True)
        return GPWWatermark(model=model, tokenizer=tokenizer, gpw_cfg=gpw_cfg, device=str(model_device))
    elif method == "Unigram":
        return UnigramWatermark(model=model, tokenizer=tokenizer, device=str(model_device))
    elif method == "KGW":
        return KGWWatermark(model=model, tokenizer=tokenizer, device=str(model_device))
    else:
        raise ValueError(f"Unknown method: {method}")


def run_watermark_experiment(model, tokenizer, prompts, device, model_name, method, num_samples=100):
    """Run watermarking experiment for one method."""
    print(f"\n{'='*70}")
    print(f"WATERMARKER: {method} - {model_name}")
    print(f"{'='*70}")
    
    try:
        wm = create_watermarker(method, model, tokenizer, device)
    except Exception as e:
        print(f"  ERROR creating watermarker: {e}")
        return None
    
    ppl_device = next(model.parameters()).device
    
    # Generate watermarked texts
    generated_texts = []
    perplexities = []
    
    for i, prompt in enumerate(tqdm(prompts[:num_samples], desc=f"Generating {method}")):
        try:
            text = wm.generate(prompt, max_new_tokens=200)
            generated_texts.append(text)
            
            ppl = calculate_perplexity(model, tokenizer, text, ppl_device)
            if not np.isnan(ppl) and ppl < 1000:
                perplexities.append(ppl)
        except Exception as e:
            generated_texts.append(None)
            continue
    
    valid_texts = [t for t in generated_texts if t is not None]
    print(f"  Generated: {len(valid_texts)}/{num_samples} samples")
    print(f"  Avg Perplexity: {np.mean(perplexities):.2f}" if perplexities else "  Perplexity: N/A")
    
    # Run detection on each attack
    results = {
        "model": model_name,
        "method": method,
        "num_samples": len(valid_texts),
        "avg_perplexity": np.mean(perplexities) if perplexities else float('nan'),
    }
    
    for attack_name, attack_cfg in ATTACKS.items():
        detections = []
        z_scores = []
        
        for text in tqdm(valid_texts, desc=f"  {attack_name}", leave=False):
            try:
                attacked_text = apply_attack(text, attack_cfg)
                result = wm.detect(attacked_text)
                detections.append(1 if result.is_watermarked else 0)
                z_scores.append(result.z_score)
            except Exception as e:
                continue
        
        if detections:
            det_rate = np.mean(detections) * 100
            avg_z = np.mean(z_scores)
            print(f"  {attack_name}: Detection {det_rate:.1f}%, Z-score {avg_z:.2f}")
            results[f"detection_{attack_name}"] = det_rate
            results[f"z_score_{attack_name}"] = avg_z
        else:
            results[f"detection_{attack_name}"] = 0.0
            results[f"z_score_{attack_name}"] = 0.0
    
    return results


def main(model_name, model_path, num_params=None, methods=None, num_samples=100):
    """Main experiment runner."""
    print("=" * 70)
    print(f"COMPREHENSIVE WATERMARKING EXPERIMENT")
    print(f"Model: {model_name}")
    print(f"Start: {datetime.now()}")
    print("=" * 70)
    
    device = get_device()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_path, num_params, device)
    
    # Load prompts
    prompts = load_prompts(num_samples)
    print(f"Loaded {len(prompts)} prompts")
    
    all_results = []
    
    # Run baseline if requested
    if methods is None or "Baseline" in methods:
        baseline_result = run_baseline(model, tokenizer, prompts, device, model_name, num_samples)
        if baseline_result:
            all_results.append(baseline_result)
    
    # Run watermarking methods (excluding Baseline)
    if methods is None:
        methods = ["GPW", "GPW-SP", "GPW-SP-LOW", "Unigram", "KGW"]
    else:
        methods = [m for m in methods if m != "Baseline"]
    
    for method in methods:
        result = run_watermark_experiment(model, tokenizer, prompts, device, model_name, method, num_samples)
        if result:
            all_results.append(result)
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = model_name.replace("/", "_").replace("-", "_")
        csv_path = RESULTS_DIR / f"comprehensive_{safe_model_name}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        print("\nSummary:")
        print(df.to_string())
    
    # Clear memory
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    print(f"\nExperiment completed at {datetime.now()}")
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--path", required=True, help="Model path")
    parser.add_argument("--params", type=int, default=None, help="Number of parameters")
    parser.add_argument("--methods", nargs="+", default=None, help="Methods to test")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    args = parser.parse_args()
    
    main(args.model, args.path, args.params, args.methods, args.samples)
