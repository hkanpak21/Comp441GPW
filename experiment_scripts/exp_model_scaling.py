#!/usr/bin/env python3
"""
Model Scaling Experiment: How does model capacity affect watermarking?

This experiment tests watermarking methods across different model sizes
using the Pythia model family (70M to 12B parameters).

Metrics: Detection Rate, Perplexity
Methods: GPW, GPW-SP, Unigram, KGW
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

# Set offline mode to avoid permission issues with shared cache
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Add project root to path
sys.path.insert(0, '/scratch/hkanpak21/Comp441GPW')

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pandas as pd

from watermarkers.gpw import GPWWatermark
from watermarkers.unigram import UnigramWatermark
from watermarkers.kgw import KGWWatermark

# ============================================================================
# CONFIGURATION
# ============================================================================

# Pythia model family - direct paths to shared cache (read-only)
SHARED_CACHE = "/datasets/NLP/huggingface/hub"

# Model name -> (display_name, num_params, snapshot_path)
PYTHIA_MODELS = [
    ("pythia-70m", 70_000_000, f"{SHARED_CACHE}/models--EleutherAI--pythia-70m/snapshots/2ab25ed47af79376eed2baaf8bbb7a192a0c73ff"),
    ("pythia-160m", 160_000_000, f"{SHARED_CACHE}/models--EleutherAI--pythia-160m/snapshots/26b94a336e5683a217752ab7ae4bf3cbe5661365"),
    ("pythia-410m", 410_000_000, f"{SHARED_CACHE}/models--EleutherAI--pythia-410m/snapshots/33fc2fdda3ea75631397cc28cec556f0ef401ae7"),
    ("pythia-1b", 1_000_000_000, f"{SHARED_CACHE}/models--EleutherAI--pythia-1b/snapshots/4c96d81536e92f85f8c2a45b5397057ce83a8636"),
    ("pythia-1.4b", 1_400_000_000, f"{SHARED_CACHE}/models--EleutherAI--pythia-1.4b/snapshots/8f0af839162ddb466006494a08733ee9cfa2d338"),
    ("pythia-2.8b", 2_800_000_000, f"{SHARED_CACHE}/models--EleutherAI--pythia-2.8b/snapshots/f20536c52a97faea73b8997cc789bd913853f14a"),
    ("pythia-6.9b", 6_900_000_000, f"{SHARED_CACHE}/models--EleutherAI--pythia-6.9b/snapshots/0b4fc522e9aeb35aeebbc44d05236cb68dd805cd"),
    ("pythia-12b", 12_000_000_000, f"{SHARED_CACHE}/models--EleutherAI--pythia-12b/snapshots/3fef353ace0849cccae3f4d5b45a4a962217be9d"),
]

# Watermarking methods to test
WATERMARK_METHODS = ["GPW", "GPW-SP", "Unigram", "KGW"]

# Experiment parameters
NUM_SAMPLES = 100  # Samples per model/method combination
MAX_NEW_TOKENS = 200
SEED = 42

# GPW optimal parameters (from tuning)
GPW_ALPHA = 3.0
GPW_OMEGA = 50.0
GPW_Z_THRESHOLD = 4.0

# Results directory
RESULTS_DIR = Path("/scratch/hkanpak21/Comp441GPW/results/scaling")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Found {n_gpus} GPU(s)")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return "cuda"
    return "cpu"


def load_model_and_tokenizer(model_name, num_params, model_path, device):
    """Load model and tokenizer from direct path."""
    print(f"\nLoading {model_name} from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with appropriate settings
    load_kwargs = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    
    # For larger models, use device_map for multi-GPU
    if num_params >= 6_000_000_000:  # 6B+ models
        load_kwargs["device_map"] = "auto"
        print(f"  Using device_map='auto' for {num_params/1e9:.1f}B model")
    else:
        load_kwargs["device_map"] = None
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    
    # Move to device if not using device_map
    if load_kwargs["device_map"] is None:
        model = model.to(device)
    
    model.eval()
    actual_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {actual_params/1e6:.1f}M parameters")
    
    return model, tokenizer


def load_prompts(num_samples):
    """Load prompts - use pre-saved or generate simple ones."""
    print(f"\nPreparing {num_samples} prompts...")
    
    # Use a set of diverse prompts for consistency
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
    
    # Repeat and shuffle for more samples
    prompts = []
    for i in range(num_samples):
        base = base_prompts[i % len(base_prompts)]
        # Add slight variation
        prompts.append(f"{base}")
    
    print(f"  Prepared {len(prompts)} prompts")
    return prompts


def create_watermarker(method, model, tokenizer, device):
    """Create a watermarker instance for the given method."""
    from watermarkers.gpw import GPWConfig, SRConfig
    
    # Get device from model for offloaded models
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = device
    
    if method == "GPW":
        gpw_cfg = GPWConfig(alpha=GPW_ALPHA, omega=GPW_OMEGA, salted=False)
        return GPWWatermark(
            model=model,
            tokenizer=tokenizer,
            gpw_cfg=gpw_cfg,
            device=str(model_device)
        )
    elif method == "GPW-SP":
        gpw_cfg = GPWConfig(alpha=GPW_ALPHA, omega=GPW_OMEGA, salted=True)
        return GPWWatermark(
            model=model,
            tokenizer=tokenizer,
            gpw_cfg=gpw_cfg,
            device=str(model_device)
        )
    elif method == "Unigram":
        return UnigramWatermark(
            model=model,
            tokenizer=tokenizer,
            device=str(model_device)
        )
    elif method == "KGW":
        return KGWWatermark(
            model=model,
            tokenizer=tokenizer,
            device=str(model_device)
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_perplexity(model, tokenizer, text, device):
    """Calculate perplexity of text."""
    try:
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encodings.input_ids.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        
        return torch.exp(loss).item()
    except Exception as e:
        print(f"    Perplexity calculation error: {e}")
        return float('nan')


def run_experiment_for_model(model_name, num_params, model_path, prompts, device):
    """Run all watermarking methods for a single model."""
    results = []
    
    # Load model
    try:
        model, tokenizer = load_model_and_tokenizer(model_name, num_params, model_path, device)
    except Exception as e:
        print(f"ERROR loading {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return results
    
    # Get device for perplexity calculation
    ppl_device = next(model.parameters()).device
    
    for method in WATERMARK_METHODS:
        print(f"\n  Testing {method}...")
        
        try:
            watermarker = create_watermarker(method, model, tokenizer, device)
        except Exception as e:
            print(f"    ERROR creating watermarker: {e}")
            continue
        
        detections = []
        perplexities = []
        z_scores = []
        
        for i, prompt in enumerate(prompts):
            if (i + 1) % 20 == 0:
                print(f"    Sample {i+1}/{len(prompts)}")
            
            try:
                # Generate watermarked text
                watermarked_text = watermarker.generate(
                    prompt,
                    max_new_tokens=MAX_NEW_TOKENS
                )
                
                # Detect watermark
                detection_result = watermarker.detect(watermarked_text)
                is_detected = detection_result.get("is_watermarked", False)
                z_score = detection_result.get("z_score", 0)
                
                detections.append(1 if is_detected else 0)
                z_scores.append(z_score)
                
                # Calculate perplexity
                ppl = calculate_perplexity(model, tokenizer, watermarked_text, ppl_device)
                if not np.isnan(ppl) and ppl < 1000:  # Filter outliers
                    perplexities.append(ppl)
                    
            except Exception as e:
                print(f"    Sample {i} error: {e}")
                continue
        
        # Aggregate results
        if detections:
            result = {
                "model": model_name,
                "num_params": num_params,
                "params_billions": num_params / 1e9,
                "method": method,
                "num_samples": len(detections),
                "detection_rate": np.mean(detections) * 100,
                "avg_z_score": np.mean(z_scores),
                "std_z_score": np.std(z_scores),
                "avg_perplexity": np.mean(perplexities) if perplexities else float('nan'),
                "std_perplexity": np.std(perplexities) if perplexities else float('nan'),
                "median_perplexity": np.median(perplexities) if perplexities else float('nan'),
            }
            results.append(result)
            
            print(f"    Detection: {result['detection_rate']:.1f}%")
            print(f"    Avg Z-score: {result['avg_z_score']:.2f}")
            print(f"    Avg Perplexity: {result['avg_perplexity']:.2f}")
    
    # Clear model from memory
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return results


def main():
    print("=" * 70)
    print("MODEL SCALING WATERMARK EXPERIMENT")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    print(f"Models: {len(PYTHIA_MODELS)} Pythia variants (70M - 12B)")
    print(f"Methods: {WATERMARK_METHODS}")
    print(f"Samples per combination: {NUM_SAMPLES}")
    
    device = get_device()
    
    # Load prompts once
    prompts = load_prompts(NUM_SAMPLES)
    
    all_results = []
    
    # Run experiments for each model
    for model_name, num_params, model_path in PYTHIA_MODELS:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name} ({num_params/1e9:.2f}B parameters)")
        print(f"{'='*70}")
        
        results = run_experiment_for_model(model_name, num_params, model_path, prompts, device)
        all_results.extend(results)
        
        # Save intermediate results
        if results:
            df = pd.DataFrame(all_results)
            df.to_csv(RESULTS_DIR / "scaling_results_partial.csv", index=False)
            print(f"\n  Saved intermediate results ({len(all_results)} rows)")
    
    # Save final results
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save CSV
    csv_path = RESULTS_DIR / f"scaling_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Save summary
    summary_path = RESULTS_DIR / f"scaling_summary_{timestamp}.txt"
    with open(summary_path, 'w') as f:
        f.write("MODEL SCALING WATERMARK EXPERIMENT SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Samples per model/method: {NUM_SAMPLES}\n\n")
        
        f.write("DETECTION RATES BY MODEL SIZE:\n")
        f.write("-" * 50 + "\n")
        pivot = df.pivot_table(
            index='params_billions', 
            columns='method', 
            values='detection_rate'
        )
        f.write(pivot.to_string())
        f.write("\n\n")
        
        f.write("PERPLEXITY BY MODEL SIZE:\n")
        f.write("-" * 50 + "\n")
        pivot_ppl = df.pivot_table(
            index='params_billions', 
            columns='method', 
            values='avg_perplexity'
        )
        f.write(pivot_ppl.to_string())
    
    print(f"Summary saved to: {summary_path}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print("\nDetection Rates (%):")
    print(df.pivot_table(index='params_billions', columns='method', values='detection_rate').round(1))
    print("\nAverage Perplexity:")
    print(df.pivot_table(index='params_billions', columns='method', values='avg_perplexity').round(1))
    
    print(f"\nExperiment completed at {datetime.now()}")


if __name__ == "__main__":
    main()
