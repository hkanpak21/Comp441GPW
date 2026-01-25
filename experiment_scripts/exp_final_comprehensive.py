#!/usr/bin/env python3
"""
Final Comprehensive Watermarking Experiments

Compares ALL watermarking methods under ALL attack scenarios:
- Watermarkers: Unigram, KGW, GPW, GPW-SP, GPW-SP+SR, SEMSTAMP
- Attacks: Synonym, Swap, Typo, CopyPaste, Paraphrase
- Models: OPT-1.3B (primary), GPT-2, Qwen-7B
- Metrics: Detection rate, z-score, false positive rate, perplexity

Usage:
    python exp_final_comprehensive.py --model opt-1.3b --n_samples 200
"""

import os
import sys
import csv
import json
import hashlib
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

import torch
from tqdm import tqdm

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Watermarkers
from watermarkers import UnigramWatermark, KGWWatermark
from watermarkers.gpw import GPWWatermark, GPWConfig, SRConfig
from watermarkers.semstamp import SEMSTAMPWatermark

# Global embedder for SEMSTAMP (loaded once)
_SEMSTAMP_EMBEDDER = None

# Attacks
from attacks import SynonymAttack, SwapAttack, TypoAttack, CopyPasteAttack
from attacks.paraphrase import PegasusAttack

# Data loaders
from data_loaders import load_c4

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_CONFIGS = {
    "gpt2": {
        "name": "gpt2",
        "dtype": torch.float32,
        "max_tokens": 200,
    },
    "opt-1.3b": {
        "name": "facebook/opt-1.3b",
        "dtype": torch.float16,
        "max_tokens": 200,
    },
    "qwen-7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "dtype": torch.float16,
        "max_tokens": 200,
    },
}

# Watermarker configurations - using OPTIMAL parameters from tuning
WATERMARKER_CONFIGS = {
    "unigram": {
        "type": "unigram",
        "gamma": 0.5,
        "delta": 2.0,
        "z_threshold": 4.0,
    },
    "kgw": {
        "type": "kgw",
        "gamma": 0.5,
        "delta": 2.0,
        "z_threshold": 4.0,
        "seeding_scheme": "simple_1",
    },
    # GPW baseline (no salting)
    "gpw": {
        "type": "gpw",
        "variant": "GPW",
        "alpha": 3.0,
        "omega": 50.0,  # OPTIMAL from tuning
        "z_threshold": 4.0,
    },
    # GPW-SP (salted phase) - OPTIMAL config
    "gpw_sp": {
        "type": "gpw",
        "variant": "GPW-SP",
        "alpha": 3.0,
        "omega": 50.0,  # OPTIMAL from tuning
        "z_threshold": 4.0,
    },
    # GPW-SP with low omega (for comparison)
    "gpw_sp_low": {
        "type": "gpw",
        "variant": "GPW-SP",
        "alpha": 3.0,
        "omega": 2.0,  # Original low omega
        "z_threshold": 4.0,
    },
    # GPW-SP+SR (semantic representation coupling)
    "gpw_sp_sr": {
        "type": "gpw",
        "variant": "GPW-SP+SR",
        "alpha": 3.0,
        "omega": 50.0,
        "z_threshold": 4.0,
        "sr_lambda": 0.1,
        "sr_rank": 16,
    },
    # SEMSTAMP
    "semstamp": {
        "type": "semstamp",
        "n_lsh_bits": 3,
        "lsh_margin": 0.0,
        "z_threshold": 4.0,
        "max_rejections": 2,  # User specified
    },
}

# Attack configurations
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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_device():
    """Get compute device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def compute_perplexity(text: str, model, tokenizer, device: str) -> float:
    """Compute perplexity of text under model."""
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            return float(torch.exp(outputs.loss).cpu())
    except Exception as e:
        return -1.0


def create_watermarker(config: Dict, model, tokenizer, device: str):
    """Factory function to create watermarker from config."""
    wm_type = config["type"]
    
    if wm_type == "unigram":
        return UnigramWatermark(
            model=model,
            tokenizer=tokenizer,
            gamma=config["gamma"],
            delta=config["delta"],
            z_threshold=config["z_threshold"],
            device=device,
        )
    
    elif wm_type == "kgw":
        return KGWWatermark(
            model=model,
            tokenizer=tokenizer,
            gamma=config["gamma"],
            delta=config["delta"],
            z_threshold=config["z_threshold"],
            seeding_scheme=config["seeding_scheme"],
            device=device,
        )
    
    elif wm_type == "gpw":
        variant = config["variant"]
        gpw_cfg = GPWConfig(
            alpha=config["alpha"],
            omega=config["omega"],
            salted=(variant != "GPW"),
            ctx_mode="ngram",
            ngram=4,
        )
        sr_enabled = "SR" in variant
        sr_cfg = SRConfig(
            enabled=sr_enabled,
            lambda_couple=config.get("sr_lambda", 0.1) if sr_enabled else 0.09,
            rank=config.get("sr_rank", 16) if sr_enabled else 32,
        )
        wm = GPWWatermark(
            model=model,
            tokenizer=tokenizer,
            gpw_cfg=gpw_cfg,
            sr_cfg=sr_cfg,
            hash_key=hashlib.sha256(b"gpw-final-exp").digest(),
            device=device,
        )
        wm.z_threshold = config["z_threshold"]
        return wm
    
    elif wm_type == "semstamp":
        # Load embedder on first use (cached globally)
        global _SEMSTAMP_EMBEDDER
        if _SEMSTAMP_EMBEDDER is None:
            print("Loading SentenceTransformer for SEMSTAMP...")
            _SEMSTAMP_EMBEDDER = SentenceTransformer("all-mpnet-base-v2")
            _SEMSTAMP_EMBEDDER.to(device)
            print("  Embedder loaded!")
        
        return SEMSTAMPWatermark(
            model=model,
            tokenizer=tokenizer,
            embedder=_SEMSTAMP_EMBEDDER,
            lsh_dim=config["n_lsh_bits"],
            margin=config["lsh_margin"],
            z_threshold=config["z_threshold"],
            max_rejections=config["max_rejections"],
            device=device,
        )
    
    else:
        raise ValueError(f"Unknown watermarker type: {wm_type}")


def create_attack(config: Optional[Dict]):
    """Factory function to create attack from config."""
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
        return CopyPasteAttack(
            n_segments=config["n_segments"],
            watermark_ratio=config["watermark_ratio"],
        )
    elif attack_type == "paraphrase":
        return PegasusAttack()
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")


def apply_attack(attack, text: str, human_text: Optional[str] = None) -> str:
    """Apply attack and return attacked text."""
    if attack is None:
        return text
    
    attack_name = attack.__class__.__name__
    
    if "CopyPaste" in attack_name:
        if human_text is None:
            return text
        result = attack.attack(text, human_text=human_text)
    else:
        result = attack.attack(text)
    
    # Handle AttackResult object
    if hasattr(result, 'attacked_text'):
        return result.attacked_text
    return str(result) if result else text


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_watermarker_experiment(
    wm_name: str,
    wm_config: Dict,
    model,
    tokenizer,
    prompts: List[str],
    human_texts: List[str],
    device: str,
    max_tokens: int,
    attacks_to_run: List[str] = None,
) -> List[Dict]:
    """Run experiment for a single watermarker across all attacks."""
    
    print(f"\n{'='*70}")
    print(f"WATERMARKER: {wm_name.upper()}")
    print(f"Config: {wm_config}")
    print("="*70)
    
    results = []
    
    # Create watermarker
    try:
        watermarker = create_watermarker(wm_config, model, tokenizer, device)
    except Exception as e:
        print(f"  ERROR creating watermarker: {e}")
        return results
    
    # Generate watermarked texts
    print(f"\nGenerating {len(prompts)} watermarked texts...")
    watermarked_texts = []
    perplexities = []
    
    for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
        try:
            wm_text = watermarker.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=1.0,
                top_k=50,
                top_p=0.95,
            )
            watermarked_texts.append(wm_text)
            
            # Compute perplexity on generated part only
            generated_part = wm_text[len(prompt):].strip()
            if len(generated_part) > 20:
                ppl = compute_perplexity(generated_part, model, tokenizer, device)
                perplexities.append(ppl)
            else:
                perplexities.append(-1.0)
                
        except Exception as e:
            watermarked_texts.append(None)
            perplexities.append(-1.0)
    
    valid_count = sum(1 for t in watermarked_texts if t is not None)
    print(f"  Successfully generated: {valid_count}/{len(prompts)}")
    
    valid_ppls = [p for p in perplexities if p > 0]
    if valid_ppls:
        print(f"  Mean perplexity: {sum(valid_ppls)/len(valid_ppls):.2f}")
    
    # Test each attack
    attacks_to_test = attacks_to_run or list(ATTACK_CONFIGS.keys())
    
    for attack_name in attacks_to_test:
        attack_config = ATTACK_CONFIGS.get(attack_name)
        
        print(f"\n  Attack: {attack_name}")
        
        try:
            attack = create_attack(attack_config)
        except Exception as e:
            print(f"    ERROR creating attack: {e}")
            continue
        
        detected_count = 0
        z_scores = []
        
        for i, wm_text in enumerate(watermarked_texts):
            if wm_text is None:
                continue
            
            try:
                # Apply attack
                attacked_text = apply_attack(attack, wm_text, human_texts[i] if human_texts else None)
                
                if not attacked_text:
                    continue
                
                # Detect
                det_result = watermarker.detect(attacked_text)
                z_scores.append(det_result.z_score)
                
                if det_result.is_watermarked:
                    detected_count += 1
                
                # Store result
                results.append({
                    "watermarker": wm_name,
                    "attack": attack_name,
                    "sample_idx": i,
                    "z_score": det_result.z_score,
                    "p_value": det_result.p_value,
                    "is_detected": 1 if det_result.is_watermarked else 0,
                    "perplexity": perplexities[i] if attack_name == "clean" else -1,
                    "prompt_len": len(prompts[i].split()),
                    "text_len": len(attacked_text.split()),
                })
                
            except Exception as e:
                continue
        
        n_valid = len(z_scores)
        if n_valid > 0:
            det_rate = detected_count / n_valid
            mean_z = sum(z_scores) / n_valid
            print(f"    Detection: {det_rate*100:.1f}% ({detected_count}/{n_valid}), Mean Z: {mean_z:.2f}")
        else:
            print(f"    No valid samples")
    
    return results


def run_baseline_experiment(
    model,
    tokenizer,
    prompts: List[str],
    human_texts: List[str],
    device: str,
    max_tokens: int,
) -> List[Dict]:
    """Test false positive rates on human text and unwatermarked AI text."""
    
    print(f"\n{'='*70}")
    print("BASELINES: Human Text & Unwatermarked AI")
    print("="*70)
    
    results = []
    
    # Test each watermarker's FPR on human text
    for wm_name, wm_config in WATERMARKER_CONFIGS.items():
        print(f"\n  Testing FPR for: {wm_name}")
        
        try:
            watermarker = create_watermarker(wm_config, model, tokenizer, device)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
        
        # Test on human texts
        fp_count = 0
        z_scores = []
        
        for i, text in enumerate(human_texts[:100]):  # Use 100 for FPR
            try:
                det = watermarker.detect(text)
                z_scores.append(det.z_score)
                if det.is_watermarked:
                    fp_count += 1
                    
                results.append({
                    "watermarker": wm_name,
                    "text_type": "human",
                    "sample_idx": i,
                    "z_score": det.z_score,
                    "is_detected": 1 if det.is_watermarked else 0,
                })
            except:
                continue
        
        n_valid = len(z_scores)
        if n_valid > 0:
            fpr = fp_count / n_valid
            mean_z = sum(z_scores) / n_valid
            print(f"    Human FPR: {fpr*100:.1f}% ({fp_count}/{n_valid}), Mean Z: {mean_z:.2f}")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Watermarking Experiments")
    parser.add_argument("--model", type=str, default="opt-1.3b", 
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--watermarkers", type=str, nargs="+", default=None,
                        help="Specific watermarkers to test (default: all)")
    parser.add_argument("--attacks", type=str, nargs="+", default=None,
                        help="Specific attacks to test (default: all)")
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--output_prefix", type=str, default="final")
    args = parser.parse_args()
    
    device = get_device()
    model_config = MODEL_CONFIGS[args.model]
    
    print("="*70)
    print("COMPREHENSIVE WATERMARKING EXPERIMENTS")
    print("="*70)
    print(f"Model: {model_config['name']}")
    print(f"Samples: {args.n_samples}")
    print(f"Device: {device}")
    print(f"Watermarkers: {args.watermarkers or list(WATERMARKER_CONFIGS.keys())}")
    print(f"Attacks: {args.attacks or list(ATTACK_CONFIGS.keys())}")
    
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
    print("\nLoading C4 dataset...")
    c4_data = load_c4(num_samples=args.n_samples)
    prompts = [d['prompt'] for d in c4_data]
    human_texts = [d['text'] for d in c4_data]
    print(f"Loaded {len(prompts)} samples")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    
    # Determine which watermarkers to test
    watermarkers_to_test = args.watermarkers or list(WATERMARKER_CONFIGS.keys())
    
    # Run experiments for each watermarker
    for wm_name in watermarkers_to_test:
        if wm_name not in WATERMARKER_CONFIGS:
            print(f"Unknown watermarker: {wm_name}, skipping")
            continue
        
        wm_config = WATERMARKER_CONFIGS[wm_name]
        
        results = run_watermarker_experiment(
            wm_name=wm_name,
            wm_config=wm_config,
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            human_texts=human_texts,
            device=device,
            max_tokens=model_config["max_tokens"],
            attacks_to_run=args.attacks,
        )
        
        all_results.extend(results)
        
        # Save intermediate results
        intermediate_file = os.path.join(
            RESULTS_DIR, 
            f"{args.output_prefix}_{wm_name}_{args.model}_{timestamp}.csv"
        )
        if results:
            with open(intermediate_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"  Saved: {intermediate_file}")
    
    # Run baseline experiments
    if not args.skip_baselines:
        baseline_results = run_baseline_experiment(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            human_texts=human_texts,
            device=device,
            max_tokens=model_config["max_tokens"],
        )
        all_results.extend(baseline_results)
    
    # Save all results
    all_file = os.path.join(RESULTS_DIR, f"{args.output_prefix}_all_{args.model}_{timestamp}.csv")
    if all_results:
        with open(all_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nAll results saved to: {all_file}")
    
    # Generate summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    summary = {}
    for wm_name in watermarkers_to_test:
        if wm_name not in WATERMARKER_CONFIGS:
            continue
        summary[wm_name] = {}
        
        for attack_name in (args.attacks or list(ATTACK_CONFIGS.keys())):
            wm_attack_results = [r for r in all_results 
                                if r.get("watermarker") == wm_name 
                                and r.get("attack") == attack_name]
            
            if wm_attack_results:
                det_rate = sum(r["is_detected"] for r in wm_attack_results) / len(wm_attack_results)
                mean_z = sum(r["z_score"] for r in wm_attack_results) / len(wm_attack_results)
                summary[wm_name][attack_name] = {
                    "detection_rate": det_rate,
                    "mean_z": mean_z,
                    "n_samples": len(wm_attack_results),
                }
                print(f"{wm_name:15} | {attack_name:15} | Det: {det_rate*100:5.1f}% | Z: {mean_z:5.2f}")
    
    # Save summary
    summary_file = os.path.join(RESULTS_DIR, f"{args.output_prefix}_summary_{args.model}_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
