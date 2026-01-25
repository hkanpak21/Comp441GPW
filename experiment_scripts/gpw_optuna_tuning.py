#!/usr/bin/env python3
"""
GPW Hyperparameter Optimization with Optuna

Goal: Find GPW parameters (alpha, omega) that maximize:
1. Detection rate on clean text (z-score > threshold)
2. Robustness against attacks (detection survives after attack)
3. Text quality (low perplexity increase)

Key insight from user: HIGHER omega should be better for robustness.
We will systematically test this hypothesis.

The watermark must be inherently robust - we cannot know which tokens
were attacked, so detection must work on the full attacked text.
"""

import os
import sys
import json
import math
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any

import torch
import optuna
from optuna.trial import Trial
from tqdm import tqdm

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from watermarkers.gpw import GPWWatermark, GPWConfig, SRConfig, create_gpw_variant
from attacks import SynonymAttack, WordSwapAttack, TypoAttack, CopyPasteAttack
from data_loaders import load_c4_sample


def compute_perplexity(model, tokenizer, text: str, device: str) -> float:
    """Compute perplexity of text under the model."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(input_ids=enc["input_ids"], labels=enc["input_ids"])
        return math.exp(outputs.loss.item())


def evaluate_config(
    model,
    tokenizer,
    alpha: float,
    omega: float,
    prompts: List[str],
    attacks: Dict[str, Any],
    device: str,
    z_threshold: float = 4.0,
    max_new_tokens: int = 150,
) -> Dict[str, float]:
    """
    Evaluate a GPW configuration on detection, robustness, and quality.
    
    Returns dict with:
    - clean_detection_rate: % of generated texts detected as watermarked
    - clean_z_mean: Mean z-score on clean watermarked text
    - attack_detection_rates: Dict[attack_name -> detection_rate]
    - attack_z_means: Dict[attack_name -> mean_z_score]
    - ppl_ratio: Ratio of watermarked perplexity to unwatermarked baseline
    """
    # Create watermarker with this config
    gpw_cfg = GPWConfig(
        alpha=alpha,
        omega=omega,
        salted=True,  # Always use GPW-SP
        ctx_mode="ngram",
        ngram=4,
        robust_detect=False,  # NO CHEATING - standard detection
        trim_fraction=0.0,
    )
    sr_cfg = SRConfig(enabled=False)
    
    watermarker = GPWWatermark(
        model=model,
        tokenizer=tokenizer,
        gpw_cfg=gpw_cfg,
        sr_cfg=sr_cfg,
        device=device,
    )
    watermarker.z_threshold = z_threshold
    
    results = {
        "clean_detections": 0,
        "clean_z_scores": [],
        "attack_detections": {name: 0 for name in attacks.keys()},
        "attack_z_scores": {name: [] for name in attacks.keys()},
        "ppl_watermarked": [],
        "ppl_baseline": [],
    }
    
    for prompt in prompts:
        # Generate watermarked text
        try:
            watermarked_text = watermarker.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_k=50,
                top_p=0.95,
            )
        except Exception as e:
            print(f"Generation error: {e}")
            continue
        
        # Get only the generated part (remove prompt)
        generated_only = watermarked_text[len(prompt):].strip()
        if len(generated_only.split()) < 20:
            continue  # Skip too short generations
        
        # Detect on clean text
        det_clean = watermarker.detect(watermarked_text)
        results["clean_z_scores"].append(det_clean.z_score)
        if det_clean.is_watermarked:
            results["clean_detections"] += 1
        
        # Compute perplexity
        try:
            ppl_wm = compute_perplexity(model, tokenizer, generated_only, device)
            results["ppl_watermarked"].append(ppl_wm)
        except:
            pass
        
        # Test each attack
        for attack_name, attack in attacks.items():
            try:
                attacked_text = attack.attack(watermarked_text)
                det_attack = watermarker.detect(attacked_text)
                results["attack_z_scores"][attack_name].append(det_attack.z_score)
                if det_attack.is_watermarked:
                    results["attack_detections"][attack_name] += 1
            except Exception as e:
                print(f"Attack {attack_name} error: {e}")
    
    n_samples = len(results["clean_z_scores"])
    if n_samples == 0:
        return None
    
    # Compute metrics
    metrics = {
        "n_samples": n_samples,
        "alpha": alpha,
        "omega": omega,
        "clean_detection_rate": results["clean_detections"] / n_samples,
        "clean_z_mean": sum(results["clean_z_scores"]) / n_samples,
        "clean_z_min": min(results["clean_z_scores"]),
    }
    
    # Attack metrics
    for attack_name in attacks.keys():
        n_attack = len(results["attack_z_scores"][attack_name])
        if n_attack > 0:
            metrics[f"{attack_name}_detection_rate"] = results["attack_detections"][attack_name] / n_attack
            metrics[f"{attack_name}_z_mean"] = sum(results["attack_z_scores"][attack_name]) / n_attack
        else:
            metrics[f"{attack_name}_detection_rate"] = 0.0
            metrics[f"{attack_name}_z_mean"] = 0.0
    
    # Quality metric
    if results["ppl_watermarked"]:
        metrics["ppl_mean"] = sum(results["ppl_watermarked"]) / len(results["ppl_watermarked"])
    else:
        metrics["ppl_mean"] = float('inf')
    
    # Composite robustness score (average detection rate across attacks)
    attack_rates = [metrics[f"{name}_detection_rate"] for name in attacks.keys()]
    metrics["attack_robustness_mean"] = sum(attack_rates) / len(attack_rates) if attack_rates else 0.0
    
    return metrics


def objective(trial: Trial, model, tokenizer, prompts, attacks, device, z_threshold) -> float:
    """
    Optuna objective function.
    
    We optimize for a weighted combination of:
    1. Clean detection rate (must be high)
    2. Attack robustness (average detection rate under attacks)
    3. Text quality (perplexity should not be too high)
    
    User insight: HIGHER omega should improve robustness.
    Test range: omega from 5 to 50 (high values)
    """
    # Hyperparameter search space
    # alpha: controls strength of logit bias (higher = more detectable, may hurt quality)
    # omega: cosine frequency (USER SAYS HIGHER IS BETTER - test high values)
    alpha = trial.suggest_float("alpha", 1.0, 10.0, log=True)
    omega = trial.suggest_float("omega", 5.0, 100.0, log=True)  # HIGH VALUES as user requested
    
    metrics = evaluate_config(
        model=model,
        tokenizer=tokenizer,
        alpha=alpha,
        omega=omega,
        prompts=prompts,
        attacks=attacks,
        device=device,
        z_threshold=z_threshold,
    )
    
    if metrics is None:
        return float('-inf')  # Failed trial
    
    # Log intermediate values
    trial.set_user_attr("clean_detection_rate", metrics["clean_detection_rate"])
    trial.set_user_attr("clean_z_mean", metrics["clean_z_mean"])
    trial.set_user_attr("attack_robustness_mean", metrics["attack_robustness_mean"])
    trial.set_user_attr("ppl_mean", metrics["ppl_mean"])
    
    for attack_name in attacks.keys():
        trial.set_user_attr(f"{attack_name}_detection_rate", metrics[f"{attack_name}_detection_rate"])
    
    # Objective: Maximize robustness while maintaining clean detection
    # Penalty if clean detection drops below 80%
    clean_penalty = 0.0 if metrics["clean_detection_rate"] >= 0.8 else -10 * (0.8 - metrics["clean_detection_rate"])
    
    # Penalty for very high perplexity (quality degradation)
    ppl_penalty = 0.0
    if metrics["ppl_mean"] > 50:
        ppl_penalty = -0.1 * (metrics["ppl_mean"] - 50)
    
    # Main objective: attack robustness
    score = metrics["attack_robustness_mean"] + clean_penalty + ppl_penalty
    
    return score


def run_grid_search(
    model,
    tokenizer,
    prompts: List[str],
    attacks: Dict[str, Any],
    device: str,
    z_threshold: float = 4.0,
    output_dir: str = "results/gpw_tuning",
) -> List[Dict]:
    """
    Run a systematic grid search over alpha and omega values.
    Focus on HIGH omega values as per user's insight.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Grid: alpha x omega
    # User said HIGHER omega is better - test this hypothesis
    alphas = [1.0, 2.0, 3.0, 5.0, 8.0]
    omegas = [2.0, 5.0, 10.0, 20.0, 50.0, 100.0]  # Focus on HIGH values
    
    all_results = []
    
    print("\n" + "="*80)
    print("GPW GRID SEARCH - Testing HIGH omega values (user hypothesis)")
    print("="*80)
    
    for alpha in alphas:
        for omega in omegas:
            print(f"\n--- Testing alpha={alpha}, omega={omega} ---")
            
            metrics = evaluate_config(
                model=model,
                tokenizer=tokenizer,
                alpha=alpha,
                omega=omega,
                prompts=prompts,
                attacks=attacks,
                device=device,
                z_threshold=z_threshold,
            )
            
            if metrics:
                all_results.append(metrics)
                
                # Print summary
                print(f"  Clean detection: {metrics['clean_detection_rate']*100:.1f}% (z={metrics['clean_z_mean']:.2f})")
                print(f"  Attack robustness: {metrics['attack_robustness_mean']*100:.1f}%")
                for attack_name in attacks.keys():
                    print(f"    {attack_name}: {metrics[f'{attack_name}_detection_rate']*100:.1f}%")
    
    # Sort by attack robustness
    all_results.sort(key=lambda x: x["attack_robustness_mean"], reverse=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"grid_search_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("TOP 5 CONFIGURATIONS BY ATTACK ROBUSTNESS:")
    print("="*80)
    for i, r in enumerate(all_results[:5]):
        print(f"\n{i+1}. alpha={r['alpha']}, omega={r['omega']}")
        print(f"   Clean: {r['clean_detection_rate']*100:.1f}% | Robustness: {r['attack_robustness_mean']*100:.1f}%")
    
    return all_results


def run_optuna_optimization(
    model,
    tokenizer,
    prompts: List[str],
    attacks: Dict[str, Any],
    device: str,
    z_threshold: float = 4.0,
    n_trials: int = 50,
    output_dir: str = "results/gpw_tuning",
) -> optuna.Study:
    """Run Optuna optimization to find best GPW parameters."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name="gpw_robustness_optimization",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, model, tokenizer, prompts, attacks, device, z_threshold),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    
    # Print results
    print("\n" + "="*80)
    print("OPTUNA OPTIMIZATION RESULTS")
    print("="*80)
    
    print(f"\nBest trial:")
    print(f"  Value (robustness score): {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")
    print(f"  User attrs: {study.best_trial.user_attrs}")
    
    # Save study
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save trial data
    trials_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trials_data.append({
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            })
    
    trials_file = os.path.join(output_dir, f"optuna_trials_{timestamp}.json")
    with open(trials_file, "w") as f:
        json.dump(trials_data, f, indent=2)
    
    print(f"\nResults saved to {trials_file}")
    
    return study


def main():
    parser = argparse.ArgumentParser(description="GPW Hyperparameter Optimization")
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--n_samples", type=int, default=30, help="Number of prompts to test")
    parser.add_argument("--mode", type=str, default="grid", choices=["grid", "optuna", "both"])
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--output_dir", type=str, default="results/gpw_tuning")
    parser.add_argument("--z_threshold", type=float, default=4.0)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load prompts
    print(f"\nLoading {args.n_samples} prompts from C4...")
    prompts = load_c4_sample(n_samples=args.n_samples, max_length=50)
    print(f"Loaded {len(prompts)} prompts")
    
    # Setup attacks (realistic attack strengths)
    print("\nSetting up attacks...")
    attacks = {
        "synonym_30": SynonymAttack(prob=0.30),
        "swap_20": WordSwapAttack(prob=0.20),
        "typo_10": TypoAttack(prob=0.10),
        "copypaste_50": CopyPasteAttack(mix_ratio=0.50),
    }
    
    # Run optimization
    if args.mode in ["grid", "both"]:
        print("\n" + "="*80)
        print("RUNNING GRID SEARCH")
        print("="*80)
        grid_results = run_grid_search(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            attacks=attacks,
            device=device,
            z_threshold=args.z_threshold,
            output_dir=args.output_dir,
        )
    
    if args.mode in ["optuna", "both"]:
        print("\n" + "="*80)
        print("RUNNING OPTUNA OPTIMIZATION")
        print("="*80)
        study = run_optuna_optimization(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            attacks=attacks,
            device=device,
            z_threshold=args.z_threshold,
            n_trials=args.n_trials,
            output_dir=args.output_dir,
        )
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
