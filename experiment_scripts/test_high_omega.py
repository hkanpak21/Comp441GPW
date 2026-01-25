#!/usr/bin/env python3
"""
Quick test: HIGH omega values for GPW robustness.

User hypothesis: Higher omega = more robust against attacks.
Test this on a small sample set to verify before full experiments.
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermarkers.gpw import GPWWatermark, GPWConfig, SRConfig
from attacks import SynonymAttack, SwapAttack, TypoAttack, CopyPasteAttack
from data_loaders import load_c4


def test_omega_robustness():
    """Test different omega values for attack robustness."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading model: facebook/opt-1.3b")
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load prompts
    print("\nLoading 30 samples from C4...")
    c4_data = load_c4(num_samples=30)
    prompts = [d['prompt'] for d in c4_data]
    human_texts = [d['text'] for d in c4_data]  # For copypaste
    
    # Setup attacks
    attacks = {
        "clean": None,
        "synonym_30": SynonymAttack(edit_rate=0.30),
        "swap_20": SwapAttack(edit_rate=0.20),
        "typo_10": TypoAttack(edit_rate=0.10),
        "copypaste_50": CopyPasteAttack(n_segments=3, watermark_ratio=0.50),
    }
    
    # Test different omega values (LOW to HIGH)
    omega_values = [2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    alpha = 3.0  # Keep alpha fixed
    z_threshold = 4.0
    
    print("\n" + "="*80)
    print("TESTING OMEGA VALUES: Does higher omega = more robustness?")
    print("="*80)
    print(f"Alpha (fixed): {alpha}")
    print(f"Omega values: {omega_values}")
    print(f"Z-threshold: {z_threshold}")
    print(f"Samples: {len(prompts)}")
    
    results = {}
    
    for omega in omega_values:
        print(f"\n{'='*60}")
        print(f"Testing omega={omega}")
        print("="*60)
        
        # Create watermarker
        gpw_cfg = GPWConfig(
            alpha=alpha,
            omega=omega,
            salted=True,  # GPW-SP
            ctx_mode="ngram",
            ngram=4,
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
        
        # Generate watermarked texts
        watermarked_texts = []
        print("Generating watermarked texts...")
        for i, prompt in enumerate(prompts):
            try:
                wm_text = watermarker.generate(
                    prompt,
                    max_new_tokens=150,
                    temperature=1.0,
                    top_k=50,
                    top_p=0.95,
                )
                watermarked_texts.append(wm_text)
            except Exception as e:
                watermarked_texts.append(None)
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i+1}/{len(prompts)}")
        
        # Test detection under each attack
        attack_results = {}
        for attack_name, attack in attacks.items():
            detected = 0
            z_scores = []
            
            for i, wm_text in enumerate(watermarked_texts):
                if wm_text is None:
                    continue
                
                try:
                    # Apply attack
                    if attack is None:
                        test_text = wm_text
                    elif attack_name == "copypaste_50":
                        result = attack.attack(wm_text, human_text=human_texts[i % len(human_texts)])
                        test_text = result.attacked_text if hasattr(result, 'attacked_text') else str(result)
                    else:
                        result = attack.attack(wm_text)
                        test_text = result.attacked_text if hasattr(result, 'attacked_text') else str(result)
                    
                    if not test_text:
                        continue
                    
                    # Detect
                    det = watermarker.detect(test_text)
                    z_scores.append(det.z_score)
                    if det.is_watermarked:
                        detected += 1
                except Exception as e:
                    continue
            
            n = len(z_scores)
            if n > 0:
                det_rate = detected / n
                mean_z = sum(z_scores) / n
            else:
                det_rate = 0.0
                mean_z = 0.0
            
            attack_results[attack_name] = {
                "detection_rate": det_rate,
                "mean_z": mean_z,
                "n": n,
            }
            print(f"  {attack_name}: {det_rate*100:.1f}% (z={mean_z:.2f})")
        
        results[omega] = attack_results
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: Detection Rates by Omega Value")
    print("="*80)
    print(f"{'Omega':<10}", end="")
    for attack_name in attacks.keys():
        print(f" | {attack_name:<12}", end="")
    print(" | Avg Attack")
    print("-"*90)
    
    for omega in omega_values:
        print(f"{omega:<10.1f}", end="")
        attack_rates = []
        for attack_name in attacks.keys():
            rate = results[omega][attack_name]["detection_rate"]
            print(f" | {rate*100:>10.1f}%", end="")
            if attack_name != "clean":
                attack_rates.append(rate)
        avg_attack = sum(attack_rates) / len(attack_rates) if attack_rates else 0
        print(f" | {avg_attack*100:>6.1f}%")
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS: Change in robustness relative to omega=2.0 baseline")
    print("="*80)
    baseline_omega = 2.0
    
    for omega in omega_values:
        if omega == baseline_omega:
            continue
        
        clean_diff = results[omega]["clean"]["detection_rate"] - results[baseline_omega]["clean"]["detection_rate"]
        
        attack_diffs = []
        for attack_name in ["synonym_30", "swap_20", "typo_10", "copypaste_50"]:
            diff = results[omega][attack_name]["detection_rate"] - results[baseline_omega][attack_name]["detection_rate"]
            attack_diffs.append(diff)
        
        avg_attack_improvement = sum(attack_diffs) / len(attack_diffs)
        
        print(f"omega={omega}: Clean change: {clean_diff*100:+.1f}pp, Attack robustness change: {avg_attack_improvement*100:+.1f}pp")
    
    # Find best omega
    best_omega = None
    best_score = float('-inf')
    for omega in omega_values:
        # Score = attack robustness (weighted) + clean detection
        clean = results[omega]["clean"]["detection_rate"]
        attacks_avg = sum(results[omega][a]["detection_rate"] for a in ["synonym_30", "swap_20", "typo_10", "copypaste_50"]) / 4
        score = 0.3 * clean + 0.7 * attacks_avg  # Weight robustness higher
        if score > best_score:
            best_score = score
            best_omega = omega
    
    print(f"\n*** BEST OMEGA for robustness (70% attack, 30% clean): omega={best_omega} ***")
    
    return results


if __name__ == "__main__":
    test_omega_robustness()
