#!/usr/bin/env python3
"""
GPW Ablation Study - Test different hyperparameter configurations
"""

import argparse
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from watermarkers.gpw import GPWWatermark, GPWConfig, SRConfig
from attacks import SynonymAttack, SwapAttack, TypoAttack, CopyPasteAttack


def load_prompts(n_samples):
    """Load prompts from C4 dataset."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("c4", "realnewslike", split="validation", streaming=True)
        prompts = []
        for item in dataset:
            text = item.get('text', '')
            if 100 <= len(text) <= 1000:
                prompts.append(text[:50])
                if len(prompts) >= n_samples:
                    break
        return prompts, None
    except:
        base_prompts = [
            "The future of artificial intelligence will",
            "Climate change is affecting our planet because",
            "The history of computing began when",
            "In modern society, technology has",
            "Scientists have discovered that",
        ]
        return (base_prompts * (n_samples // len(base_prompts) + 1))[:n_samples], None


def apply_attack(attack, text, human_text=None):
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

    if hasattr(result, 'attacked_text'):
        return result.attacked_text
    return str(result) if result else text


def run_ablation(args):
    print(f"Loading model: {args.model}")

    if args.model == "gpt2":
        model_name = "gpt2"
    elif args.model == "opt-1.3b":
        model_name = "facebook/opt-1.3b"
    else:
        model_name = args.model

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.n_samples} prompts...")
    prompts, human_texts = load_prompts(args.n_samples)

    # Ablation configurations
    configs = [
        # Omega ablation (cosine frequency - affects robustness)
        {"name": "omega_1", "alpha": 3.0, "omega": 1.0, "salted": False, "sr": False},
        {"name": "omega_5", "alpha": 3.0, "omega": 5.0, "salted": False, "sr": False},
        {"name": "omega_10", "alpha": 3.0, "omega": 10.0, "salted": False, "sr": False},
        {"name": "omega_25", "alpha": 3.0, "omega": 25.0, "salted": False, "sr": False},
        {"name": "omega_50", "alpha": 3.0, "omega": 50.0, "salted": False, "sr": False},
        {"name": "omega_100", "alpha": 3.0, "omega": 100.0, "salted": False, "sr": False},

        # Alpha ablation (logit bias strength)
        {"name": "alpha_1", "alpha": 1.0, "omega": 50.0, "salted": False, "sr": False},
        {"name": "alpha_2", "alpha": 2.0, "omega": 50.0, "salted": False, "sr": False},
        {"name": "alpha_3", "alpha": 3.0, "omega": 50.0, "salted": False, "sr": False},
        {"name": "alpha_5", "alpha": 5.0, "omega": 50.0, "salted": False, "sr": False},
        {"name": "alpha_10", "alpha": 10.0, "omega": 50.0, "salted": False, "sr": False},

        # Mode ablation (GPW vs GPW-SP vs GPW-SP+SR)
        {"name": "GPW", "alpha": 3.0, "omega": 50.0, "salted": False, "sr": False},
        {"name": "GPW-SP", "alpha": 3.0, "omega": 50.0, "salted": True, "sr": False},
        {"name": "GPW-SP+SR", "alpha": 3.0, "omega": 50.0, "salted": True, "sr": True},

        # Low omega variants (for text quality)
        {"name": "GPW_low", "alpha": 2.0, "omega": 2.0, "salted": False, "sr": False},
        {"name": "GPW-SP_low", "alpha": 2.0, "omega": 2.0, "salted": True, "sr": False},
    ]

    # Attack instances
    attacks = {
        "clean": None,
        "synonym_30": SynonymAttack(edit_rate=0.3),
        "swap_20": SwapAttack(edit_rate=0.2),
        "typo_10": TypoAttack(edit_rate=0.1),
    }

    all_results = []

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Config: {config['name']}")
        print(f"  alpha={config['alpha']}, omega={config['omega']}, salted={config['salted']}, sr={config['sr']}")
        print(f"{'='*60}")

        # Create GPW watermarker with custom config
        gpw_cfg = GPWConfig(
            alpha=config['alpha'],
            omega=config['omega'],
            salted=config['salted'],
            ctx_mode="ngram",
            ngram=4
        )
        sr_cfg = SRConfig(enabled=config['sr'])

        watermarker = GPWWatermark(
            model=model,
            tokenizer=tokenizer,
            gpw_cfg=gpw_cfg,
            sr_cfg=sr_cfg,
            device=device
        )

        # Generate texts
        print("Generating texts...")
        generated_texts = []
        for prompt in tqdm(prompts, desc="Generating"):
            try:
                text = watermarker.generate(prompt, max_new_tokens=150)
                generated_texts.append(text)
            except Exception as e:
                print(f"  Error: {e}")
                generated_texts.append(prompt)

        # Test each attack
        for attack_name, attack_instance in attacks.items():
            print(f"  Testing attack: {attack_name}")

            detected = 0
            z_scores = []

            for text in generated_texts:
                try:
                    attacked_text = apply_attack(attack_instance, text)
                    result = watermarker.detect(attacked_text)
                    z_scores.append(result.z_score)
                    if result.is_watermarked:
                        detected += 1
                except Exception as e:
                    z_scores.append(0)

            detection_rate = detected / len(generated_texts) * 100 if generated_texts else 0
            mean_z = np.mean(z_scores) if z_scores else 0

            print(f"    Detection: {detection_rate:.1f}%, Mean Z: {mean_z:.2f}")

            all_results.append({
                'config': config['name'],
                'alpha': config['alpha'],
                'omega': config['omega'],
                'salted': config['salted'],
                'sr_enabled': config['sr'],
                'attack': attack_name,
                'detection_rate': detection_rate,
                'detected_count': detected,
                'total_samples': len(generated_texts),
                'mean_z_score': mean_z,
                'std_z_score': np.std(z_scores) if z_scores else 0,
            })

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")

    # Print summary
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)

    # Clean detection rates
    print("\nClean detection rates by config:")
    clean_df = df[df['attack'] == 'clean'][['config', 'alpha', 'omega', 'detection_rate', 'mean_z_score']]
    print(clean_df.to_string(index=False))

    # Average robustness (across all attacks)
    print("\nAverage detection rate across all attacks:")
    avg_df = df.groupby('config').agg({
        'detection_rate': 'mean',
        'mean_z_score': 'mean'
    }).round(2).reset_index()
    avg_df.columns = ['config', 'avg_detection', 'avg_z_score']
    print(avg_df.sort_values('avg_detection', ascending=False).to_string(index=False))

    # Best config for each attack
    print("\nBest config per attack:")
    for attack in attacks.keys():
        attack_df = df[df['attack'] == attack]
        best = attack_df.loc[attack_df['detection_rate'].idxmax()]
        print(f"  {attack}: {best['config']} ({best['detection_rate']:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--output", type=str, default="results/gpw_ablation.csv")
    args = parser.parse_args()

    run_ablation(args)
