"""
Utilities for watermarking experiments.

Functions for:
- CSV logging (per-sample results)
- Model loading
- Argument parsing
- Result aggregation
"""

import os
import csv
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_csv_logger(output_path: str, fieldnames: List[str]) -> csv.DictWriter:
    """Create CSV file and return writer.
    
    Args:
        output_path: Path to output CSV file
        fieldnames: List of column names
    
    Returns:
        CSV DictWriter object
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    file_handle = open(output_path, 'w', newline='')
    writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
    writer.writeheader()
    
    return writer, file_handle


def log_sample_result(writer: csv.DictWriter, row_dict: Dict[str, Any]):
    """Log a single sample result to CSV.
    
    Args:
        writer: CSV DictWriter
        row_dict: Dictionary with sample data
    """
    writer.writerow(row_dict)


def load_model_and_tokenizer(model_config: Dict[str, Any], device: str = "cuda"):
    """Load language model and tokenizer.
    
    Args:
        model_config: Model configuration dict (from config.py)
        device: Device to load model on
    
    Returns:
        model, tokenizer, device
    """
    print(f"Loading model: {model_config['name']}...")
    
    # Determine dtype
    dtype = torch.float16 if model_config['dtype'] == "float16" and device == "cuda" else torch.float32
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_config['name'],
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    
    if device != "cuda" or "device_map" not in locals():
        model = model.to(device)
    
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    
    return model, tokenizer, device


def load_watermarker(watermark_type: str, model, tokenizer, params: Dict[str, Any], device: str = "cuda"):
    """Load watermarker instance.
    
    Args:
        watermark_type: Type of watermark ("unigram", "kgw", "semstamp", "gpw")
        model: Language model
        tokenizer: Tokenizer
        params: Watermarker parameters
        device: Device
    
    Returns:
        Watermarker instance
    """
    if watermark_type == "unigram":
        from watermarkers import UnigramWatermark
        return UnigramWatermark(
            model=model,
            tokenizer=tokenizer,
            gamma=params['gamma'],
            delta=params['delta'],
            z_threshold=params['z_threshold'],
            device=device
        )
    
    elif watermark_type == "kgw":
        from watermarkers import KGWWatermark
        return KGWWatermark(
            model=model,
            tokenizer=tokenizer,
            gamma=params['gamma'],
            delta=params['delta'],
            z_threshold=params['z_threshold'],
            hash_key=params['hash_key'],
            seeding_scheme=params['seeding_scheme'],
            device=device
        )
    
    elif watermark_type == "semstamp":
        from watermarkers import SEMSTAMPWatermark
        from sentence_transformers import SentenceTransformer
        
        embedder = SentenceTransformer(params['embedder_name'])
        return SEMSTAMPWatermark(
            model=model,
            tokenizer=tokenizer,
            embedder=embedder,
            lsh_dim=params['lsh_dim'],
            z_threshold=params['z_threshold'],
            hash_key=params['hash_key'],
            max_rejections=params['max_rejections'],
            device=device
        )
    
    elif watermark_type == "gpw":
        from watermarkers import create_gpw_variant
        return create_gpw_variant(
            model=model,
            tokenizer=tokenizer,
            variant=params['variant'],
            alpha=params['alpha'],
            omega=params['omega'],
            device=device
        )
    
    else:
        raise ValueError(f"Unknown watermark type: {watermark_type}")


def load_attack(attack_type: str, params: Dict[str, Any]):
    """Load attack instance.
    
    Args:
        attack_type: Type of attack
        params: Attack parameters
    
    Returns:
        Attack instance
    """
    if attack_type == "synonym":
        from attacks import SynonymAttack
        return SynonymAttack(**params)
    
    elif attack_type == "swap":
        from attacks import SwapAttack
        return SwapAttack(**params)
    
    elif attack_type == "typo":
        from attacks import TypoAttack
        return TypoAttack(**params)
    
    elif attack_type == "copypaste":
        from attacks import CopyPasteAttack
        return CopyPasteAttack(**params)
    
    elif attack_type == "paraphrase_pegasus":
        from attacks import PegasusAttack
        return PegasusAttack(**params)
    
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")


def create_experiment_parser(description: str) -> argparse.ArgumentParser:
    """Create argument parser for experiments.
    
    Args:
        description: Experiment description
    
    Returns:
        ArgumentParser
    """
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('--model', type=str, default='opt-1.3b',
                        choices=['gpt2', 'opt-1.3b', 'qwen-7b'],
                        help='Model to use')
    parser.add_argument('--watermark', type=str, default='unigram',
                        choices=['unigram', 'kgw', 'semstamp', 'gpw'],
                        help='Watermark method')
    parser.add_argument('--num_samples', type=int, default=200,
                        help='Number of samples to process')
    parser.add_argument('--output_dir', type=str, default='../results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser


def compute_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute aggregate metrics from per-sample results.
    
    Args:
        results: List of per-sample result dictionaries
    
    Returns:
        Dictionary of aggregate metrics
    """
    if not results:
        return {}
    
    # Extract z-scores
    z_scores = [r['z_score'] for r in results if 'z_score' in r]
    
    # Compute statistics
    metrics = {
        'num_samples': len(results),
        'mean_z_score': sum(z_scores) / len(z_scores) if z_scores else 0.0,
        'min_z_score': min(z_scores) if z_scores else 0.0,
        'max_z_score': max(z_scores) if z_scores else 0.0,
        'detection_rate': sum(1 for r in results if r.get('is_watermarked', False)) / len(results),
    }
    
    return metrics


def save_aggregate_metrics(metrics: Dict[str, float], output_path: str):
    """Save aggregate metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to output JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Aggregate metrics saved to: {output_path}")
