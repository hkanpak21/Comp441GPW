"""
Experiment Utilities

Shared functions for experiment scripts:
- Dataset loading (C4)
- Model initialization
- CSV logging
- Metric computation
"""

import os
import csv
import time
import random
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from config import *


def set_seed(seed: int = RANDOM_SEED):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_c4_dataset(
    num_samples: int = NUM_SAMPLES,
    min_length: int = MIN_TEXT_LENGTH,
    max_length: int = MAX_TEXT_LENGTH,
    prompt_length: int = PROMPT_LENGTH,
    seed: int = RANDOM_SEED,
) -> List[Dict[str, str]]:
    """Load C4 RealNewsLike dataset.
    
    Args:
        num_samples: Number of samples to collect
        min_length: Minimum text length (characters)
        max_length: Maximum text length (characters)
        prompt_length: Number of words for prompt extraction
        seed: Random seed
        
    Returns:
        List of dicts with keys: text, prompt, source
    """
    print(f"Loading C4 RealNewsLike dataset...")
    print(f"This may take a few minutes for the first download...")
    
    random.seed(seed)
    
    # Load C4 dataset with streaming
    try:
        print("Attempting to load C4 realnewslike dataset...")
        dataset = load_dataset(
            DATASET_NAME,
            DATASET_CONFIG,
            split=DATASET_SPLIT,
            streaming=True,
            trust_remote_code=True
        )
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading with '{DATASET_NAME}' name: {e}")
        print("Trying alternative: allenai/c4...")
        try:
            dataset = load_dataset(
                "allenai/c4",
                DATASET_CONFIG,
                split=DATASET_SPLIT,
                streaming=True,
                trust_remote_code=True
            )
            print("Dataset loaded successfully with allenai/c4!")
        except Exception as e2:
            raise RuntimeError(f"Failed to load C4 dataset. Errors: {e}, {e2}") from e2
    
    # Collect samples
    c4_data = []
    print(f"Collecting {num_samples} samples...")
    
    for item in dataset:
        text = item.get("text", "")
        
        # Filter by length
        if len(text) < min_length or len(text) > max_length:
            continue
        
        # Create prompt from first N words
        words = text.split()
        if len(words) < 10:
            continue
        
        prompt_words = words[:prompt_length]
        prompt = " ".join(prompt_words)
        
        c4_data.append({
            "text": text,
            "prompt": prompt,
            "source": "c4-realnewslike"
        })
        
        if len(c4_data) >= num_samples:
            break
    
    print(f"\nLoaded {len(c4_data)} C4 samples")
    if len(c4_data) > 0:
        print(f"Example prompt: {c4_data[0]['prompt'][:100]}...")
    
    return c4_data


def load_model_and_tokenizer(
    model_name: str = MODEL_NAME,
    device: str = DEVICE,
    dtype: torch.dtype = DTYPE,
) -> Tuple[Any, Any]:
    """Load model and tokenizer.
    
    Args:
        model_name: HuggingFace model name
        device: Device to load model on
        dtype: Data type for model
        
    Returns:
        (model, tokenizer) tuple
    """
    print(f"\nLoading model: {model_name}")
    print(f"Device: {device}, dtype: {dtype}")
    
    start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    
    if device == "cpu":
        model = model.to(device)
    
    elapsed = time.time() - start
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    
    print(f"✓ Model loaded in {elapsed:.1f}s")
    print(f"  Parameters: {num_params:.1f}M")
    print(f"  Device: {next(model.parameters()).device}")
    
    return model, tokenizer


def initialize_watermarkers(
    model,
    tokenizer,
    device: str = DEVICE,
    include_semstamp: bool = False,
) -> Dict[str, Any]:
    """Initialize all watermarkers.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        device: Device
        include_semstamp: Whether to include SEMSTAMP (requires sentence encoder)
        
    Returns:
        Dict mapping watermarker name to instance
    """
    from watermarkers import (
        UnigramWatermark,
        KGWWatermark,
        create_gpw_variant,
    )
    
    print("\nInitializing watermarkers...")
    
    watermarkers = {}
    
    # Unigram
    print("  - Unigram...")
    watermarkers["unigram"] = UnigramWatermark(
        model, tokenizer, device=device, **UNIGRAM_CONFIG
    )
    
    # KGW
    print("  - KGW...")
    watermarkers["kgw"] = KGWWatermark(
        model, tokenizer, device=device, **KGW_CONFIG
    )
    
    # GPW
    print("  - GPW...")
    watermarkers["gpw"] = create_gpw_variant(
        model, tokenizer, variant="GPW", device=device,
        alpha=GPW_CONFIG["alpha"], omega=GPW_CONFIG["omega"]
    )
    
    # GPW-SP
    print("  - GPW-SP...")
    watermarkers["gpw_sp"] = create_gpw_variant(
        model, tokenizer, variant="GPW-SP", device=device,
        alpha=GPW_SP_CONFIG["alpha"], omega=GPW_SP_CONFIG["omega"]
    )
    
    # SEMSTAMP (optional)
    if include_semstamp:
        try:
            from watermarkers import SEMSTAMPWatermark
            from sentence_transformers import SentenceTransformer
            
            print("  - SEMSTAMP (loading encoder)...")
            embedder = SentenceTransformer("all-mpnet-base-v2")
            watermarkers["semstamp"] = SEMSTAMPWatermark(
                model, tokenizer, embedder=embedder, device=device, **SEMSTAMP_CONFIG
            )
        except Exception as e:
            print(f"  ⚠ SEMSTAMP skipped: {e}")
    
    print(f"✓ Initialized {len(watermarkers)} watermarkers")
    
    return watermarkers


def save_samples_csv(
    filepath: str,
    samples: List[Dict[str, Any]],
    columns: List[str] = SAMPLE_CSV_COLUMNS,
):
    """Save per-sample results to CSV.
    
    Args:
        filepath: Output CSV file path
        samples: List of sample dictionaries
        columns: Column names
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(samples)
    
    print(f"✓ Saved {len(samples)} samples to: {filepath}")


def save_summary_csv(
    filepath: str,
    summary: List[Dict[str, Any]],
    columns: List[str] = SUMMARY_CSV_COLUMNS,
):
    """Save summary metrics to CSV.
    
    Args:
        filepath: Output CSV file path
        summary: List of summary dictionaries
        columns: Column names
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary)
    
    print(f"✓ Saved summary to: {filepath}")


def compute_detection_metrics(
    z_scores_positive: List[float],
    z_scores_negative: List[float],
    threshold: float = 4.0,
) -> Dict[str, float]:
    """Compute detection metrics.
    
    Args:
        z_scores_positive: Z-scores for watermarked texts
        z_scores_negative: Z-scores for non-watermarked texts
        threshold: Detection threshold
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    import scipy.stats as stats
    
    # Convert to numpy
    z_pos = np.array(z_scores_positive)
    z_neg = np.array(z_scores_negative)
    
    # Compute TPR and FPR at threshold
    tpr = np.mean(z_pos > threshold)
    fpr = np.mean(z_neg > threshold)
    
    # Compute AUC-ROC
    labels = np.concatenate([np.ones(len(z_pos)), np.zeros(len(z_neg))])
    scores = np.concatenate([z_pos, z_neg])
    
    try:
        auc_roc = roc_auc_score(labels, scores)
    except:
        auc_roc = 0.5
    
    # Compute TPR at specific FPR levels
    try:
        fpr_curve, tpr_curve, thresholds = roc_curve(labels, scores)
        
        # TPR at FPR = 1%
        idx_1pct = np.argmin(np.abs(fpr_curve - 0.01))
        tpr_at_1pct = tpr_curve[idx_1pct]
        
        # TPR at FPR = 5%
        idx_5pct = np.argmin(np.abs(fpr_curve - 0.05))
        tpr_at_5pct = tpr_curve[idx_5pct]
    except:
        tpr_at_1pct = 0.0
        tpr_at_5pct = 0.0
    
    return {
        "mean_z_score": float(np.mean(z_pos)),
        "std_z_score": float(np.std(z_pos)),
        "median_z_score": float(np.median(z_pos)),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "auc_roc": float(auc_roc),
        "tpr_at_fpr_1pct": float(tpr_at_1pct),
        "tpr_at_fpr_5pct": float(tpr_at_5pct),
    }


def get_experiment_name(base_name: str) -> str:
    """Generate experiment name with timestamp.
    
    Args:
        base_name: Base experiment name
        
    Returns:
        Experiment name with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}"


def print_experiment_header(name: str):
    """Print experiment header."""
    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {name}")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print("=" * 80 + "\n")


def print_experiment_summary(
    num_samples: int,
    watermarkers: List[str],
    elapsed_time: float,
):
    """Print experiment summary."""
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED")
    print("=" * 80)
    print(f"Samples processed: {num_samples}")
    print(f"Watermarkers: {', '.join(watermarkers)}")
    print(f"Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
