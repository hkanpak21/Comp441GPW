"""
Experiment Configuration

Shared hyperparameters and settings for all experiments.
"""

import torch

# ============================================================================
# Model Configuration
# ============================================================================

# Primary model for experiments
MODEL_NAME = "facebook/opt-1.3b"  # Start with this, later switch to Qwen-7B

# Alternative models (for future experiments)
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_NAME = "gpt2"  # For quick debugging

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ============================================================================
# Dataset Configuration
# ============================================================================

DATASET_NAME = "c4"
DATASET_CONFIG = "realnewslike"
DATASET_SPLIT = "validation"
NUM_SAMPLES = 200
MIN_TEXT_LENGTH = 100
MAX_TEXT_LENGTH = 1000
PROMPT_LENGTH = 30  # words for prompt extraction

# Seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# Watermarker Configuration
# ============================================================================

# Unigram
UNIGRAM_CONFIG = {
    "gamma": 0.5,
    "delta": 2.0,
    "z_threshold": 4.0,
}

# KGW
KGW_CONFIG = {
    "gamma": 0.5,
    "delta": 2.0,
    "z_threshold": 4.0,
    "context": "simple_1",  # bigram context
}

# SEMSTAMP
SEMSTAMP_CONFIG = {
    "lsh_dim": 3,
    "z_threshold": 4.0,
    "max_rejections": 1,  # User requirement: only 1 attempt
}

# GPW - Original params
GPW_CONFIG = {
    "alpha": 2.0,
    "omega": 2.0,
    "z_threshold": 4.0,
}

# GPW-SP with TUNED parameters for better attack robustness:
# - Lower omega (1.0): Wider green bands = more robust to synonym/swap
# - Higher alpha (3.0): Stronger bias = higher z-scores to compensate
GPW_SP_CONFIG = {
    "alpha": 3.0,     # Increased from 2.0 for stronger signal
    "omega": 1.0,     # Decreased from 2.0 for better attack robustness
    "z_threshold": 4.0,
    "variant": "GPW-SP",
}

# GPW-SP Robust variant for comparison experiments
GPW_SP_ROBUST_CONFIG = {
    "alpha": 4.0,     # Even stronger
    "omega": 0.5,     # Very wide green bands for maximum robustness
    "z_threshold": 4.0,
    "variant": "GPW-SP",
}

# ============================================================================
# Generation Configuration
# ============================================================================

GENERATION_CONFIG = {
    "max_new_tokens": 120,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 0.95,
    "do_sample": True,
}

# ============================================================================
# Attack Configuration
# ============================================================================

# Lexical attacks
SYNONYM_ATTACK_CONFIG = {
    "edit_rate": 0.3,  # Replace 30% of words
}

SWAP_ATTACK_CONFIG = {
    "edit_rate": 0.2,  # Swap 20% of adjacent words
}

TYPO_ATTACK_CONFIG = {
    "edit_rate": 0.1,  # 10% character errors
}

# Text mixing attacks
COPYPASTE_ATTACK_CONFIG = {
    "n_segments": 3,
    "watermark_ratio": 0.5,  # 50% watermarked, 50% human
}

MIX_ATTACK_CONFIG = {
    "watermarked_ratio": 0.5,  # Alternate sentences
}

# Paraphrase attacks
PARAPHRASE_ATTACK_CONFIG = {
    "num_beams": 3,
    "temperature": 1.5,
}

# ============================================================================
# Results Configuration
# ============================================================================

RESULTS_DIR = "/scratch/hkanpak21/Comp441GPW/results"

# CSV column names
SAMPLE_CSV_COLUMNS = [
    "sample_id",
    "prompt",
    "generated_text",
    "watermarker",
    "variant",
    "z_score",
    "p_value",
    "is_detected",
    "num_tokens",
    "green_fraction",
    "attack",
    "attack_params",
    "generation_time",
    "detection_time",
]

SUMMARY_CSV_COLUMNS = [
    "watermarker",
    "variant",
    "num_samples",
    "mean_z_score",
    "std_z_score",
    "median_z_score",
    "tpr",
    "fpr",
    "auc_roc",
    "tpr_at_fpr_1pct",
    "tpr_at_fpr_5pct",
    "mean_detection_time",
    "total_time",
]

# ============================================================================
# Experiment Metadata
# ============================================================================

WATERMARKER_LIST = ["unigram", "kgw", "gpw", "gpw_sp"]  # "semstamp" optional

# Attack ordering (simple → complex)
ATTACK_ORDER = [
    "none",
    "synonym",
    "swap",
    "typo",
    "copypaste",
    "mix",
    "paraphrase",
]

# ============================================================================
# Structured configurations for utils functions
# ============================================================================

MODEL_CONFIGS = {
    "gpt2": {
        "name": "gpt2",
        "dtype": "float32",  # GPT2 works better with float32
    },
    "opt-1.3b": {
        "name": "facebook/opt-1.3b",
        "dtype": "float16",
    },
    "qwen-7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "dtype": "float16",
    },
    "qwen-14b": {
        "name": "Qwen/Qwen2.5-14B-Instruct",
        "dtype": "float16",
    },
}

WATERMARK_PARAMS = {
    "unigram": {
        "gamma": 0.5,
        "delta": 2.0,
        "z_threshold": 4.0,
    },
    "kgw": {
        "gamma": 0.5,
        "delta": 2.0,
        "z_threshold": 4.0,
        "hash_key": 15485863,
        "seeding_scheme": "simple_1",
    },
    "semstamp": {
        "embedder_name": "sentence-transformers/all-MiniLM-L6-v2",
        "lsh_dim": 3,
        "z_threshold": 4.0,
        "hash_key": 15485863,
        "max_rejections": 1,
    },
    "gpw": {
        "variant": "GPW",
        "alpha": 2.0,
        "omega": 2.0,
        "z_threshold": 4.0,
        "sr_cfg": {"enabled": False},
    },
    # GPW-SP with TUNED parameters for attack robustness
    "gpw_sp": {
        "variant": "GPW-SP",
        "alpha": 3.0,     # Stronger bias
        "omega": 1.0,     # Wider green bands = more robust
        "z_threshold": 4.0,
        "sr_cfg": {"enabled": False},
    },
    # Original GPW-SP params for comparison
    "gpw_sp_orig": {
        "variant": "GPW-SP",
        "alpha": 2.0,
        "omega": 2.0,
        "z_threshold": 4.0,
        "sr_cfg": {"enabled": False},
    },
    # Maximum robustness variant
    "gpw_sp_robust": {
        "variant": "GPW-SP",
        "alpha": 4.0,
        "omega": 0.5,
        "z_threshold": 4.0,
        "sr_cfg": {"enabled": False},
    },
}

GENERATION_PARAMS = GENERATION_CONFIG
