"""
Dataset loader for C4 RealNewsLike dataset.

Follows the user's specified pattern for loading and processing C4 data.
"""

import random
from typing import List, Dict, Any
from datasets import load_dataset


def load_c4_dataset(
    num_samples: int = 200,
    min_length: int = 100,
    max_length: int = 1000,
    prompt_words: int = 30,
    seed: int = 42
) -> List[Dict[str, str]]:
    """Load C4 RealNewsLike dataset.
    
    Args:
        num_samples: Number of samples to collect
        min_length: Minimum text length (characters)
        max_length: Maximum text length (characters)
        prompt_words: Number of words to use as prompt
        seed: Random seed
    
    Returns:
        List of dictionaries with 'text', 'prompt', and 'source' keys
    """
    random.seed(seed)
    
    print("Loading C4 RealNewsLike dataset...")
    print("This may take a few minutes for the first download...")
    
    # Load C4 dataset with streaming
    try:
        print("Attempting to load C4 realnewslike dataset...")
        dataset = load_dataset("c4", "realnewslike", split="validation", streaming=True, trust_remote_code=True)
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading with 'c4' name: {e}")
        print("Trying alternative: allenai/c4...")
        try:
            dataset = load_dataset("allenai/c4", "realnewslike", split="validation", streaming=True, trust_remote_code=True)
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
        
        prompt_words_list = words[:prompt_words]
        prompt = " ".join(prompt_words_list)
        
        c4_data.append({
            "text": text,
            "prompt": prompt,
            "source": "c4-realnewslike"
        })
        
        if len(c4_data) >= num_samples:
            break
    
    print(f"\n✓ Loaded {len(c4_data)} C4 samples")
    if len(c4_data) > 0:
        print(f"  Example prompt: {c4_data[0]['prompt'][:100]}...")
    
    return c4_data


def split_human_ai_data(c4_data: List[Dict[str, str]], split_ratio: float = 0.5) -> tuple:
    """Split C4 data into human text and prompts for AI generation.
    
    Args:
        c4_data: List of C4 samples
        split_ratio: Ratio of data to use for human text (rest for AI generation)
    
    Returns:
        (human_texts, ai_prompts) tuple
    """
    split_idx = int(len(c4_data) * split_ratio)
    
    # First half: human texts
    human_texts = [item['text'] for item in c4_data[:split_idx]]
    
    # Second half: prompts for AI generation
    ai_prompts = [item['prompt'] for item in c4_data[split_idx:]]
    
    return human_texts, ai_prompts
