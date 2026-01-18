"""
Dataset Loaders for Watermark Experiments

Loaders for datasets used in watermarking papers:
- C4 (Raffel et al., 2020) - Used by KGW, SEMSTAMP
- OpenGen (Krishna et al., 2023) - Used by Unigram
- LFQA/ELI5 (Fan et al., 2019) - Used by Unigram
- RealNews - Used by SEMSTAMP
- BookSum - Used by SEMSTAMP
"""

from typing import List, Dict, Any, Optional, Tuple
from datasets import load_dataset
import random
import time
from tqdm.auto import tqdm


def load_c4(
    split: str = "validation",
    num_samples: int = 500,
    min_length: int = 100,
    max_length: int = 1000,
    subset: str = "realnewslike",
    seed: int = 42,
    max_retries: int = 3,
    timeout: int = 300
) -> List[Dict[str, str]]:
    """Load C4 dataset for watermark experiments.
    
    C4 (Colossal Clean Crawled Corpus) is commonly used for
    evaluating text generation and watermarking.
    
    Args:
        split: Dataset split ("train" or "validation")
        num_samples: Number of samples to return
        min_length: Minimum text length in characters
        max_length: Maximum text length in characters
        subset: C4 subset ("realnewslike" or "en")
        seed: Random seed for sampling
        max_retries: Maximum number of retry attempts
        timeout: Timeout in seconds for dataset loading
        
    Returns:
        List of dicts with "text" and "prompt" keys
        
    Note:
        KGW paper uses C4 RealNewsLike.
        SEMSTAMP uses C4 for training sentence encoder.
        WaterPark uses C4 for main experiments.
        
    Example:
        >>> data = load_c4(num_samples=100)
        >>> print(f"Loaded {len(data)} samples")
        >>> print(f"First prompt: {data[0]['prompt'][:50]}...")
    """
    random.seed(seed)
    
    print(f"Loading C4 dataset (subset={subset}, split={split})...")
    print(f"This may take a few minutes for the first download...")
    
    # Check datasets version first
    try:
        import datasets
        datasets_version = datasets.__version__
        print(f"Using datasets library version: {datasets_version}")
        major_version = int(datasets_version.split('.')[0])
        
        if major_version >= 4:
            print("\n" + "!"*70)
            print("WARNING: datasets >= 4.0.0 detected!")
            print("C4 dataset script loading is not supported in this version.")
            print("Attempting to use alternative loading method...")
            print("!"*70 + "\n")
    except:
        major_version = 4  # Assume newer version
    
    # Try loading with retries
    dataset = None
    for attempt in range(max_retries):
        try:
            if subset == "realnewslike":
                print(f"Attempt {attempt + 1}/{max_retries}: Loading C4 realnewslike...")
                
                # Method 1: Try old API first (works with datasets < 4.0)
                try:
                    dataset = load_dataset("c4", "realnewslike", split=split, streaming=True)
                    print("Successfully loaded using old API (c4/realnewslike)")
                except Exception as e1:
                    error_str = str(e1).lower()
                    # Check if it's the script error
                    if "no longer supported" in error_str or "dataset scripts" in error_str or "c4.py" in error_str:
                        print("Old API failed (scripts not supported). Trying alternative methods...")
                        
                        # Method 2: Try allenai/c4 with config name
                        try:
                            print("Trying allenai/c4 with config name...")
                            dataset = load_dataset(
                                "allenai/c4",
                                "realnewslike",
                                split=split,
                                streaming=True
                            )
                            print("Successfully loaded using allenai/c4")
                        except Exception as e2:
                            # Method 3: Try with data_files (for datasets >= 4.0)
                            try:
                                print("Trying data_files method...")
                                # Load first shard only for testing
                                dataset = load_dataset(
                                    "allenai/c4",
                                    data_files={
                                        split: f"realnewslike/c4-{split}.00000-of-*.json.gz"
                                    },
                                    split=split,
                                    streaming=True
                                )
                                print("Successfully loaded using data_files method")
                            except Exception as e3:
                                # All methods failed - provide clear error
                                raise RuntimeError(
                                    "\n" + "="*70 + "\n"
                                    "ERROR: Cannot load C4 dataset with current datasets version\n"
                                    "="*70 + "\n"
                                    "The C4 dataset requires datasets < 4.0.0 to use the loading script.\n\n"
                                    "IMMEDIATE FIX:\n"
                                    "1. Run this command in a cell:\n"
                                    "   !pip install 'datasets<4.0.0' --force-reinstall\n"
                                    "2. RESTART RUNTIME (Runtime -> Restart runtime)\n"
                                    "3. Run cells again from the beginning\n\n"
                                    "Alternative: Use a different dataset or preprocessed C4 data\n"
                                    "="*70 + "\n"
                                    f"Error details:\n- Old API: {e1}\n- New API: {e2}\n- Data files: {e3}"
                                ) from e3
                    else:
                        # Different error, re-raise
                        raise e1
                        
            else:
                print(f"Attempt {attempt + 1}/{max_retries}: Loading C4 en...")
                try:
                    dataset = load_dataset("c4", "en", split=split, streaming=True)
                except Exception as e1:
                    if "no longer supported" in str(e1) or "Dataset scripts" in str(e1) or "c4.py" in str(e1):
                        try:
                            dataset = load_dataset("allenai/c4", "en", split=split, streaming=True)
                        except Exception as e2:
                            try:
                                dataset = load_dataset(
                                    "allenai/c4",
                                    data_files={split: f"en/c4-{split}.00000-of-*.json.gz"},
                                    split=split,
                                    streaming=True
                                )
                            except Exception as e3:
                                raise RuntimeError(
                                    f"Cannot load C4 dataset. Run: !pip install 'datasets<4.0.0' --force-reinstall\n"
                                    f"Then restart runtime. Errors: {e1}, {e2}, {e3}"
                                ) from e3
                    else:
                        raise e1
            
            # Test if dataset is accessible by trying to get first item
            print("Testing dataset connection...")
            test_item = next(iter(dataset))
            if "text" not in test_item:
                raise ValueError("Dataset item missing 'text' field")
            print("Dataset loaded successfully!")
            break
            
        except RuntimeError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            error_msg = str(e).lower()
            if "dataset scripts are no longer supported" in error_msg or "no longer supported" in error_msg or "c4.py" in error_msg:
                raise RuntimeError(
                    "\n" + "="*70 + "\n"
                    "ERROR: Dataset scripts are no longer supported\n"
                    "="*70 + "\n"
                    "QUICK FIX:\n"
                    "1. Run: !pip install 'datasets<4.0.0' --force-reinstall\n"
                    "2. RESTART RUNTIME (Runtime -> Restart runtime)\n"
                    "3. Run all cells from the beginning\n"
                    "="*70 + "\n"
                    f"Original error: {e}"
                ) from e
            
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"Error loading dataset: {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(
                    f"Failed to load C4 dataset after {max_retries} attempts.\n"
                    f"Last error: {e}\n\n"
                    f"If error mentions 'Dataset scripts', run:\n"
                    f"  !pip install 'datasets<4.0.0' --force-reinstall\n"
                    f"Then restart runtime."
                ) from e
    
    if dataset is None:
        raise RuntimeError("Failed to initialize dataset")
    
    # Collect samples with progress bar
    samples = []
    print(f"Collecting {num_samples} samples (filtering by length {min_length}-{max_length} chars)...")
    
    try:
        with tqdm(total=num_samples, desc="Loading samples") as pbar:
            for item in dataset:
                try:
                    text = item.get("text", "")
                    
                    if not text or len(text) < min_length or len(text) > max_length:
                        continue
                    
                    # Create prompt from first sentence(s) (~50 tokens)
                    words = text.split()
                    if len(words) < 10:
                        continue
                    
                    prompt_words = words[:30]  # Approximately 30 words as prompt
                    prompt = " ".join(prompt_words)
                    
                    samples.append({
                        "text": text,
                        "prompt": prompt,
                        "source": "c4-realnewslike" if subset == "realnewslike" else "c4-en"
                    })
                    
                    pbar.update(1)
                    
                    if len(samples) >= num_samples:
                        break
                        
                except Exception as e:
                    print(f"Warning: Error processing item: {e}")
                    continue
                    
    except KeyboardInterrupt:
        print(f"\nInterrupted. Collected {len(samples)} samples so far.")
        if len(samples) == 0:
            raise RuntimeError("No samples collected. Please try again.")
    
    if len(samples) < num_samples:
        print(f"Warning: Only collected {len(samples)}/{num_samples} samples.")
        if len(samples) == 0:
            raise RuntimeError(
                "No samples collected. This might be due to:\n"
                "1. Network connectivity issues\n"
                "2. Dataset download timeout\n"
                "3. Length filtering too strict\n"
                "Try reducing min_length or increasing max_length."
            )
    
    print(f"Successfully loaded {len(samples)} samples!")
    return samples


def load_opengen(
    num_samples: int = 500,
    seed: int = 42
) -> List[Dict[str, str]]:
    """Load OpenGen dataset for watermark experiments.
    
    OpenGen contains human-written text paired with prompts,
    used as baseline in Unigram-Watermark paper.
    
    Args:
        num_samples: Number of samples to return
        seed: Random seed
        
    Returns:
        List of dicts with "text", "prompt", "source" keys
        
    Note:
        Used in Zhao et al. (2023) for comparing watermarked
        text against human-written text.
        
    Example:
        >>> data = load_opengen(num_samples=100)
        >>> print(f"Loaded {len(data)} samples")
    """
    random.seed(seed)
    
    # OpenGen is part of the detection-paraphrases dataset
    try:
        dataset = load_dataset(
            "martiansideofthemoon/detection-paraphrases",
            split="test"
        )
    except Exception:
        # Fallback: generate synthetic prompts from C4
        print("Warning: OpenGen not available, falling back to C4")
        return load_c4(num_samples=num_samples, seed=seed)
    
    samples = []
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for idx in indices:
        item = dataset[idx]
        samples.append({
            "text": item.get("text", item.get("human_text", "")),
            "prompt": item.get("prompt", ""),
            "source": "opengen"
        })
    
    return samples


def load_lfqa(
    num_samples: int = 500,
    min_answer_length: int = 100,
    seed: int = 42,
    max_retries: int = 3
) -> List[Dict[str, str]]:
    """Load LFQA (Long-Form Question Answering) / ELI5 dataset.
    
    Contains questions and long-form answers from Reddit's
    "Explain Like I'm Five" forum.
    
    Args:
        num_samples: Number of samples to return
        min_answer_length: Minimum answer length
        seed: Random seed
        max_retries: Maximum number of retry attempts
        
    Returns:
        List of dicts with "question", "answer", "prompt" keys
        
    Note:
        Used in Zhao et al. (2023) for QA generation experiments.
        Questions serve as prompts for generating watermarked answers.
        
    Example:
        >>> data = load_lfqa(num_samples=100)
        >>> print(f"Question: {data[0]['prompt']}")
        >>> print(f"Answer: {data[0]['text'][:100]}...")
    """
    random.seed(seed)
    
    print(f"Loading ELI5/LFQA dataset...")
    print(f"This may take a few minutes for the first download...")
    
    # Try loading with retries
    dataset = None
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}: Loading eli5 dataset...")
            dataset = load_dataset("eli5", split="test_eli5")
            print("Dataset loaded successfully!")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"Error loading dataset: {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(
                    f"Failed to load ELI5 dataset after {max_retries} attempts. "
                    f"Last error: {e}\n"
                    f"Please check your internet connection and try again."
                ) from e
    
    if dataset is None:
        raise RuntimeError("Failed to initialize dataset")
    
    samples = []
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    print(f"Collecting {num_samples} samples...")
    with tqdm(total=num_samples, desc="Loading samples") as pbar:
        for idx in indices:
            try:
                item = dataset[idx]
                question = item.get("title", "")
                
                if not question:
                    continue
                
                # Get the best answer (most upvoted)
                answers = item.get("answers", {}).get("text", [])
                if not answers:
                    continue
                
                answer = answers[0]  # First answer (usually most relevant)
                
                if len(answer) < min_answer_length:
                    continue
                
                samples.append({
                    "text": answer,
                    "prompt": f"Question: {question}\n\nAnswer:",
                    "question": question,
                    "source": "eli5"
                })
                
                pbar.update(1)
                
                if len(samples) >= num_samples:
                    break
            except Exception as e:
                print(f"Warning: Error processing item {idx}: {e}")
                continue
    
    if len(samples) == 0:
        raise RuntimeError(
            "No samples collected from ELI5 dataset. "
            "This might be due to network issues or dataset download problems."
        )
    
    print(f"Successfully loaded {len(samples)} samples!")
    return samples


def load_realnews(
    num_samples: int = 500,
    min_length: int = 100,
    max_length: int = 1000,
    seed: int = 42
) -> List[Dict[str, str]]:
    """Load RealNews dataset for watermark experiments.
    
    Contains news articles, used for evaluating semantic
    watermarks like SEMSTAMP.
    
    Args:
        num_samples: Number of samples to return
        min_length: Minimum text length
        max_length: Maximum text length
        seed: Random seed
        
    Returns:
        List of dicts with "text", "prompt", "title" keys
        
    Note:
        SEMSTAMP uses RealNews for experiments.
        Article titles can serve as prompts.
        
    Example:
        >>> data = load_realnews(num_samples=100)
        >>> print(f"Title: {data[0]['title']}")
    """
    random.seed(seed)
    
    # RealNews is part of C4 realnewslike
    return load_c4(
        num_samples=num_samples,
        min_length=min_length,
        max_length=max_length,
        subset="realnewslike",
        seed=seed
    )


def load_booksum(
    num_samples: int = 500,
    min_length: int = 100,
    seed: int = 42
) -> List[Dict[str, str]]:
    """Load BookSum dataset for watermark experiments.
    
    Contains book summaries, used for long-form generation
    experiments.
    
    Args:
        num_samples: Number of samples to return
        min_length: Minimum summary length
        seed: Random seed
        
    Returns:
        List of dicts with "text", "prompt", "book_title" keys
        
    Note:
        SEMSTAMP paper mentions BookSum for experiments.
        Summaries require maintaining coherence over longer text.
        
    Example:
        >>> data = load_booksum(num_samples=100)
        >>> print(f"Book: {data[0]['book_title']}")
    """
    random.seed(seed)
    
    try:
        dataset = load_dataset("kmfoda/booksum", split="test")
    except Exception:
        # Fallback
        print("Warning: BookSum not available, falling back to C4")
        return load_c4(num_samples=num_samples, seed=seed)
    
    samples = []
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    for idx in indices:
        item = dataset[idx]
        summary = item.get("summary", item.get("summary_text", ""))
        
        if len(summary) < min_length:
            continue
        
        title = item.get("book_title", item.get("title", "Unknown"))
        
        # Create prompt from first part of summary
        words = summary.split()
        prompt = " ".join(words[:30])
        
        samples.append({
            "text": summary,
            "prompt": f"Summary of {title}: {prompt}",
            "book_title": title,
            "source": "booksum"
        })
        
        if len(samples) >= num_samples:
            break
    
    return samples


def create_experiment_dataset(
    dataset_name: str,
    num_prompts: int = 100,
    num_human_texts: int = 100,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """Create paired dataset for watermark experiments.
    
    Returns prompts for generation and human texts for comparison.
    
    Args:
        dataset_name: One of "c4", "opengen", "lfqa", "realnews", "booksum"
        num_prompts: Number of prompts for generation
        num_human_texts: Number of human texts for baseline
        seed: Random seed
        
    Returns:
        Tuple of (prompts, human_texts)
        
    Example:
        >>> prompts, human_texts = create_experiment_dataset("c4", 100, 100)
        >>> print(f"Got {len(prompts)} prompts and {len(human_texts)} human texts")
    """
    loaders = {
        "c4": load_c4,
        "opengen": load_opengen,
        "lfqa": load_lfqa,
        "realnews": load_realnews,
        "booksum": load_booksum,
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Choose from: {list(loaders.keys())}")
    
    loader = loaders[dataset_name]
    
    # Load samples
    total_needed = num_prompts + num_human_texts
    data = loader(num_samples=total_needed, seed=seed)
    
    # Split into prompts and human texts
    random.seed(seed + 1)
    random.shuffle(data)
    
    prompts = [item["prompt"] for item in data[:num_prompts]]
    human_texts = [item["text"] for item in data[num_prompts:num_prompts + num_human_texts]]
    
    return prompts, human_texts
