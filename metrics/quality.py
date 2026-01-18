"""
Quality Metrics for Watermark Evaluation

Metrics for evaluating text quality after watermarking including
perplexity, BERTScore, MAUVE, and diversity measures.

Based on evaluation methods from:
- Zhao et al., 2023 (Unigram-Watermark) - Perplexity
- Kirchenbauer et al., 2023 (KGW) - Perplexity, BERTScore
- Hou et al., 2024 (SEMSTAMP) - BERTScore, P-SP
- Liang et al., 2025 (WaterPark) - MAUVE, BERTScore
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import torch


def compute_perplexity(
    texts: List[str],
    model_name: str = "gpt2",
    device: str = "cuda",
    batch_size: int = 8
) -> Dict[str, float]:
    """Compute perplexity of generated texts.
    
    Perplexity measures how "surprising" the text is to a language model.
    Lower perplexity = higher quality.
    
    Args:
        texts: List of texts to evaluate
        model_name: Hugging Face model name (default: "gpt2")
        device: Device to run on
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with mean and std perplexity
        
    Note:
        Papers typically use GPT-2 or GPT-2-XL for computing perplexity.
        Can also use: "facebook/opt-1.3b", "EleutherAI/gpt-neo-1.3B"
        
    Example:
        >>> texts = ["The quick brown fox.", "Machine learning is great."]
        >>> result = compute_perplexity(texts)
        >>> print(f"Mean PPL: {result['mean']:.2f}")
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    perplexities = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        encodings = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**encodings, labels=encodings["input_ids"])
            
            # Per-sequence perplexity
            for j in range(len(batch)):
                mask = encodings["attention_mask"][j]
                seq_len = mask.sum().item()
                
                if seq_len > 1:
                    # Get per-token losses
                    shift_logits = outputs.logits[j, :-1, :]
                    shift_labels = encodings["input_ids"][j, 1:]
                    shift_mask = mask[1:]
                    
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    losses = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    
                    # Average over non-padding tokens
                    avg_loss = (losses * shift_mask.view(-1)).sum() / shift_mask.sum()
                    ppl = torch.exp(avg_loss).item()
                    
                    # Cap extremely high perplexities
                    ppl = min(ppl, 10000)
                    perplexities.append(ppl)
    
    return {
        "mean": float(np.mean(perplexities)) if perplexities else 0.0,
        "std": float(np.std(perplexities)) if perplexities else 0.0,
        "min": float(np.min(perplexities)) if perplexities else 0.0,
        "max": float(np.max(perplexities)) if perplexities else 0.0,
        "all": perplexities
    }


def compute_bertscore(
    candidates: List[str],
    references: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    device: str = "cuda"
) -> Dict[str, float]:
    """Compute BERTScore between candidate and reference texts.
    
    BERTScore measures semantic similarity using contextual embeddings.
    Higher score = more similar to reference.
    
    Args:
        candidates: Generated/watermarked texts
        references: Reference texts (e.g., prompts or human text)
        model_type: Model for BERTScore (default: DeBERTa)
        device: Device to run on
        
    Returns:
        Dictionary with precision, recall, F1 scores
        
    Note:
        WaterPark uses DeBERTa-xlarge-mnli for BERTScore.
        SEMSTAMP uses roberta-large.
        
    Example:
        >>> candidates = ["The quick brown fox jumps."]
        >>> references = ["A fast brown fox leaps."]
        >>> result = compute_bertscore(candidates, references)
        >>> print(f"F1: {result['f1']:.3f}")
    """
    try:
        from bert_score import score as bert_score
    except ImportError:
        raise ImportError("Please install bert-score: pip install bert-score")
    
    P, R, F1 = bert_score(
        candidates, 
        references, 
        model_type=model_type,
        device=device,
        verbose=False
    )
    
    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F1.mean()),
        "precision_std": float(P.std()),
        "recall_std": float(R.std()),
        "f1_std": float(F1.std()),
    }


def compute_mauve(
    generated_texts: List[str],
    reference_texts: List[str],
    device_id: int = 0,
    max_text_length: int = 256
) -> float:
    """Compute MAUVE score between generated and reference texts.
    
    MAUVE measures the gap between the distribution of generated
    text and real text using neural features.
    
    Args:
        generated_texts: Texts produced by model (possibly watermarked)
        reference_texts: Human-written reference texts
        device_id: GPU device ID
        max_text_length: Maximum text length
        
    Returns:
        MAUVE score (0-1, higher is better)
        
    Note:
        WaterPark uses MAUVE as primary quality metric.
        Score of 1.0 = indistinguishable from human text.
        
    Example:
        >>> generated = ["AI is transforming the world."] * 100
        >>> reference = ["Technology changes everything."] * 100
        >>> score = compute_mauve(generated, reference)
        >>> print(f"MAUVE: {score:.3f}")
    """
    try:
        import mauve
    except ImportError:
        raise ImportError("Please install mauve-text: pip install mauve-text")
    
    # MAUVE requires at least 100 samples for reliable estimates
    if len(generated_texts) < 10 or len(reference_texts) < 10:
        print("Warning: MAUVE works best with at least 100 samples")
    
    result = mauve.compute_mauve(
        p_text=reference_texts,
        q_text=generated_texts,
        device_id=device_id,
        max_text_length=max_text_length,
        verbose=False
    )
    
    return float(result.mauve)


def compute_diversity(
    texts: List[str],
    n_gram_sizes: List[int] = None
) -> Dict[str, float]:
    """Compute diversity metrics for generated texts.
    
    Measures:
    - Distinct-n: Ratio of unique n-grams to total n-grams
    - Repetition-n: Fraction of n-grams that appear more than once
    
    Args:
        texts: List of generated texts
        n_gram_sizes: N-gram sizes to compute (default: [2, 3, 4])
        
    Returns:
        Dictionary with diversity metrics
        
    Note:
        Unigram paper uses Repetition-4 as diversity measure.
        Higher distinct-n = more diverse.
        Lower repetition-n = less repetitive.
        
    Example:
        >>> texts = ["The quick brown fox.", "A fast brown dog."]
        >>> result = compute_diversity(texts)
        >>> print(f"Distinct-2: {result['distinct_2']:.3f}")
    """
    if n_gram_sizes is None:
        n_gram_sizes = [2, 3, 4]
    
    from collections import Counter
    
    all_text = ' '.join(texts).lower()
    words = all_text.split()
    
    results = {}
    
    for n in n_gram_sizes:
        if len(words) < n:
            results[f"distinct_{n}"] = 0.0
            results[f"repetition_{n}"] = 0.0
            continue
        
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        ngram_counts = Counter(ngrams)
        
        total = len(ngrams)
        unique = len(ngram_counts)
        repeated = sum(1 for c in ngram_counts.values() if c > 1)
        
        results[f"distinct_{n}"] = unique / total if total > 0 else 0.0
        results[f"repetition_{n}"] = repeated / unique if unique > 0 else 0.0
    
    return results


def compute_p_sp(
    original_texts: List[str],
    attacked_texts: List[str],
    embedder=None,
    device: str = "cuda"
) -> Dict[str, float]:
    """Compute Paraphrase Semantic Preservation (P-SP) score.
    
    P-SP measures how well semantic meaning is preserved after
    an attack (like paraphrasing).
    
    Args:
        original_texts: Original texts before attack
        attacked_texts: Texts after attack
        embedder: Sentence encoder (default: uses SBERT)
        device: Device to run on
        
    Returns:
        Dictionary with mean and std P-SP scores
        
    Note:
        SEMSTAMP uses P-SP (cosine similarity of sentence embeddings)
        to measure semantic preservation after paraphrase attacks.
        
    Example:
        >>> original = ["The quick brown fox jumps over the lazy dog."]
        >>> attacked = ["A fast brown fox leaps over a sleepy dog."]
        >>> result = compute_p_sp(original, attacked)
        >>> print(f"P-SP: {result['mean']:.3f}")
    """
    if embedder is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-mpnet-base-v2", device=device)
    
    similarities = []
    
    for orig, attacked in zip(original_texts, attacked_texts):
        emb_orig = embedder.encode(orig, convert_to_tensor=True)
        emb_attacked = embedder.encode(attacked, convert_to_tensor=True)
        
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            emb_orig.unsqueeze(0), 
            emb_attacked.unsqueeze(0)
        ).item()
        
        similarities.append(cos_sim)
    
    return {
        "mean": float(np.mean(similarities)) if similarities else 0.0,
        "std": float(np.std(similarities)) if similarities else 0.0,
        "all": similarities
    }


def compute_quality_metrics(
    generated_texts: List[str],
    reference_texts: List[str] = None,
    compute_ppl: bool = True,
    compute_bert: bool = True,
    compute_mauve_score: bool = False,
    ppl_model: str = "gpt2",
    device: str = "cuda"
) -> Dict[str, Any]:
    """Compute comprehensive quality metrics.
    
    Args:
        generated_texts: Generated/watermarked texts
        reference_texts: Reference texts (for BERTScore, MAUVE)
        compute_ppl: Whether to compute perplexity
        compute_bert: Whether to compute BERTScore
        compute_mauve_score: Whether to compute MAUVE (slower)
        ppl_model: Model for perplexity
        device: Device to run on
        
    Returns:
        Dictionary with all computed metrics
        
    Example:
        >>> metrics = compute_quality_metrics(generated, reference)
        >>> print(f"PPL: {metrics['perplexity']['mean']:.2f}")
        >>> print(f"BERTScore F1: {metrics['bertscore']['f1']:.3f}")
    """
    results = {}
    
    if compute_ppl:
        results["perplexity"] = compute_perplexity(
            generated_texts, 
            model_name=ppl_model,
            device=device
        )
    
    if compute_bert and reference_texts is not None:
        results["bertscore"] = compute_bertscore(
            generated_texts,
            reference_texts,
            device=device
        )
    
    if compute_mauve_score and reference_texts is not None:
        results["mauve"] = compute_mauve(
            generated_texts,
            reference_texts
        )
    
    results["diversity"] = compute_diversity(generated_texts)
    
    return results
