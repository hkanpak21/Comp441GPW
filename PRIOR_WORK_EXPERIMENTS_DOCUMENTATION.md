# LLM Watermarking Experiments Documentation

This document provides comprehensive details on how to replicate experiments from 4 seminal LLM watermarking papers using the `watermark_experiments` framework.

## Table of Contents

1. [Overview](#overview)
2. [Paper Methodologies](#paper-methodologies)
3. [Experimental Setup](#experimental-setup)
4. [Hyperparameters](#hyperparameters)
5. [Datasets](#datasets)
6. [Attack Scenarios](#attack-scenarios)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Expected Results](#expected-results)
9. [Running Experiments](#running-experiments)

---

## Overview

Our `watermark_experiments` framework implements three watermarking methods from the literature:

| Method | Paper | Key Mechanism | Level |
|--------|-------|---------------|-------|
| **Unigram** | Zhao et al., 2023 | Fixed green/red list | Token |
| **KGW** | Kirchenbauer et al., 2023 | Context-dependent green list | Token |
| **SEMSTAMP** | Hou et al., 2024 | LSH semantic partitioning | Sentence |

Plus a unified evaluation framework inspired by WaterPark (Liang et al., 2025).

---

## Paper Methodologies

### 1. Unigram-Watermark (Zhao et al., 2023)

**Key Idea**: Use a fixed (context-free) green list determined only by secret key.

```
Algorithm: Unigram Generation
1. Generate fixed green list G using hash(secret_key)
2. For each token generation:
   - Get logits from LLM
   - Add δ to logits of all tokens in G
   - Sample from modified distribution
   
Algorithm: Unigram Detection
1. Tokenize input text
2. Count tokens t ∈ G (green tokens)
3. Compute z-score: z = (|G| - γT) / √(Tγ(1-γ))
4. Watermarked if z > z_threshold
```

**Theoretical Guarantee**: Provably 2× more robust to random edits than KGW.

---

### 2. KGW/TGRL Watermark (Kirchenbauer et al., 2023)

**Key Idea**: Green list depends on previous token(s) via hash function.

```
Algorithm: KGW Generation
1. For each token at position t:
   - Seed RNG with hash(key, token[t-1])
   - Generate random permutation of vocabulary
   - First γ|V| tokens = green list for position t
   - Add δ to green token logits (soft) or block red (hard)
   - Sample from modified distribution

Algorithm: KGW Detection
1. For each position t (starting at position 2):
   - Recompute green list using token[t-1]
   - Check if token[t] is in green list
2. Count green tokens and compute z-score
```

**Variants**:
- `simple_1`: Use only previous 1 token (bigram)
- `lefthash`: Use h previous tokens (h-gram)

---

### 3. SEMSTAMP (Hou et al., 2024)

**Key Idea**: Watermark at sentence level using semantic embeddings + LSH.

```
Algorithm: SEMSTAMP Generation
1. Initialize LSH hyperplanes and target signature
2. For each sentence s_t:
   a. Compute valid signature based on previous sentence
   b. Rejection loop:
      - Generate candidate sentence
      - Encode with sentence encoder → embedding e
      - Compute LSH signature: sig(e) = [sign(e · n_i)]
      - Check margin constraint: |e · n_i| > m for all i
      - Accept if sig(e) = valid_signature AND margin passed
   c. If max_rejections reached, use last candidate
   
Algorithm: SEMSTAMP Detection
1. Split text into sentences
2. For each consecutive pair (s_{t-1}, s_t):
   - Compute expected signature from s_{t-1}
   - Check if s_t matches expected signature
3. Compute z-score on valid transition count
```

**Key Components**:
- Paraphrase-robust sentence encoder (fine-tuned SBERT)
- LSH with d=3 hyperplanes (8 partitions)
- Margin constraint m=0.02 for robustness

---

### 4. WaterPark Framework (Liang et al., 2025)

**Key Contribution**: Unified evaluation platform for comparing watermarkers.

**Design Space Mapping**:
- Context dependency: h=0 (unigram) vs h>0 (n-gram)
- Generation: Soft vs Hard (rejection sampling)
- Detection: Z-test vs Log-likelihood ratio

---

## Experimental Setup

### Models Used

| Paper | Primary Models | Notes |
|-------|---------------|-------|
| Zhao et al. | GPT2-XL, OPT-1.3B, LLaMA-7B | Focus on robustness |
| Kirchenbauer et al. | OPT-1.3B, OPT-2.7B, OPT-6.7B | Varying model sizes |
| Hou et al. | OPT-1.3B | + Fine-tuned sentence encoder |
| Liang et al. | Multiple (Qwen, LLaMA, OPT) | Comprehensive comparison |

**Our Implementation**: Uses `Qwen2.5-14B-Instruct` by default for state-of-the-art results. Comments indicate original models for replication.

### GPU Requirements

| Task | VRAM Required |
|------|---------------|
| OPT-1.3B inference | 4GB |
| LLaMA-7B inference | 16GB |
| Qwen2.5-14B inference | 32GB |
| SEMSTAMP encoder | 2GB |
| DIPPER paraphraser | 24GB |

---

## Hyperparameters

### Watermark Parameters

```python
# Unigram-Watermark (Zhao et al.)
UNIGRAM_CONFIG = {
    "gamma": 0.5,        # Green list ratio
    "delta": 2.0,        # Logit bias
    "z_threshold": 4.0,  # Detection threshold
    "use_unique_detector": False,  # Standard vs Unique detector
}

# KGW Watermark (Kirchenbauer et al.)
KGW_CONFIG = {
    "gamma": 0.5,
    "delta": 2.0,
    "z_threshold": 4.0,
    "context_width": 1,  # h=1 (bigram seeding)
    "seeding_scheme": "simple_1",
    "ignore_repeated_bigrams": True,
    "hard_watermark": False,  # Soft watermark
}

# SEMSTAMP (Hou et al.)
SEMSTAMP_CONFIG = {
    "lsh_dim": 3,         # d=3 hyperplanes (8 partitions)
    "margin": 0.02,       # Margin constraint
    "max_rejections": 100,
    "z_threshold": 4.0,
    "embedder": "AbeHou/SemStamp-c4-sbert",  # Fine-tuned encoder
}
```

### Generation Parameters

```python
GENERATION_CONFIG = {
    "max_new_tokens": 200,
    "temperature": 1.0,
    "top_p": 0.9,       # Nucleus sampling
    "top_k": 0,         # Disabled by default
    "do_sample": True,
}
```

---

## Datasets

### Primary Datasets

| Dataset | Paper Usage | Samples | Description |
|---------|------------|---------|-------------|
| **C4 RealNewsLike** | KGW, SEMSTAMP, WaterPark | 500+ | News-style web text |
| **OpenGen** | Unigram | 500+ | Human-written baseline |
| **LFQA/ELI5** | Unigram | 500+ | Question-answering |
| **RealNews** | SEMSTAMP | 500+ | News articles |
| **BookSum** | SEMSTAMP | 500+ | Book summaries |

### Loading Datasets

```python
from watermark_experiments.datasets import load_c4, load_lfqa

# Load C4 RealNewsLike
c4_data = load_c4(num_samples=500, subset="realnewslike")

# Load LFQA for QA experiments
lfqa_data = load_lfqa(num_samples=500)
```

---

## Attack Scenarios

### Lexical Attacks

| Attack | Parameters | Expected Effect |
|--------|------------|-----------------|
| **Synonym** | edit_rate=0.3 | Replace 30% words with synonyms |
| **Swap** | edit_rate=0.2 | Swap 10% adjacent word pairs |
| **Typo** | edit_rate=0.3 | Add typos to 30% words |

### Paraphrase Attacks

| Attack | Model | Parameters |
|--------|-------|------------|
| **DIPPER** | kalpeshk2011/dipper-paraphraser-xxl | lex_div=40, order_div=20 |
| **PEGASUS** | tuner007/pegasus_paraphrase | num_beams=5 |
| **Parrot** | prithivida/parrot_paraphraser_on_T5 | - |
| **GPT** | gpt-3.5-turbo | temperature=1.0 |
| **Bigram** | PEGASUS/GPT | Minimize bigram overlap |

### Text Mixing

| Attack | Configuration | Description |
|--------|--------------|-------------|
| **CP-3-50** | n_segments=3, ratio=0.5 | 3 segments, 50% watermarked |
| **CP-5-30** | n_segments=5, ratio=0.3 | 5 segments, 30% watermarked |

---

## Evaluation Metrics

### Detection Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Z-score** | (G - γT) / sqrt(Tγ(1-γ)) | Higher = more watermarked |
| **TPR** | TP / (TP + FN) | True positive rate |
| **FPR** | FP / (FP + TN) | False positive rate |
| **AUC-ROC** | Area under ROC | 1.0 = perfect separation |
| **TPR@FPR=1%** | TPR at 1% FPR | Primary WaterPark metric |

### Quality Metrics

| Metric | Tool | Interpretation |
|--------|------|----------------|
| **Perplexity** | GPT-2 | Lower = higher quality |
| **BERTScore** | DeBERTa | Higher = more similar |
| **MAUVE** | Neural | Higher = more human-like |
| **Distinct-n** | n-gram count | Higher = more diverse |
| **P-SP** | SBERT cosine | Higher = meaning preserved |

---

## Expected Results

### Detection Accuracy (No Attack)

| Method | TPR@FPR=1% | AUC | Source |
|--------|------------|-----|--------|
| **Unigram** | >0.99 | >0.99 | Table 4, Zhao et al. |
| **KGW** | >0.98 | >0.99 | Table 2, Kirchenbauer et al. |
| **SEMSTAMP** | ~0.95 | >0.99 | Table 1, Hou et al. |

### After DIPPER Attack

| Method | TPR@FPR=1% | Notes |
|--------|------------|-------|
| **Unigram** | 0.73-0.92 | 2x more robust than KGW |
| **KGW** | 0.39-0.74 | Drops significantly |
| **SEMSTAMP** | ~0.85 | Semantic robustness |

### Text Quality (Perplexity)

| Method | PPL (OPT-1.3B) | Impact |
|--------|---------------|--------|
| No watermark | ~15-20 | Baseline |
| **Unigram** | ~16-22 | +5-10% |
| **KGW (soft)** | ~16-22 | +5-10% |
| **SEMSTAMP** | ~18-25 | +10-20% (sentence-level) |

---

## Running Experiments

### Quick Start

```python
from watermark_experiments import UnigramWatermark, KGWWatermark
from watermark_experiments.metrics import compute_detection_metrics
from watermark_experiments.datasets import load_c4
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model (can use OPT-1.3B for paper replication)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

# Initialize watermarkers
unigram = UnigramWatermark(model, tokenizer)
kgw = KGWWatermark(model, tokenizer)

# Load data
data = load_c4(num_samples=100)
prompts = [d["prompt"] for d in data]
human_texts = [d["text"] for d in data[:50]]

# Generate watermarked texts
wm_texts_unigram = [unigram.generate(p) for p in prompts[:50]]
wm_texts_kgw = [kgw.generate(p) for p in prompts[:50]]

# Detect
wm_scores = [unigram.detect(t).z_score for t in wm_texts_unigram]
human_scores = [unigram.detect(t).z_score for t in human_texts]

# Evaluate
metrics = compute_detection_metrics(wm_scores, human_scores)
print(f"AUC: {metrics['auc']:.3f}")
print(f"TPR@FPR=1%: {metrics['tpr_at_fpr_1']:.3f}")
```

### Attack Robustness Experiment

```python
from watermark_experiments.attacks import DIPPERAttack, SynonymAttack

# Apply attacks
dipper = DIPPERAttack(lexical_diversity=40, order_diversity=20)
synonym = SynonymAttack(edit_rate=0.3)

attacked_dipper = [dipper(t) for t in wm_texts_unigram]
attacked_synonym = [synonym(t) for t in wm_texts_unigram]

# Measure detection after attack
wm_scores_dipper = [unigram.detect(t).z_score for t in attacked_dipper]
wm_scores_synonym = [unigram.detect(t).z_score for t in attacked_synonym]

# Compare
print(f"Before attack TPR@1%: {metrics['tpr_at_fpr_1']:.3f}")
metrics_dipper = compute_detection_metrics(wm_scores_dipper, human_scores)
print(f"After DIPPER TPR@1%: {metrics_dipper['tpr_at_fpr_1']:.3f}")
```

### Quality Evaluation

```python
from watermark_experiments.metrics import compute_quality_metrics

quality = compute_quality_metrics(
    generated_texts=wm_texts_unigram,
    reference_texts=human_texts,
    compute_ppl=True,
    compute_bert=True,
    compute_mauve_score=True
)

print(f"Perplexity: {quality['perplexity']['mean']:.2f}")
print(f"BERTScore F1: {quality['bertscore']['f1']:.3f}")
print(f"MAUVE: {quality['mauve']:.3f}")
```

---

## References

1. Zhao et al., "Provable Robust Watermarking for AI-Generated Text", ICLR 2024
2. Kirchenbauer et al., "A Watermark for Large Language Models", ICML 2023
3. Hou et al., "SemStamp: A Semantic Watermark with Paraphrastic Robustness", NAACL 2024
4. Liang et al., "Watermark under Fire: A Robustness Evaluation of LLM Watermarking", ACL 2025

## GitHub Repositories

- KGW: https://github.com/jwkirchenbauer/lm-watermarking
- Unigram: https://github.com/XuandongZhao/Unigram-Watermark
- SEMSTAMP: https://github.com/abehou/SemStamp
