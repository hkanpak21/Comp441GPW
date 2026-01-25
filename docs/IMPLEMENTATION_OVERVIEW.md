# GPW Watermarking: Complete Experimental Implementation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Research Objectives](#research-objectives)
3. [Watermarking Methods](#watermarking-methods)
4. [Attack Methodology](#attack-methodology)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Experimental Setup](#experimental-setup)
7. [Implementation Details](#implementation-details)
8. [Results Summary](#results-summary)

---

## Project Overview

This project implements and evaluates **Gaussian Pancakes Watermarking (GPW)** and its variants for detecting AI-generated text. We compare GPW against baseline watermarking methods (Unigram, KGW, SEMSTAMP) across multiple language models and attack scenarios.

### Project Structure

```
Comp441GPW/
├── watermarkers/               # Watermarking implementations
│   ├── __init__.py            # Exports: UnigramWatermark, KGWWatermark
│   ├── base.py                # Base classes: TokenLevelWatermarker, DetectionResult
│   ├── gpw.py                 # GPW family: GPW, GPW-SP, GPW-SP+SR
│   ├── semstamp.py            # SEMSTAMP watermarker (semantic-based)
│   ├── unigram.py             # Unigram baseline watermarker
│   └── kgw.py                 # KGW baseline watermarker
│
├── attacks/                    # Text attack implementations
│   ├── __init__.py            # Exports attack classes
│   ├── base.py                # BaseAttack class with AttackResult
│   ├── synonym.py             # SynonymAttack (word replacement)
│   ├── swap.py                # SwapAttack (word reordering)
│   ├── typo.py                # TypoAttack (character-level noise)
│   ├── copypaste.py           # CopyPasteAttack (human text insertion)
│   └── paraphrase.py          # ParaphraseAttack (LLM rewriting)
│
├── data_loaders/              # Dataset loading utilities
│   ├── __init__.py
│   └── c4_loader.py           # C4 RealNewsLike dataset loader
│
├── experiment_scripts/         # Experiment runners
│   ├── generate_texts.py      # Text generation with/without watermarks
│   ├── run_attacks_on_texts.py# Attack and detection pipeline
│   ├── gpw_ablation_study.py  # GPW hyperparameter ablation
│   └── submit_*.sh            # SLURM job scripts for cluster
│
├── results/                    # Experiment results
│   ├── unified_results.csv    # All results combined
│   └── gpw_ablation_gpt2.csv  # GPW ablation study results
│
├── generated_texts/            # Generated text pickles (.pkl files)
├── logs/                       # SLURM job logs
└── docs/                       # Documentation
```

---

## Research Objectives

### Primary Goals

1. **Evaluate GPW watermarking robustness** against various text attacks
2. **Compare GPW variants**: GPW vs GPW-SP vs GPW-SP+SR
3. **Benchmark against baselines**: Unigram, KGW, SEMSTAMP
4. **Ablation study**: Find optimal hyperparameters (alpha, omega)
5. **Cross-model evaluation**: Test on OPT-1.3B, GPT-2, Qwen-7B, Pythia family

### Research Questions

- How does omega (cosine frequency) affect attack robustness?
- Does salted phase (GPW-SP) improve or hurt robustness?
- Why does SR coupling (GPW-SP+SR) fail under attacks?
- Which watermarking method is most robust to copy-paste attacks?

---

## Watermarking Methods

### 1. GPW (Gaussian Pancakes Watermarking)

**Core Idea**: Embed watermark by biasing token selection based on cosine score.

```python
# Generation
score[token] = cos(ω * s[token])        # s[token] = E[token] @ w
logits_wm = logits + α * score          # Apply bias
next_token = sample(logits_wm)

# Detection
S = Σ cos(ω * s[token_t])               # Sum scores for all tokens
z_score = S * sqrt(2/n)                 # Normalize
is_watermarked = (z_score > threshold)
```

**Key Components**:
- `w`: Secret direction vector (derived from hash key)
- `E`: Token embedding matrix [vocab_size × embedding_dim]
- `s = E @ w`: Projection scores for each token
- `α` (alpha): Logit bias strength (default: 3.0)
- `ω` (omega): Cosine frequency (default: 50.0)

**Why Higher Omega = More Robust**:
- Higher ω creates faster oscillation in cos(ω*s)
- More tokens get positive scores (are "green")
- Signal spreads across more tokens
- When attacks modify some tokens, remaining tokens still carry signal

### 2. GPW-SP (Salted Phase)

**Enhancement**: Add context-dependent phase shift.

```python
# Phase computation
context = hash(last_4_tokens)
φ = 2π * PRF(key, context)              # Pseudo-random phase

# Generation
score[token] = cos(ω * s[token] + φ)    # Phase-shifted cosine
```

**Purpose**:
- Makes watermark depend on local context
- Different parts of text have different phases
- Theoretically harder to reverse-engineer

**Reality**:
- Actually HURTS robustness (81% vs 100% clean detection)
- Context dependency means attacks can disrupt phase alignment

### 3. GPW-SP+SR (Semantic Representation Coupling)

**Enhancement**: Make direction w depend on hidden states.

```python
# At each position t
h_t = model(prefix).hidden_states[-1]   # Get hidden state
w_t = w + λ * A @ h_t                   # Position-dependent direction
w_t = normalize(w_t)

# Generation
s_t = E @ w_t                           # Recompute projections
score[token] = cos(ω * s_t[token] + φ)
```

**Coupling Matrix A**:
```python
A = B @ C  # Low-rank: [d_embed × rank] @ [rank × d_hidden]
# rank=16 (default), λ=0.1 (coupling strength)
```

**Why SR Fails Under Attacks**:
- Hidden states `h_t` capture semantic meaning
- When text is modified (synonyms, swaps), hidden states change
- Detection computes different `w_t` than generation
- Watermark becomes undetectable

### 4. Unigram Watermark (Baseline)

**Method**: Simple green/red token partitioning.

```python
# Split vocabulary
green_tokens = top γ% tokens by hash(token)
red_tokens = remaining (1-γ)% tokens

# Generation
logits[green_tokens] += δ               # Boost green tokens

# Detection
green_count = count(tokens in green_tokens)
z_score = (green_count - n*γ) / sqrt(n*γ*(1-γ))
```

**Parameters**: γ=0.5, δ=2.0

### 5. KGW Watermark (Baseline)

**Method**: Context-dependent green list (Kirchenbauer et al.).

```python
# For each position
prev_token = tokens[t-1]
seed = hash(key, prev_token)
green_list = random_subset(vocab, size=γ*|V|, seed=seed)

# Generation
logits[green_list] += δ

# Detection
green_count = Σ (token_t in green_list(token_{t-1}))
```

**Parameters**: γ=0.5, δ=2.0, seeding_scheme="simple_1"

### 6. SEMSTAMP (Baseline)

**Method**: Semantic embedding-based watermarking using LSH.

```python
# Generation
embedding = sentence_transformer(generated_prefix)
lsh_hash = locality_sensitive_hash(embedding)
bias_tokens = tokens_matching_hash_pattern(lsh_hash)
logits[bias_tokens] += δ

# Detection
embedding = sentence_transformer(text)
expected_hash = lsh(embedding)
actual_pattern = extract_pattern(tokens)
score = match(expected_hash, actual_pattern)
```

**Status**: BROKEN in our experiments (3-8% detection = near random)

---

## Attack Methodology

### Attack Types

| Attack | Description | Edit Rate | Implementation |
|--------|-------------|-----------|----------------|
| **Clean** | No modification | 0% | Baseline measurement |
| **Synonym** | Replace words with synonyms | 30% | WordNet + word2vec similarity |
| **Swap** | Randomly swap adjacent words | 20% | Pairwise word swapping |
| **Typo** | Introduce character-level typos | 10% | Insert/delete/substitute chars |
| **CopyPaste** | Insert human-written text | 50% | Splice C4 human text |
| **Paraphrase** | Rewrite using LLM | 100% | GPT-3.5/Claude API |

### Attack Implementation Details

#### Synonym Attack (30% edit rate)
```python
class SynonymAttack:
    def attack(self, text):
        words = tokenize(text)
        n_edit = int(len(words) * 0.3)
        for _ in range(n_edit):
            word = random_choice(words)
            synonyms = wordnet.synsets(word)
            if synonyms:
                replacement = most_similar(synonyms, word)
                replace(word, replacement)
        return detokenize(words)
```

#### Swap Attack (20% edit rate)
```python
class SwapAttack:
    def attack(self, text):
        words = tokenize(text)
        n_swaps = int(len(words) * 0.2)
        for _ in range(n_swaps):
            i = random_index(len(words) - 1)
            words[i], words[i+1] = words[i+1], words[i]
        return detokenize(words)
```

#### Typo Attack (10% edit rate)
```python
class TypoAttack:
    def attack(self, text):
        chars = list(text)
        n_typos = int(len(chars) * 0.1)
        for _ in range(n_typos):
            i = random_index(len(chars))
            op = random_choice(['insert', 'delete', 'substitute'])
            apply_operation(chars, i, op)
        return ''.join(chars)
```

#### CopyPaste Attack (50% substitution)
```python
class CopyPasteAttack:
    def attack(self, watermarked_text, human_text):
        wm_sentences = split_sentences(watermarked_text)
        human_sentences = split_sentences(human_text)
        n_replace = int(len(wm_sentences) * 0.5)

        indices = random_sample(range(len(wm_sentences)), n_replace)
        for i, idx in enumerate(indices):
            wm_sentences[idx] = human_sentences[i % len(human_sentences)]

        return join_sentences(wm_sentences)
```

---

## Evaluation Metrics

### Primary Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Detection Rate** | `detected / total * 100` | % of watermarked texts correctly identified |
| **Z-Score** | `S * sqrt(2/n)` | Normalized detection score |
| **False Positive Rate** | `false_positives / human_texts` | Human texts incorrectly flagged |
| **AUC** | Area under ROC curve | Overall detection quality |
| **TPR@FPR=1%** | True positive rate at 1% FPR | Detection at low false positive |

### Quality Metrics

| Metric | Description |
|--------|-------------|
| **Perplexity** | Language model perplexity (lower = more fluent) |
| **Semantic Similarity** | Cosine similarity of embeddings before/after attack |

### Detection Threshold

```python
z_threshold = 4.0  # Default for all methods

# Statistical interpretation:
# z=4.0 corresponds to p-value ≈ 3.2e-5 (one-tailed)
# Approximately 1 in 31,000 false positive rate
```

---

## Experimental Setup

### Models Evaluated

| Model | Size | Parameters | Source |
|-------|------|------------|--------|
| GPT-2 | 124M | 124M | OpenAI |
| OPT-1.3B | 1.3B | 1.3B | Meta |
| Qwen-7B | 7B | 7B | Alibaba |
| Pythia-400M | 400M | 400M | EleutherAI |
| Pythia-1.4B | 1.4B | 1.4B | EleutherAI |
| Pythia-2.8B | 2.8B | 2.8B | EleutherAI |
| Pythia-6.9B | 6.9B | 6.9B | EleutherAI |
| Pythia-12B | 12B | 12B | EleutherAI |

### Dataset

**C4 RealNewsLike** (validation split)
- Source: Common Crawl filtered for news-like content
- Prompt extraction: First 50 characters of text (100-1000 char filter)
- Human text: Full text used for copy-paste attacks

```python
def load_c4(num_samples=200):
    dataset = load_dataset("c4", "realnewslike", split="validation")
    samples = []
    for item in dataset:
        text = item['text']
        if 100 <= len(text) <= 1000:
            samples.append({
                'prompt': text[:50],
                'text': text
            })
            if len(samples) >= num_samples:
                break
    return samples
```

### Generation Parameters

```python
# Common generation settings
max_new_tokens = 200
temperature = 1.0
top_k = 50
top_p = 0.95
```

### Cluster Configuration

```bash
# SLURM job settings
#SBATCH --partition=t4_ai
#SBATCH --gres=gpu:1
#SBATCH --mem=32G-48G
#SBATCH --time=0:30:00-1:30:00
```

---

## Implementation Details

### GPW Configuration Classes

```python
@dataclass
class GPWConfig:
    """GPW watermarking configuration."""
    alpha: float = 3.0      # Logit bias strength
    omega: float = 50.0     # Cosine frequency (higher = more robust)
    salted: bool = True     # Enable context-keyed phase (GPW-SP)
    ctx_mode: str = "ngram" # Context mode: "prev_token" | "ngram" | "rolling"
    ngram: int = 4          # n-gram size for phase computation

@dataclass
class SRConfig:
    """Semantic Representation coupling configuration."""
    enabled: bool = False       # Enable SR coupling
    lambda_couple: float = 0.09 # Coupling strength
    rank: int = 32              # Low-rank approximation dimension
```

### Variant Creation

```python
# GPW (non-salted, static direction)
gpw = create_gpw_variant(model, tokenizer, variant="GPW")

# GPW-SP (salted phase)
gpw_sp = create_gpw_variant(model, tokenizer, variant="GPW-SP")

# GPW-SP+SR (semantic representation coupling)
gpw_sp_sr = create_gpw_variant(model, tokenizer, variant="GPW-SP+SR")
```

### Key Algorithms

#### Secret Direction Derivation
```python
def derive_secret_direction_w(key: bytes, d: int, device: str):
    """Derive deterministic secret direction from key."""
    seed = hash(key + b"GPW_w")
    generator = torch.Generator().manual_seed(seed)
    v = torch.randn(d, generator=generator)
    return v / v.norm()  # Normalize to unit vector
```

#### Projection Computation
```python
def precompute_projections(E: torch.Tensor, w: torch.Tensor):
    """Compute normalized projections s = E @ w."""
    s = E @ w
    # Normalize to zero mean, unit variance
    s = (s - s.mean()) / s.std()
    return s
```

#### Phase Computation (GPW-SP)
```python
def salted_phase_phi(key: bytes, input_ids: torch.Tensor, cfg: GPWConfig):
    """Compute context-dependent phase."""
    # Get last n tokens as context
    context = input_ids[0, -cfg.ngram:].tolist()
    fingerprint = hash(key + str(context).encode())
    # Map to [0, 2π)
    u = (fingerprint % 2**53) / 2**53
    return 2.0 * math.pi * u
```

#### SR Coupling
```python
def compute_w_t(w, A, h_t, lambda_couple):
    """Compute position-dependent direction."""
    # w_t = w + λ * A @ h_t
    v = w + lambda_couple * (A @ h_t)
    return v / v.norm()  # Normalize
```

### Detection Optimization

**Fast Path (Non-SR)**:
```python
# No forward passes needed - O(n) token lookups
for t in range(1, len(tokens)):
    phi = compute_phase(context[:t])  # Fast hash
    score = cos(omega * s_base[token[t]] + phi)  # Lookup
    S += score
```

**Slow Path (SR)**:
```python
# Requires forward pass per position - O(n) model calls
for t in range(1, len(tokens)):
    h_t = model(tokens[:t]).hidden_states[-1]  # Forward pass
    w_t = compute_w_t(w, A, h_t, lambda_couple)
    s_t = E @ w_t  # Recompute projections
    score = cos(omega * s_t[token[t]] + phi)
    S += score
```

---

## Results Summary

### Ablation Study (GPT-2, 50 samples)

#### Omega Ablation (Cosine Frequency)

| Omega | Clean Det | Synonym | Swap | Typo | Avg |
|-------|-----------|---------|------|------|-----|
| 1.0 | 96% | 98% | 96% | 96% | 96.5% |
| 5.0 | 98% | 98% | 98% | 98% | 98.0% |
| **10.0** | **100%** | **100%** | **100%** | **100%** | **100%** |
| 25.0 | 100% | 100% | 100% | 100% | 100% |
| 50.0 | 100% | 100% | 100% | 100% | 100% |
| 100.0 | 98% | 98% | 98% | 98% | 98% |

**Finding**: Optimal omega = 10-50. Too high (100) starts to hurt.

#### Alpha Ablation (Logit Bias Strength)

| Alpha | Clean Det | Synonym | Swap | Typo | Avg |
|-------|-----------|---------|------|------|-----|
| **1.0** | **60%** | **48%** | **62%** | **58%** | **57%** |
| 2.0 | 100% | 100% | 100% | 100% | 100% |
| 3.0 | 100% | 100% | 100% | 100% | 100% |
| 5.0 | 100% | 98% | 98% | 98% | 98.5% |
| 10.0 | 100% | 100% | 100% | 100% | 100% |

**Finding**: Alpha=1 is too weak. Alpha≥2 required for reliable detection.

#### Mode Ablation (GPW Variants)

| Mode | Clean | Synonym | Swap | Typo | Avg |
|------|-------|---------|------|------|-----|
| **GPW** | **100%** | **100%** | **100%** | **100%** | **100%** |
| GPW-SP | 94% | 92% | 94% | 94% | 93.5% |
| GPW-SP+SR | 100% | **28%** | **40%** | **54%** | **55.5%** |

**Finding**: GPW-SP+SR breaks under attacks. Use GPW for robustness.

### Cross-Model Results (OPT-1.3B, 200 samples)

| Method | Clean | Synonym | Swap | Typo | CopyPaste |
|--------|-------|---------|------|------|-----------|
| **GPW** | **100%** | **99%** | **100%** | **99.5%** | **73%** |
| GPW-SP | 81% | 64.5% | 67% | 74.5% | 38% |
| GPW-SP+SR | 94% | 0% | 0% | 2% | 8% |
| Unigram | 96.5% | 92% | 95% | 93.5% | 23.5% |
| KGW | 94.5% | 75% | 76.5% | 86% | 8.5% |
| SEMSTAMP | 3% | 1% | 2% | 3% | 1% |

### Key Findings

1. **GPW (non-salted) is most robust**: Achieves near-perfect detection across all attacks
2. **Copy-paste is the hardest attack**: All methods struggle (8-73% detection)
3. **GPW excels at copy-paste**: 73% vs 23.5% (Unigram) vs 8.5% (KGW)
4. **SEMSTAMP is broken**: 1-3% detection = near random (implementation issue)
5. **SR coupling defeats robustness**: 100% clean but 0-54% under attacks

### Pythia Scaling Results

| Model | GPW (AUC) | KGW (AUC) | Unigram (AUC) |
|-------|-----------|-----------|---------------|
| Pythia-400M | 99.75% | 98.25% | 100% |
| Pythia-1.4B | 99.25% | 100% | 99.5% |
| Pythia-2.8B | 93.25% | 100% | 98.63% |
| Pythia-6.9B | 98.5% | 100% | 98.75% |
| Pythia-12B | 100% | 99.75% | 93.88% |

**Finding**: All methods scale well across model sizes.

---

## Conclusion

### Recommended Configuration

```python
# Best overall robustness
gpw_cfg = GPWConfig(
    alpha=3.0,      # Moderate bias
    omega=50.0,     # High frequency for robustness
    salted=False,   # No salted phase (GPW, not GPW-SP)
)
sr_cfg = SRConfig(enabled=False)  # No SR coupling
```

### Summary

| Aspect | Recommendation |
|--------|----------------|
| **Best method** | GPW (non-salted) |
| **Best omega** | 10-50 |
| **Best alpha** | 2-5 |
| **Avoid** | GPW-SP+SR (fragile under attacks) |
| **Hardest attack** | Copy-paste (requires 50%+ insertion) |
| **Most robust to** | Synonym, Swap, Typo attacks |
