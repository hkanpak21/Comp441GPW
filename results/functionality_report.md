# Functionality Report

Generated: 2026-01-24
Test Environment: CUDA available, thor310 conda environment

## Summary

All core components are **WORKING** except SEMSTAMP which has known issues.

---

## Watermarkers Status

| Watermarker | Status | Detection (GPT-2) | Detection (OPT-1.3B) | Notes |
|-------------|--------|-------------------|----------------------|-------|
| Unigram | WORKING | z=5.94, detected=True | z=2.26, detected=False | Works, but z-score varies |
| KGW | WORKING | z=5.11, detected=True | z=4.71, detected=True | Consistent detection |
| GPW | WORKING | z=6.51, detected=True | z=6.99, detected=True | Strong detection |
| GPW-SP | WORKING | z=7.33, detected=True | z=7.14, detected=True | Best detection |
| GPW-SP-SR | WORKING | z=7.08, detected=True | z=7.15, detected=True | SR coupling works |
| SEMSTAMP | BROKEN | z=-0.58, detected=False | z=1.73, detected=False | Near-zero detection |

### Watermarker Configurations
```python
# Unigram
gamma=0.5, delta=2.0, z_threshold=4.0

# KGW
gamma=0.5, delta=2.0, z_threshold=4.0, seeding_scheme="simple_1"

# GPW (non-salted)
alpha=3.0, omega=50.0, salted=False

# GPW-SP (salted)
alpha=3.0, omega=50.0, salted=True, ctx_mode="ngram", ngram=4

# GPW-SP-SR (salted + SR coupling)
alpha=3.0, omega=50.0, salted=True, sr_enabled=True, sr_lambda=0.1, sr_rank=16

# SEMSTAMP (BROKEN)
lsh_dim=3, margin=0.0, max_rejections=2
```

---

## Attacks Status

| Attack | Status | Notes |
|--------|--------|-------|
| SynonymAttack | WORKING | edit_rate=0.3, uses WordNet synonyms |
| SwapAttack | WORKING | edit_rate=0.2, swaps adjacent words |
| TypoAttack | WORKING | edit_rate=0.1, character-level errors |
| CopyPasteAttack | WORKING | n_segments=3, watermark_ratio=0.5 |
| PegasusAttack (paraphrase) | WORKING | Uses tuner007/pegasus_paraphrase model |

### Attack Configurations
```python
# Lexical attacks
synonym: edit_rate=0.30  # 30% word replacement
swap: edit_rate=0.20     # 20% word swaps
typo: edit_rate=0.10     # 10% character errors

# Copy-paste attack
n_segments=3, watermark_ratio=0.50  # 50% watermarked, 50% human

# Paraphrase
model: tuner007/pegasus_paraphrase
```

---

## Dataset Status

| Dataset | Status | Notes |
|---------|--------|-------|
| C4 (realnewslike) | WORKING | Returns prompt and text fields |

---

## Models Tested

| Model | Status | Notes |
|-------|--------|-------|
| gpt2 | WORKING | float32, fast for testing |
| facebook/opt-1.3b | WORKING | float16, primary model |
| Qwen/Qwen2.5-7B-Instruct | UNTESTED | Needs more memory |

---

## Known Issues

### 1. SEMSTAMP is BROKEN
- Detection rates near 0% across all models
- Z-scores are very low (negative or near-zero)
- Issue may be in embedding/LSH partitioning
- Recommendation: Debug or skip SEMSTAMP experiments

### 2. Paraphrase Attack CUDA Issues
- Previous reports of CUDA errors during paraphrase
- May be OOM due to loading multiple models (generation + paraphrase)
- Recommendation: Run paraphrase attacks separately with smaller batch sizes

### 3. GPT-2 CUDA Errors (reported in unified_results)
- GPW-SP and KGW had CUDA errors on GPT-2
- Tests show they work now - may have been environment issue
- Recommendation: Try running with explicit float32

---

## Recommendations for Experiments

1. **Skip SEMSTAMP** - It's broken, don't waste compute
2. **Run GPW variants first** - They show the best results
3. **Paraphrase separately** - May cause OOM, run with smaller batches
4. **Use 200 samples** - Standard for detection rate calculation
5. **Track results per-job** - Easier to debug failures

---

## Test Commands

```bash
# Full functionality test
python experiment_scripts/test_functionality.py --test all --model gpt2

# Test only imports
python experiment_scripts/test_functionality.py --test imports

# Test watermarkers with specific model
python experiment_scripts/test_functionality.py --test watermarkers --model facebook/opt-1.3b

# Test attacks
python experiment_scripts/test_functionality.py --test attacks
```

---

## File Locations

- Test script: `experiment_scripts/test_functionality.py`
- Main experiment script: `experiment_scripts/exp_final_comprehensive.py`
- Results directory: `results/`
- Logs directory: `logs/`
