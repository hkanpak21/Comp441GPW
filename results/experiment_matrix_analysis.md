# Comprehensive Experiment Matrix Analysis

Generated: 2026-01-24

## Current State Summary

### Models
| Model | Status | Notes |
|-------|--------|-------|
| OPT-1.3B | Primary | Most experiments complete |
| GPT-2 | Validation | Partial results |
| Qwen 7B | Scaling | No results yet |

### Watermarkers
| Watermarker | Variant | Parameters | Status |
|-------------|---------|------------|--------|
| No Watermark | Control | - | Baseline needed |
| GPW | non-salted | alpha=3.0, omega=50.0 | Working |
| GPW-SP | salted | alpha=3.0, omega=50.0 | Working |
| GPW-SP-LOW | salted low-omega | alpha=3.0, omega=2.0 | Working |
| GPW-SP-SR | salted+SR coupling | alpha=3.0, omega=50.0, sr_lambda=0.1 | Needs testing |
| Unigram | baseline | gamma=0.5, delta=2.0 | Working |
| KGW | baseline | gamma=0.5, delta=2.0 | Working |
| SEMSTAMP | LSH d=3 | lsh_dim=3 | BROKEN (near-zero detection) |

### Attacks
| Attack | Parameters | Status |
|--------|------------|--------|
| clean | - | Working |
| synonym_30 | edit_rate=0.30 | Working |
| swap_20 | edit_rate=0.20 | Working |
| typo_10 | edit_rate=0.10 | Working |
| copypaste_50 | n_segments=3, ratio=0.50 | Working (needs human text) |
| paraphrase | Pegasus | CUDA issues reported |

---

## Experiment Matrix: OPT-1.3B

| Watermarker | clean | synonym | swap | typo | copypaste | paraphrase |
|-------------|:-----:|:-------:|:----:|:----:|:---------:|:----------:|
| No Watermark | ❌ NEED | - | - | - | - | - |
| GPW | ✅ 100% | ✅ 99% | ✅ 100% | ✅ 99.5% | ✅ 73% | ❌ CUDA |
| GPW-SP | ✅ 81% | ✅ 64.5% | ✅ 67% | ✅ 74.5% | ✅ 38% | ❌ NEED |
| GPW-SP-LOW | ✅ 90.5% | ✅ 75% | ✅ 75.5% | ✅ 84.5% | ✅ 36% | ❌ NEED |
| GPW-SP-SR | ❌ running | ❌ running | ❌ running | ❌ running | ❌ running | ❌ NEED |
| Unigram | ✅ 96.5% | ✅ 92% | ✅ 95% | ✅ 93.5% | ✅ 23.5% | ✅ 57.1% |
| KGW | ✅ 94.5% | ✅ 75% | ✅ 76.5% | ✅ 86% | ✅ 8.5% | ✅ 50% |
| SEMSTAMP | ✅ 3% BROKEN | ✅ 1% BROKEN | ✅ 2% BROKEN | ✅ 3% BROKEN | ✅ 1% BROKEN | ❌ NEED |

### OPT-1.3B Missing Experiments:
1. **No Watermark baseline** - Need FPR measurement
2. **GPW paraphrase** - CUDA error, needs retry
3. **GPW-SP paraphrase** - Not run
4. **GPW-SP-LOW paraphrase** - Not run
5. **GPW-SP-SR ALL attacks** - Jobs marked as running but no results
6. **SEMSTAMP paraphrase** - Not run (though SEMSTAMP is broken)

---

## Experiment Matrix: GPT-2

| Watermarker | clean | synonym | swap | typo | copypaste | paraphrase |
|-------------|:-----:|:-------:|:----:|:----:|:---------:|:----------:|
| No Watermark | ❌ NEED | - | - | - | - | - |
| GPW | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED |
| GPW-SP | ❌ CUDA | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED |
| GPW-SP-LOW | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED |
| GPW-SP-SR | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED |
| Unigram | ✅ 99.5% | ✅ 98% | ✅ 99% | ✅ 98% | ✅ 74% | ✅ 66.7% |
| KGW | ❌ CUDA | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED |
| SEMSTAMP | ✅ 8.5% BROKEN | ✅ 7.5% BROKEN | ✅ 7% BROKEN | ✅ 7.5% BROKEN | ✅ 8% BROKEN | ❌ NEED |

### GPT-2 Missing Experiments:
1. **No Watermark baseline** - FPR measurement
2. **GPW all attacks** - Complete suite needed
3. **GPW-SP all attacks** - CUDA error, may need float32
4. **GPW-SP-LOW all attacks** - Complete suite needed
5. **GPW-SP-SR all attacks** - Complete suite needed
6. **KGW all attacks** - CUDA error, may need float32
7. **SEMSTAMP paraphrase** - Not run (though SEMSTAMP is broken)

---

## Experiment Matrix: Qwen 7B

| Watermarker | clean | synonym | swap | typo | copypaste | paraphrase |
|-------------|:-----:|:-------:|:----:|:----:|:---------:|:----------:|
| No Watermark | ❌ NEED | - | - | - | - | - |
| GPW | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED |
| GPW-SP | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED |
| GPW-SP-LOW | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED |
| GPW-SP-SR | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED |
| Unigram | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED |
| KGW | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED |
| SEMSTAMP | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED | ❌ NEED |

### Qwen 7B Missing Experiments:
- **ALL experiments needed** (48 total combinations)

---

## Priority Experiment List

### HIGH PRIORITY (OPT-1.3B gaps)
1. GPW-SP-SR clean/synonym/swap/typo/copypaste (5 experiments)
2. GPW paraphrase retry
3. GPW-SP paraphrase
4. GPW-SP-LOW paraphrase
5. No Watermark baseline

### MEDIUM PRIORITY (GPT-2 gaps)
6. GPW all attacks (6 experiments)
7. GPW-SP all attacks (fix CUDA, 6 experiments)
8. GPW-SP-LOW all attacks (6 experiments)
9. GPW-SP-SR all attacks (6 experiments)
10. KGW all attacks (fix CUDA, 6 experiments)
11. No Watermark baseline

### LOW PRIORITY (Qwen 7B - later)
12. Full Qwen 7B suite (48+ experiments)

### INVESTIGATE
- SEMSTAMP is BROKEN across all models - need to debug

---

## Total Missing Experiments Count

| Model | Missing | Total Possible |
|-------|---------|----------------|
| OPT-1.3B | ~11 | 48 |
| GPT-2 | ~35 | 48 |
| Qwen 7B | 48 | 48 |
| **TOTAL** | **~94** | **144** |

---

## Known Issues

1. **CUDA errors** - GPT-2 may need float32 for some watermarkers
2. **SEMSTAMP broken** - Near-zero detection rates, needs debugging
3. **Paraphrase attack** - Uses Pegasus model, may cause CUDA OOM
4. **CopyPaste attack** - Requires human text for mixing
5. **GPW-SP-SR** - Jobs submitted but results not in unified table

---

## Recommended Experiment Order

1. First, test all components work individually
2. Run OPT-1.3B missing experiments (smallest gap)
3. Fix GPT-2 CUDA issues and run experiments
4. Debug SEMSTAMP if needed
5. Run Qwen 7B experiments (requires more memory)
