# OPT-1.3B Attack Results Summary

Generated: 2026-01-25

## Watermarked Text Detection (TPR - True Positive Rate)

| Watermarker | clean | synonym_30 | swap_20 | typo_10 | copypaste_50 |
|-------------|-------|------------|---------|---------|--------------|
| **GPW** | **100.0%** (z=13.37) | **100.0%** (z=11.14) | **100.0%** (z=12.69) | **99.5%** (z=11.58) | 71.5% (z=5.40) |
| **GPW-SP** | 81.0% (z=8.98) | 65.5% (z=4.69) | 62.0% (z=4.74) | 74.0% (z=6.13) | 40.5% (z=3.16) |
| **GPW-SP-LOW** | 90.5% (z=9.75) | 73.5% (z=5.31) | 75.0% (z=5.21) | 86.0% (z=6.75) | 39.5% (z=3.18) |
| **Unigram** | 97.5% (z=8.31) | 93.0% (z=6.23) | 96.5% (z=7.58) | 94.5% (z=6.82) | 24.5% (z=2.99) |
| **KGW** | 93.0% (z=7.10) | 84.0% (z=5.27) | 83.0% (z=5.27) | 88.0% (z=5.89) | 11.0% (z=2.44) |

## Baseline (No Watermark) Detection - FPR (False Positive Rate)

| Detector | clean | synonym_30 | swap_20 | typo_10 | copypaste_50 |
|----------|-------|------------|---------|---------|--------------|
| **GPW** | - | 0.0% (z=0.61) | 0.5% (z=0.73) | - | - |
| **GPW-SP** | 0.0% (z=0.13) | - | - | 0.0% (z=0.11) | 0.0% (z=0.13) |
| **KGW** | 0.0% (z=0.00) | 0.0% (z=-0.09) | 0.0% (z=-0.16) | 0.0% (z=-0.13) | - |

## Key Findings

### Best Performers:
1. **GPW (non-salted)** - Best overall:
   - 100% detection on clean, synonym, swap
   - 99.5% on typo attacks
   - 71.5% on copypaste (best among all methods)
   - Very high z-scores (11-13)

2. **Unigram** - Strong baseline:
   - 93-97.5% detection on lexical attacks
   - Weak on copypaste (24.5%)

### Attack Robustness Rankings (clean → copypaste):
1. **GPW**: 100% → 71.5% (best copypaste robustness)
2. **Unigram**: 97.5% → 24.5%
3. **KGW**: 93% → 11% (worst copypaste robustness)
4. **GPW-SP-LOW**: 90.5% → 39.5%
5. **GPW-SP**: 81% → 40.5%

### False Positive Rates:
- All detectors show **~0% FPR** on non-watermarked text
- This means high specificity - very few false accusations

## Missing Results (jobs still running):
- GPW-SP-SR (all attacks)
- Some baseline FPR combinations
