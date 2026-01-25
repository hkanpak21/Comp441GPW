# 🚀 Quick Start Guide - Watermarking Experiments

**Status:** ✅ All systems operational, ready for experiments!  
**Date:** January 21, 2026

## Prerequisites

- Environment: `conda activate thor310`
- GPU: NVIDIA RTX A4000 (16.8 GB)
- Python 3.10.19, PyTorch 2.2.1+cu121

## Smoke Test (Already Passed ✓)

```bash
cd /scratch/hkanpak21/Comp441GPW
conda activate thor310
python AGENTS/smoke_test.py
```

## Running Experiments

### Option 1: Test Single Experiment Locally

```bash
# Quick test with 10 samples
python experiment_scripts/exp_attack_synonym.py \
    --model opt-1.3b \
    --watermark unigram \
    --num_samples 10 \
    --output_dir results
```

### Option 2: Submit Simple Attacks (Recommended First)

```bash
cd experiment_scripts

# Submit for one watermark method
./submit_simple_attacks.sh opt-1.3b unigram

# Or submit for all watermarks
for wm in unigram kgw gpw semstamp; do
    ./submit_simple_attacks.sh opt-1.3b $wm
done
```

### Option 3: Submit All Experiments (Master Script)

```bash
cd experiment_scripts
./submit_all_experiments.sh opt-1.3b
```

This will submit ALL attacks for ALL watermarks:
- unigram: synonym, swap, typo, copypaste, paraphrase
- kgw: synonym, swap, typo, copypaste, paraphrase
- gpw: synonym, swap, typo, copypaste, paraphrase
- semstamp: synonym, swap, typo, copypaste, paraphrase

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Watch logs in real-time
tail -f results/logs/synonym_opt-1.3b_unigram_*.log

# Cancel all jobs
scancel -u $USER
```

## Results

### Output Files

**Per-sample results:**
```
results/exp_attack_synonym_opt-1.3b_unigram_20260121_143022.csv
```

**Summary metrics:**
```
results/exp_attack_synonym_opt-1.3b_unigram_20260121_143022_summary.json
```

**SLURM logs:**
```
results/logs/synonym_opt-1.3b_unigram_12345.log
```

### CSV Format (Per-Sample)

```csv
sample_id,prompt,original_text,attacked_text,z_score_original,z_score_attacked,is_watermarked_original,is_watermarked_attacked,num_tokens,generation_time,detection_time_original,detection_time_attacked
```

### JSON Format (Summary)

```json
{
  "experiment": "synonym_attack",
  "model": "opt-1.3b",
  "watermark": "unigram",
  "num_samples": 200,
  "original_mean_z_score": 8.52,
  "attacked_mean_z_score": 6.31,
  "original_detection_rate": 0.98,
  "attacked_detection_rate": 0.85,
  "z_score_drop": 2.21,
  "detection_rate_drop": 0.13
}
```

## Experiment Types

### Simple Attacks (~20 min each)
1. **Synonym Attack** - WordNet synonym substitution (30% edit rate)
2. **Swap Attack** - Adjacent word swaps (10% swap rate)
3. **Typo Attack** - Character-level errors (5% typo rate)

### Medium Attacks (~40 min)
4. **Copy-Paste Attack** - Mix watermarked + human text (50% ratio)

### Complex Attacks (~60 min)
5. **Paraphrase Attack** - Pegasus semantic paraphrasing (slow!)

## Watermark Methods

1. **Unigram** - Context-free green list (γ=0.5, δ=2.0)
2. **KGW** - Context-dependent hashing (γ=0.5, δ=2.0)
3. **GPW** - Gaussian pancakes with optimized detection (α=2.0, ω=2.0)
4. **SEMSTAMP** - Sentence-level semantic watermarking (d=3, max_rejections=1)

## Timeline Estimates

**Per watermark method:**
- Simple attacks: ~1 hour
- All attacks: ~2-3 hours

**All 4 watermarks (opt-1.3b):**
- Total time: ~8-12 hours

**Phase 2 (qwen-7b):**
- Run after opt-1.3b completes
- Similar timeline (may be slightly slower)

## Troubleshooting

### Job Fails Immediately
```bash
# Check the log
cat results/logs/synonym_opt-1.3b_unigram_*.log

# Common issues:
# - GPU not available: Wait or request different partition
# - Import error: conda activate thor310
# - Out of memory: Reduce num_samples in config
```

### Results Not Appearing
```bash
# Check if results directory exists
ls -la results/

# Check job status
squeue -u $USER

# Check if job completed
sacct -u $USER --format=JobID,JobName,State,ExitCode
```

### Slow Performance
- **GPW-SP (non-SR)** should be fast (~9K tok/s detection)
- **SEMSTAMP** with max_rejections=1 should be reasonable
- **Paraphrase** attack is inherently slow (~30s per sample)

## Advanced Usage

### Custom Parameters

```bash
python experiment_scripts/exp_attack_synonym.py \
    --model opt-1.3b \
    --watermark unigram \
    --num_samples 200 \
    --edit_rate 0.5 \
    --output_dir custom_results \
    --seed 123
```

### Run on Different Model

```bash
# Switch to Qwen-7B after opt-1.3b completes
./submit_all_experiments.sh qwen-7b
```

### Debug Mode (Interactive)

```bash
# Request interactive GPU session
srun -p t4_ai -N 1 --qos=comx29 --account=comx29 --gres=gpu:1 --time=00:30:00 --pty bash

# Activate environment
conda activate thor310
cd /scratch/hkanpak21/Comp441GPW

# Run experiment with small sample size
python experiment_scripts/exp_attack_synonym.py --model opt-1.3b --watermark unigram --num_samples 5
```

## Key Files

- **Smoke Test:** `AGENTS/smoke_test.py`
- **Config:** `experiment_scripts/config.py`
- **Utils:** `experiment_scripts/utils.py`
- **Dataset:** `experiment_scripts/dataset_loader.py`
- **Experiments:** `experiment_scripts/exp_*.py`
- **Submission:** `experiment_scripts/submit_*.sh`
- **Plan:** `AGENTS/plan.md`

## Performance Benchmarks

**From Smoke Test:**
- Model loading (gpt2): 5.3s
- Unigram generation: 1.8 tok/s (16.5s for 30 tokens)
- KGW generation: 72.9 tok/s (0.4s for 30 tokens)
- GPW generation: 6.5 tok/s (5.4s for 35 tokens)
- GPW-SP detection: **9572 tok/s** (0.006s for 54 tokens) - OPTIMIZED! ✅

## Support

For issues or questions:
1. Check `AGENTS/plan.md` for detailed documentation
2. Review smoke test results: `AGENTS/smoke_test_results.txt`
3. Check SLURM logs: `results/logs/*.log`

---

**Ready to run!** Start with simple attacks and verify results before running full suite.
