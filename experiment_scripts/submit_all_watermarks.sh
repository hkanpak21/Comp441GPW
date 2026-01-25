#!/bin/bash
# Submit all watermark experiments as independent jobs
# Each job runs a different watermarker or baseline

EXP_DIR=/scratch/hkanpak21/Comp441GPW/experiment_scripts
PYTHON=/home/hkanpak21/.conda/envs/thor310/bin/python
MODEL="opt-1.3b"
NUM_SAMPLES=200

echo "========================================"
echo "Submitting Watermark Experiments"
echo "========================================"
echo "Model: $MODEL"
echo "Samples: $NUM_SAMPLES"
echo "========================================"

# Job 1: Unigram watermark
echo "[1/4] Submitting Unigram experiments..."
sbatch --partition=t4_ai --qos=comx29 --account=comx29 --gres=gpu:1 \
    --job-name=wm_unigram \
    --output=../results/logs/wm_unigram_%j.log \
    --time=02:00:00 \
    --wrap="cd $EXP_DIR && $PYTHON -u exp_watermark_unigram.py --model $MODEL --num_samples $NUM_SAMPLES --output_dir ../results"

# Job 2: KGW watermark
echo "[2/4] Submitting KGW experiments..."
sbatch --partition=t4_ai --qos=comx29 --account=comx29 --gres=gpu:1 \
    --job-name=wm_kgw \
    --output=../results/logs/wm_kgw_%j.log \
    --time=02:00:00 \
    --wrap="cd $EXP_DIR && $PYTHON -u exp_watermark_kgw.py --model $MODEL --num_samples $NUM_SAMPLES --output_dir ../results"

# Job 3: GPW-SP watermark
echo "[3/4] Submitting GPW-SP experiments..."
sbatch --partition=t4_ai --qos=comx29 --account=comx29 --gres=gpu:1 \
    --job-name=wm_gpw_sp \
    --output=../results/logs/wm_gpw_sp_%j.log \
    --time=02:00:00 \
    --wrap="cd $EXP_DIR && $PYTHON -u exp_watermark_gpw.py --model $MODEL --num_samples $NUM_SAMPLES --output_dir ../results"

# Job 4: Baselines (Human + Unwatermarked AI)
echo "[4/4] Submitting Baseline experiments..."
sbatch --partition=t4_ai --qos=comx29 --account=comx29 --gres=gpu:1 \
    --job-name=baselines \
    --output=../results/logs/baselines_%j.log \
    --time=02:00:00 \
    --wrap="cd $EXP_DIR && $PYTHON -u exp_baselines.py --model $MODEL --num_samples $NUM_SAMPLES --output_dir ../results"

echo ""
echo "========================================"
echo "All jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "Logs: ../results/logs/"
echo "========================================"
