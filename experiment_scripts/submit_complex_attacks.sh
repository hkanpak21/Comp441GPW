#!/bin/bash
# SLURM submission script for complex attack experiments (copypaste, paraphrase)
# These are slower - submit after simple attacks complete

# Usage: ./submit_complex_attacks.sh [model] [watermark]
# Example: ./submit_complex_attacks.sh opt-1.3b unigram

MODEL=${1:-opt-1.3b}
WATERMARK=${2:-unigram}

echo "=========================================="
echo "Submitting Complex Attack Experiments"
echo "=========================================="
echo "Model: $MODEL"
echo "Watermark: $WATERMARK"
echo "=========================================="

# SLURM parameters
PARTITION="t4_ai"
QOS="comx29"
ACCOUNT="comx29"
GPU="gpu:1"
TIME="00:40:00"  # Longer time for complex attacks

# Experiment directory
EXP_DIR="/scratch/hkanpak21/Comp441GPW/experiment_scripts"

# Python path (use full path to avoid conda activation issues)
PYTHON="/home/hkanpak21/.conda/envs/thor310/bin/python -u"  # -u for unbuffered output

# 1. Copy-Paste Attack
echo "[1/2] Submitting Copy-Paste Attack..."
sbatch --partition=$PARTITION --nodes=1 --qos=$QOS --account=$ACCOUNT --gres=$GPU --time=$TIME \
    --job-name="copypaste_${WATERMARK}" \
    --output="../results/logs/copypaste_${MODEL}_${WATERMARK}_%j.log" \
    --wrap="cd $EXP_DIR && $PYTHON exp_attack_copypaste.py --model $MODEL --watermark $WATERMARK --num_samples 200 --output_dir ../results"

# 2. Paraphrase Attack (slowest)
echo "[2/2] Submitting Paraphrase Attack..."
sbatch --partition=$PARTITION --nodes=1 --qos=$QOS --account=$ACCOUNT --gres=$GPU --time="01:00:00" \
    --job-name="paraphrase_${WATERMARK}" \
    --output="../results/logs/paraphrase_${MODEL}_${WATERMARK}_%j.log" \
    --wrap="cd $EXP_DIR && $PYTHON exp_attack_paraphrase.py --model $MODEL --watermark $WATERMARK --num_samples 200 --output_dir ../results"

echo ""
echo "All complex attacks submitted!"
echo "Monitor with: squeue -u $USER"
echo "Logs in: ../results/logs/"
