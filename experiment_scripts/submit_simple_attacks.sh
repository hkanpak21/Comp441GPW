#!/bin/bash
# SLURM submission script for simple attack experiments (synonym, swap, typo)
# These are the fastest attacks - submit first

# Usage: ./submit_simple_attacks.sh [model] [watermark]
# Example: ./submit_simple_attacks.sh opt-1.3b unigram

MODEL=${1:-opt-1.3b}
WATERMARK=${2:-unigram}

echo "=========================================="
echo "Submitting Simple Attack Experiments"
echo "=========================================="
echo "Model: $MODEL"
echo "Watermark: $WATERMARK"
echo "=========================================="

# SLURM parameters
PARTITION="t4_ai"
QOS="comx29"
ACCOUNT="comx29"
GPU="gpu:1"
TIME="00:20:00"

# Experiment directory
EXP_DIR="/scratch/hkanpak21/Comp441GPW/experiment_scripts"

# Python path (use full path to avoid conda activation issues)
PYTHON="/home/hkanpak21/.conda/envs/thor310/bin/python -u"  # -u for unbuffered output

# 1. Synonym Attack
echo "[1/3] Submitting Synonym Attack..."
sbatch --partition=$PARTITION --nodes=1 --qos=$QOS --account=$ACCOUNT --gres=$GPU --time=$TIME \
    --job-name="synonym_${WATERMARK}" \
    --output="../results/logs/synonym_${MODEL}_${WATERMARK}_%j.log" \
    --wrap="cd $EXP_DIR && $PYTHON exp_attack_synonym.py --model $MODEL --watermark $WATERMARK --num_samples 200 --output_dir ../results"

# 2. Swap Attack
echo "[2/3] Submitting Swap Attack..."
sbatch --partition=$PARTITION --nodes=1 --qos=$QOS --account=$ACCOUNT --gres=$GPU --time=$TIME \
    --job-name="swap_${WATERMARK}" \
    --output="../results/logs/swap_${MODEL}_${WATERMARK}_%j.log" \
    --wrap="cd $EXP_DIR && $PYTHON exp_attack_swap.py --model $MODEL --watermark $WATERMARK --num_samples 200 --output_dir ../results"

# 3. Typo Attack
echo "[3/3] Submitting Typo Attack..."
sbatch --partition=$PARTITION --nodes=1 --qos=$QOS --account=$ACCOUNT --gres=$GPU --time=$TIME \
    --job-name="typo_${WATERMARK}" \
    --output="../results/logs/typo_${MODEL}_${WATERMARK}_%j.log" \
    --wrap="cd $EXP_DIR && $PYTHON exp_attack_typo.py --model $MODEL --watermark $WATERMARK --num_samples 200 --output_dir ../results"

echo ""
echo "All simple attacks submitted!"
echo "Monitor with: squeue -u $USER"
echo "Logs in: ../results/logs/"
