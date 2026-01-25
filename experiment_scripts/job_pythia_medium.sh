#!/bin/bash
#SBATCH --job-name=pythia_md
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:55:00
#SBATCH --output=/scratch/hkanpak21/Comp441GPW/logs/pythia_medium_%j.out
#SBATCH --error=/scratch/hkanpak21/Comp441GPW/logs/pythia_medium_%j.err

echo "=========================================="
echo "Pythia Medium Models (1B-1.4B) - Scaling"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate thor310

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd /scratch/hkanpak21/Comp441GPW

SHARED=/datasets/NLP/huggingface/hub

# Pythia-1B (first snapshot)
echo "=== Pythia-1B ==="
python experiment_scripts/exp_comprehensive_all.py \
    --model "Pythia-1B" \
    --path "$SHARED/models--EleutherAI--pythia-1b/snapshots/d32083609b90e1829dd1c3d901d6ba4523cede9d" \
    --params 1000000000 \
    --methods Baseline GPW Unigram \
    --samples 50

# Pythia-1.4B (first snapshot)
echo "=== Pythia-1.4B ==="
python experiment_scripts/exp_comprehensive_all.py \
    --model "Pythia-1.4B" \
    --path "$SHARED/models--EleutherAI--pythia-1.4b/snapshots/8f0af839162ddb466006494a08733ee9cfa2d338" \
    --params 1400000000 \
    --methods Baseline GPW Unigram \
    --samples 50

echo "End: $(date)"
