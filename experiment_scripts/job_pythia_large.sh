#!/bin/bash
#SBATCH --job-name=pythia_lg
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=1:55:00
#SBATCH --output=/scratch/hkanpak21/Comp441GPW/logs/pythia_large_%j.out
#SBATCH --error=/scratch/hkanpak21/Comp441GPW/logs/pythia_large_%j.err

echo "=========================================="
echo "Pythia Large Models (2.8B-6.9B) - Scaling"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate thor310

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd /scratch/hkanpak21/Comp441GPW

SHARED=/datasets/NLP/huggingface/hub

# Pythia-2.8B (uses device_map for offloading)
echo "=== Pythia-2.8B ==="
python experiment_scripts/exp_comprehensive_all.py \
    --model "Pythia-2.8B" \
    --path "$SHARED/models--EleutherAI--pythia-2.8b/snapshots/0b4fc522e9aeb35aeebbc44d05236cb68dd805cd" \
    --params 2800000000 \
    --methods Baseline GPW Unigram \
    --samples 50

# Pythia-6.9B (uses device_map)
echo "=== Pythia-6.9B ==="
python experiment_scripts/exp_comprehensive_all.py \
    --model "Pythia-6.9B" \
    --path "$SHARED/models--EleutherAI--pythia-6.9b/snapshots/f271943e880e60c0c715fd10e4dc74ec4e31eb44" \
    --params 6900000000 \
    --methods Baseline GPW Unigram \
    --samples 50

echo "End: $(date)"
