#!/bin/bash
#SBATCH --job-name=pythia_sm
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:55:00
#SBATCH --output=/scratch/hkanpak21/Comp441GPW/logs/pythia_small_%j.out
#SBATCH --error=/scratch/hkanpak21/Comp441GPW/logs/pythia_small_%j.err

echo "=========================================="
echo "Pythia Small Models (70M-410M) - Scaling"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate thor310

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd /scratch/hkanpak21/Comp441GPW

SHARED=/datasets/NLP/huggingface/hub

# Pythia-70M (first snapshot)
echo "=== Pythia-70M ==="
python experiment_scripts/exp_comprehensive_all.py \
    --model "Pythia-70M" \
    --path "$SHARED/models--EleutherAI--pythia-70m/snapshots/2ab25ed47af79376eed2baaf8bbb7a192a0c73ff" \
    --params 70000000 \
    --methods Baseline GPW Unigram \
    --samples 50

# Pythia-160M (first snapshot)
echo "=== Pythia-160M ==="
python experiment_scripts/exp_comprehensive_all.py \
    --model "Pythia-160M" \
    --path "$SHARED/models--EleutherAI--pythia-160m/snapshots/26b94a336e5683a217752ab7ae4bf3cbe5661365" \
    --params 160000000 \
    --methods Baseline GPW Unigram \
    --samples 50

# Pythia-410M (first snapshot)
echo "=== Pythia-410M ==="
python experiment_scripts/exp_comprehensive_all.py \
    --model "Pythia-410M" \
    --path "$SHARED/models--EleutherAI--pythia-410m/snapshots/33fc2fdda3ea75631397cc28cec556f0ef401ae7" \
    --params 410000000 \
    --methods Baseline GPW Unigram \
    --samples 50

echo "End: $(date)"
