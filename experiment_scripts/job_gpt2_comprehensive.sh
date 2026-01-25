#!/bin/bash
#SBATCH --job-name=gpt2_all
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1:55:00
#SBATCH --output=/scratch/hkanpak21/Comp441GPW/logs/gpt2_comprehensive_%j.out
#SBATCH --error=/scratch/hkanpak21/Comp441GPW/logs/gpt2_comprehensive_%j.err

echo "=========================================="
echo "GPT-2 - All Methods + Baseline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate thor310

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd /scratch/hkanpak21/Comp441GPW

python experiment_scripts/exp_comprehensive_all.py \
    --model "GPT-2" \
    --path "gpt2" \
    --params 125000000 \
    --methods Baseline GPW Unigram KGW \
    --samples 100

echo "End: $(date)"
