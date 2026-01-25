#!/bin/bash
#SBATCH --job-name=gpw_ablation
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0:55:00
#SBATCH --output=/scratch/hkanpak21/Comp441GPW/logs/efficient/gpw_ablation_%j.out
#SBATCH --error=/scratch/hkanpak21/Comp441GPW/logs/efficient/gpw_ablation_%j.err

echo "=========================================="
echo "GPW Ablation Study - GPT-2"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate thor310

cd /scratch/hkanpak21/Comp441GPW

# Run ablation study
python experiment_scripts/gpw_ablation_study.py \
    --model gpt2 \
    --n_samples 50 \
    --output results/gpw_ablation_gpt2.csv

echo ""
echo "=========================================="
echo "COMPLETE"
echo "=========================================="
echo "End: $(date)"
