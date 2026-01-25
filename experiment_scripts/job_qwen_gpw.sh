#!/bin/bash
#SBATCH --job-name=qwen_gpw
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:55:00
#SBATCH --output=/scratch/hkanpak21/Comp441GPW/logs/qwen_gpw_%j.out
#SBATCH --error=/scratch/hkanpak21/Comp441GPW/logs/qwen_gpw_%j.err

echo "=========================================="
echo "QWEN 7B - GPW Methods"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate thor310

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd /scratch/hkanpak21/Comp441GPW

python experiment_scripts/exp_comprehensive_all.py \
    --model "Qwen2.5-7B" \
    --path "/datasets/NLP/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796" \
    --params 7000000000 \
    --methods Baseline GPW \
    --samples 50

echo "End: $(date)"
