#!/bin/bash
#SBATCH --job-name=gpt2_none
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/hkanpak21/Comp441GPW/logs/efficient/gpt2_none_%j.out
#SBATCH --error=/scratch/hkanpak21/Comp441GPW/logs/efficient/gpt2_none_%j.err

echo "=========================================="
echo "GPT-2 No Watermark Baseline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate thor310

cd /scratch/hkanpak21/Comp441GPW

# Generate without watermark
python experiment_scripts/generate_texts.py \
    --model gpt2 \
    --watermarker none \
    --n_samples 200

# Run attacks
INPUT_FILE=$(ls -1t generated_texts/gpt2_none_*.pkl 2>/dev/null | head -1)

if [ -n "$INPUT_FILE" ]; then
    python experiment_scripts/run_attacks_on_texts.py \
        --input "$INPUT_FILE" \
        --attacks clean synonym_30 swap_20 typo_10 copypaste_50
fi

echo ""
echo "=========================================="
echo "COMPLETE"
echo "=========================================="
echo "End: $(date)"
