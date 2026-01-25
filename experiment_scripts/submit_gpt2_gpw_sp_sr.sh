#!/bin/bash
#SBATCH --job-name=gpt2_sr
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:55:00
#SBATCH --output=/scratch/hkanpak21/Comp441GPW/logs/efficient/gpt2_gpw_sp_sr_%j.out
#SBATCH --error=/scratch/hkanpak21/Comp441GPW/logs/efficient/gpt2_gpw_sp_sr_%j.err

echo "=========================================="
echo "GPT-2 GPW-SP-SR Experiments (50 samples)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate thor310

cd /scratch/hkanpak21/Comp441GPW

# Generate texts
echo ""
echo "Generating GPW-SP-SR texts..."
python experiment_scripts/generate_texts.py \
    --model gpt2 \
    --watermarker gpw_sp_sr \
    --n_samples 200

# Run attacks with limited samples (50 due to slow detection)
INPUT_FILE=$(ls -1 generated_texts/gpt2_gpw_sp_sr_*.pkl 2>/dev/null | head -1)

if [ -z "$INPUT_FILE" ]; then
    echo "ERROR: No input file found for gpw_sp_sr"
    exit 1
fi

echo ""
echo "Running attacks on: $INPUT_FILE"
python experiment_scripts/run_attacks_on_texts.py \
    --input "$INPUT_FILE" \
    --attacks clean synonym_30 swap_20 typo_10 copypaste_50 \
    --max_samples 50

echo ""
echo "=========================================="
echo "COMPLETE"
echo "=========================================="
echo "End: $(date)"
