#!/bin/bash
#SBATCH --job-name=paraphrase_p1
#SBATCH --output=/scratch/hkanpak21/Comp441GPW/logs/efficient/paraphrase_p1_%j.out
#SBATCH --error=/scratch/hkanpak21/Comp441GPW/logs/efficient/paraphrase_p1_%j.err
#SBATCH --partition=t4_ai
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:59:00
#SBATCH --cpus-per-task=4

# Activate environment
source /home/hkanpak21/miniconda3/etc/profile.d/conda.sh
conda activate claude_env

cd /scratch/hkanpak21/Comp441GPW

# Phase 1: Generate paraphrased texts for all watermarkers (one at a time)
# This only uses Pegasus model, no watermark models

echo "========================================"
echo "Phase 1: Generating paraphrased texts"
echo "========================================"

# Run for all watermarkers with 50 samples each
for wm in gpw gpw_sp gpw_sp_low gpw_sp_sr kgw unigram; do
    INPUT="generated_texts/opt-1.3b_${wm}_20260125_000*.pkl"
    INPUT_FILE=$(ls -1 $INPUT 2>/dev/null | head -1)

    if [ -z "$INPUT_FILE" ]; then
        echo "Skipping $wm - no input file found"
        continue
    fi

    OUTPUT="paraphrased_texts/opt-1.3b_${wm}_paraphrased.pkl"

    echo ""
    echo "========================================"
    echo "Processing: $wm"
    echo "Input: $INPUT_FILE"
    echo "Output: $OUTPUT"
    echo "========================================"

    python experiment_scripts/paraphrase_phase1_generate.py \
        --input "$INPUT_FILE" \
        --output "$OUTPUT" \
        --max_samples 50

    echo "Completed: $wm"
done

echo ""
echo "========================================"
echo "Phase 1 Complete!"
echo "========================================"
echo "Paraphrased files saved in: paraphrased_texts/"
ls -la paraphrased_texts/
