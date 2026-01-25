#!/bin/bash
#SBATCH --job-name=paraphrase_p2
#SBATCH --output=/scratch/hkanpak21/Comp441GPW/logs/efficient/paraphrase_p2_%j.out
#SBATCH --error=/scratch/hkanpak21/Comp441GPW/logs/efficient/paraphrase_p2_%j.err
#SBATCH --partition=t4_ai
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:59:00
#SBATCH --cpus-per-task=4

# Activate environment
source /home/hkanpak21/miniconda3/etc/profile.d/conda.sh
conda activate claude_env

cd /scratch/hkanpak21/Comp441GPW

echo "========================================"
echo "Phase 2: Running detection on paraphrased texts"
echo "========================================"

# Run detection for each watermarker
for wm in gpw gpw_sp gpw_sp_low gpw_sp_sr kgw unigram; do
    INPUT="paraphrased_texts/opt-1.3b_${wm}_paraphrased.pkl"

    if [ ! -f "$INPUT" ]; then
        echo "Skipping $wm - no paraphrased file found: $INPUT"
        continue
    fi

    echo ""
    echo "========================================"
    echo "Detecting: $wm"
    echo "Input: $INPUT"
    echo "========================================"

    python experiment_scripts/paraphrase_phase2_detect.py \
        --input "$INPUT" \
        --watermarker "$wm" \
        --model "opt-1.3b" \
        --output_dir "results"

    echo "Completed: $wm"
done

echo ""
echo "========================================"
echo "Phase 2 Complete!"
echo "========================================"
ls -la results/paraphrase_*.csv
