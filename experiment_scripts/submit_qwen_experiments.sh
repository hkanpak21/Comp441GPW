#!/bin/bash
#SBATCH --job-name=qwen_exp
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=1:55:00
#SBATCH --output=/scratch/hkanpak21/Comp441GPW/logs/efficient/qwen_exp_%j.out
#SBATCH --error=/scratch/hkanpak21/Comp441GPW/logs/efficient/qwen_exp_%j.err

echo "=========================================="
echo "Qwen-7B Experiments - All Attacks (except paraphrase)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate thor310

cd /scratch/hkanpak21/Comp441GPW

# Phase 1: Generate texts for all watermarkers
echo ""
echo "=========================================="
echo "PHASE 1: Generating watermarked texts"
echo "=========================================="

WATERMARKERS="gpw gpw_sp gpw_sp_low unigram kgw none"

for wm in $WATERMARKERS; do
    echo ""
    echo "Generating for: $wm"
    python experiment_scripts/generate_texts.py \
        --model qwen-7b \
        --watermarker $wm \
        --n_samples 200
done

echo ""
echo "Generated texts:"
ls -la generated_texts/qwen*.pkl

# Phase 2: Run attacks on generated texts
echo ""
echo "=========================================="
echo "PHASE 2: Running attacks"
echo "=========================================="

# Run for each watermarker (except none - that's baseline)
for wm in gpw gpw_sp gpw_sp_low unigram kgw; do
    INPUT_FILE=$(ls -1 generated_texts/qwen-7b_${wm}_*.pkl 2>/dev/null | head -1)

    if [ -z "$INPUT_FILE" ]; then
        echo "Skipping $wm - no input file found"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Attacks for: $wm"
    echo "Input: $INPUT_FILE"
    echo "=========================================="

    python experiment_scripts/run_attacks_on_texts.py \
        --input "$INPUT_FILE" \
        --attacks clean synonym_30 swap_20 typo_10 copypaste_50
done

echo ""
echo "=========================================="
echo "COMPLETE"
echo "=========================================="
echo "End: $(date)"
ls -la results/attack_qwen*.csv 2>/dev/null
