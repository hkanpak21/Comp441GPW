#!/bin/bash
#SBATCH --job-name=qwen_gpw
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=0:45:00
#SBATCH --output=/scratch/hkanpak21/Comp441GPW/logs/efficient/qwen_gpw_%j.out
#SBATCH --error=/scratch/hkanpak21/Comp441GPW/logs/efficient/qwen_gpw_%j.err

echo "=========================================="
echo "Qwen-7B: gpw watermarker"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate thor310

cd /scratch/hkanpak21/Comp441GPW

# Generate texts
echo "Generating 20 texts with gpw..."
python experiment_scripts/generate_texts.py \
    --model qwen-7b \
    --watermarker gpw \
    --n_samples 20

# Find generated file
INPUT_FILE=$(ls -1t generated_texts/qwen-7b_gpw_*.pkl 2>/dev/null | head -1)

if [ -z "$INPUT_FILE" ]; then
    echo "ERROR: No generated file found"
    exit 1
fi

echo "Running attacks on: $INPUT_FILE"

if [ "gpw" == "none" ]; then
    # For baseline, test with each detector
    for DETECTOR in gpw unigram kgw; do
        echo "Testing baseline with detector: $DETECTOR"
        python experiment_scripts/run_attacks_on_texts.py \
            --input "$INPUT_FILE" \
            --attacks clean synonym_30 swap_20 typo_10 copypaste_50 \
            --detector $DETECTOR
    done
else
    python experiment_scripts/run_attacks_on_texts.py \
        --input "$INPUT_FILE" \
        --attacks clean synonym_30 swap_20 typo_10 copypaste_50
fi

echo ""
echo "=========================================="
echo "COMPLETE"
echo "=========================================="
echo "End: $(date)"
