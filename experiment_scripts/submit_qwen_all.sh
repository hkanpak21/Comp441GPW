#!/bin/bash
#SBATCH --job-name=qwen_all
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=/scratch/hkanpak21/Comp441GPW/logs/efficient/qwen_all_%j.out
#SBATCH --error=/scratch/hkanpak21/Comp441GPW/logs/efficient/qwen_all_%j.err

echo "=========================================="
echo "Qwen-7B Complete Watermarking Experiments"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "Model: Qwen/Qwen2.5-7B-Instruct (cached locally)"

source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate thor310

cd /scratch/hkanpak21/Comp441GPW

N_SAMPLES=20

# Define watermarkers to test
WATERMARKERS="none unigram kgw gpw gpw_sp gpw_sp_low gpw_sp_sr"

# Define attacks (excluding paraphrase)
ATTACKS="clean synonym_30 swap_20 typo_10 copypaste_50"

echo ""
echo "=========================================="
echo "PHASE 1: Generate texts for all watermarkers"
echo "=========================================="

for WM in $WATERMARKERS; do
    echo ""
    echo "----------------------------------------"
    echo "Generating texts with watermarker: $WM"
    echo "----------------------------------------"

    python experiment_scripts/generate_texts.py \
        --model qwen-7b \
        --watermarker $WM \
        --n_samples $N_SAMPLES

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to generate texts with $WM"
    fi
done

echo ""
echo "=========================================="
echo "PHASE 2: Run attacks and detection"
echo "=========================================="

# Find all generated Qwen files
for WM in $WATERMARKERS; do
    INPUT_FILE=$(ls -1t generated_texts/qwen-7b_${WM}_*.pkl 2>/dev/null | head -1)

    if [ -z "$INPUT_FILE" ]; then
        echo "WARNING: No generated file found for $WM"
        continue
    fi

    echo ""
    echo "----------------------------------------"
    echo "Processing: $INPUT_FILE"
    echo "----------------------------------------"

    if [ "$WM" == "none" ]; then
        # For baseline, need to specify detector
        for DETECTOR in unigram kgw gpw gpw_sp gpw_sp_low gpw_sp_sr; do
            echo "Testing baseline with detector: $DETECTOR"
            python experiment_scripts/run_attacks_on_texts.py \
                --input "$INPUT_FILE" \
                --attacks $ATTACKS \
                --detector $DETECTOR
        done
    else
        # For watermarked texts, detector is inferred
        python experiment_scripts/run_attacks_on_texts.py \
            --input "$INPUT_FILE" \
            --attacks $ATTACKS
    fi
done

echo ""
echo "=========================================="
echo "PHASE 3: Aggregate results"
echo "=========================================="

# List all result files
echo "Generated result files:"
ls -la results/qwen*.csv 2>/dev/null || echo "No result files found yet"

echo ""
echo "=========================================="
echo "COMPLETE"
echo "=========================================="
echo "End: $(date)"
