#!/bin/bash
#SBATCH --job-name=gpt2_attacks
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:55:00
#SBATCH --output=/scratch/hkanpak21/Comp441GPW/logs/efficient/gpt2_attacks_%j.out
#SBATCH --error=/scratch/hkanpak21/Comp441GPW/logs/efficient/gpt2_attacks_%j.err

echo "=========================================="
echo "GPT-2 Attacks on Existing Texts"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate thor310

cd /scratch/hkanpak21/Comp441GPW

echo ""
echo "Available GPT-2 texts:"
ls -la generated_texts/gpt2*.pkl

# Run attacks on each available text file
for f in generated_texts/gpt2_*.pkl; do
    if [ ! -f "$f" ]; then
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Processing: $f"
    echo "=========================================="

    # Check if it's gpw_sp_sr (use fewer samples)
    if [[ "$f" == *"gpw_sp_sr"* ]]; then
        python experiment_scripts/run_attacks_on_texts.py \
            --input "$f" \
            --attacks clean synonym_30 swap_20 typo_10 copypaste_50 \
            --max_samples 50
    else
        python experiment_scripts/run_attacks_on_texts.py \
            --input "$f" \
            --attacks clean synonym_30 swap_20 typo_10 copypaste_50
    fi
done

echo ""
echo "=========================================="
echo "COMPLETE"
echo "=========================================="
echo "End: $(date)"
ls -la results/attack_gpt2*.csv 2>/dev/null
