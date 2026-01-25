#!/bin/bash
# Submit parallel Qwen jobs for each watermarker

cd /scratch/hkanpak21/Comp441GPW

N_SAMPLES=20
ATTACKS="clean synonym_30 swap_20 typo_10 copypaste_50"

# Create individual job scripts and submit them
for WM in none unigram kgw gpw gpw_sp gpw_sp_low gpw_sp_sr; do
    JOB_SCRIPT="/scratch/hkanpak21/Comp441GPW/experiment_scripts/qwen_${WM}.sh"

    cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=qwen_${WM}
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=0:45:00
#SBATCH --output=/scratch/hkanpak21/Comp441GPW/logs/efficient/qwen_${WM}_%j.out
#SBATCH --error=/scratch/hkanpak21/Comp441GPW/logs/efficient/qwen_${WM}_%j.err

echo "=========================================="
echo "Qwen-7B: ${WM} watermarker"
echo "=========================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start: \$(date)"

source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate thor310

cd /scratch/hkanpak21/Comp441GPW

# Generate texts
echo "Generating ${N_SAMPLES} texts with ${WM}..."
python experiment_scripts/generate_texts.py \\
    --model qwen-7b \\
    --watermarker ${WM} \\
    --n_samples ${N_SAMPLES}

# Find generated file
INPUT_FILE=\$(ls -1t generated_texts/qwen-7b_${WM}_*.pkl 2>/dev/null | head -1)

if [ -z "\$INPUT_FILE" ]; then
    echo "ERROR: No generated file found"
    exit 1
fi

echo "Running attacks on: \$INPUT_FILE"

if [ "${WM}" == "none" ]; then
    # For baseline, test with each detector
    for DETECTOR in gpw unigram kgw; do
        echo "Testing baseline with detector: \$DETECTOR"
        python experiment_scripts/run_attacks_on_texts.py \\
            --input "\$INPUT_FILE" \\
            --attacks ${ATTACKS} \\
            --detector \$DETECTOR
    done
else
    python experiment_scripts/run_attacks_on_texts.py \\
        --input "\$INPUT_FILE" \\
        --attacks ${ATTACKS}
fi

echo ""
echo "=========================================="
echo "COMPLETE"
echo "=========================================="
echo "End: \$(date)"
EOF

    # Submit the job
    echo "Submitting job for ${WM}..."
    sbatch "$JOB_SCRIPT"
done

echo ""
echo "All jobs submitted!"
