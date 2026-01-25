#!/bin/bash
# Quick smoke test submission script
# Tests environment and watermarkers before running full experiments

echo "=========================================="
echo "Submitting Smoke Test"
echo "=========================================="

# SLURM parameters
PARTITION="t4_ai"
QOS="comx29"
ACCOUNT="comx29"
GPU="gpu:1"
TIME="00:10:00"  # Short time for quick test

# Run smoke test
srun -p $PARTITION -N 1 --qos=$QOS --account=$ACCOUNT --gres=$GPU --time=$TIME \
    --job-name="smoke_test" \
    --output="../results/logs/smoke_test_%j.log" \
    --pty \
    python /scratch/hkanpak21/Comp441GPW/AGENTS/smoke_test.py

echo ""
echo "Smoke test complete!"
echo "Check results in: /scratch/hkanpak21/Comp441GPW/AGENTS/smoke_test_results.txt"
