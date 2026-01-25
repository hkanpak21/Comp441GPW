#!/bin/bash
# Master submission script for all watermark experiments
# Submits all attacks for all watermark methods sequentially

# Usage: ./submit_all_experiments.sh [model]
# Example: ./submit_all_experiments.sh opt-1.3b

MODEL=${1:-opt-1.3b}

echo "=========================================="
echo "Master Experiment Submission"
echo "=========================================="
echo "Model: $MODEL"
echo "Running all attacks for all watermarks:"
echo "  - unigram"
echo "  - kgw"
echo "  - gpw"
echo "  - semstamp"
echo "=========================================="

# Make scripts executable
chmod +x submit_simple_attacks.sh
chmod +x submit_complex_attacks.sh

# Submit experiments for each watermark method
for WATERMARK in unigram kgw gpw semstamp; do
    echo ""
    echo "=== Starting experiments for $WATERMARK ==="
    
    # Simple attacks first
    echo "Submitting simple attacks..."
    ./submit_simple_attacks.sh $MODEL $WATERMARK
    
    # Wait a bit to avoid overwhelming the scheduler
    sleep 5
    
    # Complex attacks
    echo "Submitting complex attacks..."
    ./submit_complex_attacks.sh $MODEL $WATERMARK
    
    echo "=== $WATERMARK experiments submitted ==="
    sleep 5
done

echo ""
echo "=========================================="
echo "All experiments submitted!"
echo "=========================================="
echo "Monitor all jobs with: squeue -u $USER"
echo "Cancel all jobs with: scancel -u $USER"
echo "Results will be saved to: ../results/"
echo "Logs will be saved to: ../results/logs/"
