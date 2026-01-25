#!/bin/bash
# Submit all watermarking experiments for multiple models
# Usage: ./submit_all_models.sh [gpt2|opt-1.3b|qwen-7b|qwen-14b|all]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONDA_ENV="/home/hkanpak21/.conda/envs/thor310"
PYTHON="${CONDA_ENV}/bin/python"
LOG_DIR="${PROJECT_ROOT}/logs"

# Library path fix for NLTK sqlite issue
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH}"

mkdir -p "$LOG_DIR"

# Common SLURM options
SLURM_OPTS="--partition=t4_ai --qos=comx29 --account=comx29 --gres=gpu:1"

# Wrapper command that sets up environment properly
WRAPPER_PREFIX="export LD_LIBRARY_PATH=${CONDA_ENV}/lib:\$LD_LIBRARY_PATH && $PYTHON -u"

submit_gpw_tuning() {
    echo "Submitting GPW tuning experiment..."
    sbatch $SLURM_OPTS --job-name=gpw_tuning --time=01:00:00 \
        --output=${LOG_DIR}/gpw_tuning_%j.out \
        --error=${LOG_DIR}/gpw_tuning_%j.err \
        --wrap="$WRAPPER_PREFIX ${SCRIPT_DIR}/exp_gpw_tuning.py"
}

submit_model() {
    local model=$1
    local time_limit=$2
    
    echo ""
    echo "=========================================="
    echo "Submitting experiments for: $model"
    echo "=========================================="
    
    # Comprehensive experiments (Unigram, KGW, GPW-SP + baselines)
    echo "  -> Submitting comprehensive experiments..."
    sbatch $SLURM_OPTS --job-name="comp_${model}" --time=${time_limit} \
        --output=${LOG_DIR}/comp_${model}_%j.out \
        --error=${LOG_DIR}/comp_${model}_%j.err \
        --wrap="$WRAPPER_PREFIX ${SCRIPT_DIR}/exp_comprehensive.py $model"
    
    # SemStamp experiments (separate due to embedder overhead)
    echo "  -> Submitting SemStamp experiments..."
    sbatch $SLURM_OPTS --job-name="semstamp_${model}" --time=${time_limit} \
        --output=${LOG_DIR}/semstamp_${model}_%j.out \
        --error=${LOG_DIR}/semstamp_${model}_%j.err \
        --wrap="$WRAPPER_PREFIX ${SCRIPT_DIR}/exp_watermark_semstamp.py $model"
}

# Parse argument
case "${1:-opt-1.3b}" in
    "tuning")
        submit_gpw_tuning
        ;;
    "gpt2")
        submit_model "gpt2" "01:00:00"
        ;;
    "opt-1.3b"|"opt")
        submit_model "opt-1.3b" "02:00:00"
        ;;
    "qwen-7b"|"qwen7")
        submit_model "qwen-7b" "03:00:00"
        ;;
    "qwen-14b"|"qwen14")
        submit_model "qwen-14b" "04:00:00"
        ;;
    "all")
        # First run GPW tuning
        submit_gpw_tuning
        
        # Then all models
        submit_model "gpt2" "01:00:00"
        submit_model "opt-1.3b" "02:00:00"
        submit_model "qwen-7b" "03:00:00"
        submit_model "qwen-14b" "04:00:00"
        ;;
    "quick")
        # Quick test with GPT2 only
        submit_model "gpt2" "01:00:00"
        ;;
    *)
        echo "Usage: $0 [tuning|gpt2|opt-1.3b|qwen-7b|qwen-14b|all|quick]"
        echo ""
        echo "Options:"
        echo "  tuning   - GPW parameter tuning only"
        echo "  gpt2     - GPT2 experiments"
        echo "  opt-1.3b - OPT-1.3B experiments (default)"
        echo "  qwen-7b  - Qwen 7B experiments"  
        echo "  qwen-14b - Qwen 14B experiments"
        echo "  all      - All models + tuning"
        echo "  quick    - GPT2 only (fastest)"
        exit 1
        ;;
esac

echo ""
echo "Jobs submitted! Check status with: squeue -u \$USER"
