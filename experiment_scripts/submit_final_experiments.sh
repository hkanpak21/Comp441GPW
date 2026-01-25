#!/bin/bash
#
# Submit comprehensive watermarking experiments for all models
# Usage: ./submit_final_experiments.sh
#

WORKDIR="/scratch/hkanpak21/Comp441GPW"
LOGDIR="${WORKDIR}/logs"
SCRIPT="${WORKDIR}/experiment_scripts/exp_final_comprehensive.py"

mkdir -p ${LOGDIR}

# Common SLURM options
PARTITION="t4_ai"
QOS="comx29"
ACCOUNT="comx29"
GRES="gpu:1"
TIME="1:55:00"  # Under 2hr partition limit
MEM="32G"

# Environment setup commands
ENV_SETUP="export LD_LIBRARY_PATH=/home/hkanpak21/.conda/envs/thor310/lib:\$LD_LIBRARY_PATH && source ~/.bashrc && conda activate thor310 && cd ${WORKDIR}"

echo "=========================================="
echo "Submitting Final Comprehensive Experiments"
echo "=========================================="

# ============================================================================
# Phase 1: OPT-1.3B (Primary Model) - Full suite
# ============================================================================
echo ""
echo "Phase 1: OPT-1.3B Full Experiments"
echo "-----------------------------------"

# All watermarkers (except semstamp for now - test separately)
JOB_OPT=$(sbatch --parsable \
    --partition=${PARTITION} \
    --qos=${QOS} \
    --account=${ACCOUNT} \
    --gres=${GRES} \
    --time=${TIME} \
    --mem=${MEM} \
    --job-name="final_opt" \
    --output="${LOGDIR}/final_opt_%j.out" \
    --error="${LOGDIR}/final_opt_%j.err" \
    --wrap="${ENV_SETUP} && python ${SCRIPT} --model opt-1.3b --n_samples 200 --watermarkers unigram kgw gpw gpw_sp gpw_sp_low gpw_sp_sr --output_prefix final")

echo "Submitted OPT-1.3B job: ${JOB_OPT}"

# SEMSTAMP separately (slower due to rejection sampling)
JOB_SEMSTAMP=$(sbatch --parsable \
    --partition=${PARTITION} \
    --qos=${QOS} \
    --account=${ACCOUNT} \
    --gres=${GRES} \
    --time=${TIME} \
    --mem=${MEM} \
    --job-name="final_semstamp" \
    --output="${LOGDIR}/final_semstamp_%j.out" \
    --error="${LOGDIR}/final_semstamp_%j.err" \
    --wrap="${ENV_SETUP} && python ${SCRIPT} --model opt-1.3b --n_samples 100 --watermarkers semstamp --skip_baselines --output_prefix final")

echo "Submitted SEMSTAMP job: ${JOB_SEMSTAMP}"

# ============================================================================
# Phase 2: GPT-2 (Fast validation)
# ============================================================================
echo ""
echo "Phase 2: GPT-2 Validation"
echo "-------------------------"

JOB_GPT2=$(sbatch --parsable \
    --partition=${PARTITION} \
    --qos=${QOS} \
    --account=${ACCOUNT} \
    --gres=${GRES} \
    --time=${TIME} \
    --mem=${MEM} \
    --job-name="final_gpt2" \
    --output="${LOGDIR}/final_gpt2_%j.out" \
    --error="${LOGDIR}/final_gpt2_%j.err" \
    --wrap="${ENV_SETUP} && python ${SCRIPT} --model gpt2 --n_samples 200 --watermarkers unigram kgw gpw_sp --output_prefix final")

echo "Submitted GPT-2 job: ${JOB_GPT2}"

# ============================================================================
# Phase 3: Qwen-7B (Larger model)
# ============================================================================
echo ""
echo "Phase 3: Qwen-7B (Larger Model)"
echo "-------------------------------"

JOB_QWEN=$(sbatch --parsable \
    --partition=${PARTITION} \
    --qos=${QOS} \
    --account=${ACCOUNT} \
    --gres=${GRES} \
    --time=${TIME} \
    --mem=48G \
    --job-name="final_qwen7b" \
    --output="${LOGDIR}/final_qwen7b_%j.out" \
    --error="${LOGDIR}/final_qwen7b_%j.err" \
    --wrap="${ENV_SETUP} && python ${SCRIPT} --model qwen-7b --n_samples 100 --watermarkers unigram kgw gpw_sp --skip_baselines --output_prefix final")

echo "Submitted Qwen-7B job: ${JOB_QWEN}"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo "OPT-1.3B main:  ${JOB_OPT}"
echo "SEMSTAMP:       ${JOB_SEMSTAMP}"
echo "GPT-2:          ${JOB_GPT2}"
echo "Qwen-7B:        ${JOB_QWEN}"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "View logs:    tail -f ${LOGDIR}/final_*.out"
