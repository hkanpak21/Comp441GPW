#!/usr/bin/env python3
"""
Submit Missing Experiments

Generates and submits individual SLURM jobs for each missing experiment.
Each job is submitted separately for better tracking.

Usage:
    python submit_missing_experiments.py --list              # List all missing experiments
    python submit_missing_experiments.py --submit opt        # Submit OPT-1.3B experiments
    python submit_missing_experiments.py --submit gpt2       # Submit GPT-2 experiments
    python submit_missing_experiments.py --submit all        # Submit all missing experiments
    python submit_missing_experiments.py --submit-one MODEL WATERMARKER ATTACK  # Submit single experiment
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

WORKDIR = "/scratch/hkanpak21/Comp441GPW"
LOGDIR = f"{WORKDIR}/logs/experiments"
SCRIPT = f"{WORKDIR}/experiment_scripts/run_single_experiment.py"

# SLURM configuration
SLURM_CONFIG = {
    "partition": "t4_ai",
    "qos": "comx29",
    "account": "comx29",
    "gres": "gpu:1",
    "time": "1:55:00",
    "mem": "32G",
}

ENV_SETUP = "export LD_LIBRARY_PATH=/home/hkanpak21/.conda/envs/thor310/lib:$LD_LIBRARY_PATH && source ~/.bashrc && conda activate thor310"

# Define completed experiments based on unified_results.csv
COMPLETED_OPT = {
    ("gpw", "clean"), ("gpw", "synonym_30"), ("gpw", "swap_20"), ("gpw", "typo_10"), ("gpw", "copypaste_50"),
    ("gpw_sp", "clean"), ("gpw_sp", "synonym_30"), ("gpw_sp", "swap_20"), ("gpw_sp", "typo_10"), ("gpw_sp", "copypaste_50"),
    ("gpw_sp_low", "clean"), ("gpw_sp_low", "synonym_30"), ("gpw_sp_low", "swap_20"), ("gpw_sp_low", "typo_10"), ("gpw_sp_low", "copypaste_50"),
    ("unigram", "clean"), ("unigram", "synonym_30"), ("unigram", "swap_20"), ("unigram", "typo_10"), ("unigram", "copypaste_50"), ("unigram", "paraphrase"),
    ("kgw", "clean"), ("kgw", "synonym_30"), ("kgw", "swap_20"), ("kgw", "typo_10"), ("kgw", "copypaste_50"), ("kgw", "paraphrase"),
    # SEMSTAMP is BROKEN but "completed" - skip it
    ("semstamp", "clean"), ("semstamp", "synonym_30"), ("semstamp", "swap_20"), ("semstamp", "typo_10"), ("semstamp", "copypaste_50"),
}

COMPLETED_GPT2 = {
    ("unigram", "clean"), ("unigram", "synonym_30"), ("unigram", "swap_20"), ("unigram", "typo_10"), ("unigram", "copypaste_50"), ("unigram", "paraphrase"),
    # SEMSTAMP is BROKEN
    ("semstamp", "clean"), ("semstamp", "synonym_30"), ("semstamp", "swap_20"), ("semstamp", "typo_10"), ("semstamp", "copypaste_50"),
}

COMPLETED_QWEN = set()  # Nothing completed for Qwen

# All watermarkers and attacks (excluding SEMSTAMP since it's broken)
ALL_WATERMARKERS = ["gpw", "gpw_sp", "gpw_sp_low", "gpw_sp_sr", "unigram", "kgw"]
ALL_ATTACKS = ["clean", "synonym_30", "swap_20", "typo_10", "copypaste_50", "paraphrase"]

# Sample counts per model
SAMPLE_COUNTS = {
    "opt-1.3b": 200,
    "gpt2": 200,
    "qwen-7b": 100,  # Fewer samples for larger model
}

# Memory requirements per model
MEMORY_REQS = {
    "opt-1.3b": "32G",
    "gpt2": "24G",
    "qwen-7b": "48G",
}


def get_missing_experiments(model: str):
    """Get list of missing experiments for a model."""
    if model == "opt-1.3b":
        completed = COMPLETED_OPT
    elif model == "gpt2":
        completed = COMPLETED_GPT2
    elif model == "qwen-7b":
        completed = COMPLETED_QWEN
    else:
        return []

    missing = []
    for wm in ALL_WATERMARKERS:
        for attack in ALL_ATTACKS:
            if (wm, attack) not in completed:
                missing.append((wm, attack))

    return missing


def list_missing_experiments():
    """List all missing experiments."""
    print("=" * 70)
    print("MISSING EXPERIMENTS")
    print("=" * 70)

    for model in ["opt-1.3b", "gpt2", "qwen-7b"]:
        missing = get_missing_experiments(model)
        print(f"\n{model.upper()}: {len(missing)} missing")
        for wm, attack in missing:
            print(f"  - {wm} + {attack}")


def submit_experiment(model: str, watermarker: str, attack: str, dry_run: bool = False):
    """Submit a single experiment as a SLURM job."""
    os.makedirs(LOGDIR, exist_ok=True)

    job_name = f"exp_{model}_{watermarker}_{attack}"
    n_samples = SAMPLE_COUNTS.get(model, 200)
    mem = MEMORY_REQS.get(model, "32G")

    log_file = f"{LOGDIR}/{job_name}_%j.out"
    err_file = f"{LOGDIR}/{job_name}_%j.err"

    cmd = f"python {SCRIPT} --model {model} --watermarker {watermarker} --attack {attack} --n_samples {n_samples}"

    sbatch_cmd = [
        "sbatch",
        f"--partition={SLURM_CONFIG['partition']}",
        f"--qos={SLURM_CONFIG['qos']}",
        f"--account={SLURM_CONFIG['account']}",
        f"--gres={SLURM_CONFIG['gres']}",
        f"--time={SLURM_CONFIG['time']}",
        f"--mem={mem}",
        f"--job-name={job_name}",
        f"--output={log_file}",
        f"--error={err_file}",
        "--parsable",
        f"--wrap={ENV_SETUP} && cd {WORKDIR} && {cmd}",
    ]

    if dry_run:
        print(f"[DRY RUN] Would submit: {job_name}")
        print(f"  Command: {cmd}")
        return None

    try:
        result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            job_id = result.stdout.strip()
            print(f"[SUBMITTED] {job_name} -> Job ID: {job_id}")
            return job_id
        else:
            print(f"[ERROR] {job_name}: {result.stderr}")
            return None
    except Exception as e:
        print(f"[ERROR] {job_name}: {e}")
        return None


def submit_model_experiments(model: str, dry_run: bool = False):
    """Submit all missing experiments for a model."""
    missing = get_missing_experiments(model)

    print(f"\n{'=' * 70}")
    print(f"SUBMITTING {model.upper()} EXPERIMENTS ({len(missing)} jobs)")
    print("=" * 70)

    job_ids = []
    for wm, attack in missing:
        job_id = submit_experiment(model, wm, attack, dry_run)
        if job_id:
            job_ids.append(job_id)

    print(f"\nSubmitted {len(job_ids)} jobs")
    return job_ids


def main():
    parser = argparse.ArgumentParser(description="Submit Missing Experiments")
    parser.add_argument("--list", action="store_true", help="List all missing experiments")
    parser.add_argument("--submit", type=str, choices=["opt", "gpt2", "qwen", "all"],
                        help="Submit experiments for a model or all")
    parser.add_argument("--submit-one", nargs=3, metavar=("MODEL", "WATERMARKER", "ATTACK"),
                        help="Submit a single experiment")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually submit, just print")
    args = parser.parse_args()

    if args.list:
        list_missing_experiments()
        return

    if args.submit_one:
        model, watermarker, attack = args.submit_one
        submit_experiment(model, watermarker, attack, args.dry_run)
        return

    if args.submit:
        if args.submit == "opt":
            submit_model_experiments("opt-1.3b", args.dry_run)
        elif args.submit == "gpt2":
            submit_model_experiments("gpt2", args.dry_run)
        elif args.submit == "qwen":
            submit_model_experiments("qwen-7b", args.dry_run)
        elif args.submit == "all":
            submit_model_experiments("opt-1.3b", args.dry_run)
            submit_model_experiments("gpt2", args.dry_run)
            submit_model_experiments("qwen-7b", args.dry_run)
        return

    # Default: list experiments
    list_missing_experiments()


if __name__ == "__main__":
    main()
