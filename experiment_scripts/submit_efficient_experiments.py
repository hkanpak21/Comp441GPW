#!/usr/bin/env python3
"""
Submit Efficient Experiments

Two-phase experiment pipeline:
1. Phase 1: Generate texts ONCE for each watermarker (including baseline)
2. Phase 2: Apply all attacks to saved texts and run detection

Usage:
    python submit_efficient_experiments.py --list                    # List what will be submitted
    python submit_efficient_experiments.py --phase1 opt-1.3b         # Submit text generation jobs
    python submit_efficient_experiments.py --phase2 opt-1.3b         # Submit attack jobs (after phase 1)
    python submit_efficient_experiments.py --all opt-1.3b            # Submit both phases
"""

import os
import sys
import glob
import subprocess
import argparse
from datetime import datetime

WORKDIR = "/scratch/hkanpak21/Comp441GPW"
LOGDIR = f"{WORKDIR}/logs/efficient"
GENERATED_DIR = f"{WORKDIR}/generated_texts"

# SLURM configuration
SLURM_CONFIG = {
    "partition": "t4_ai",
    "qos": "comx29",
    "account": "comx29",
    "gres": "gpu:1",
    "time": "1:55:00",
}

ENV_SETUP = "export LD_LIBRARY_PATH=/home/hkanpak21/.conda/envs/thor310/lib:$LD_LIBRARY_PATH && source ~/.bashrc && conda activate thor310"

# All watermarkers including baseline
ALL_WATERMARKERS = ["none", "gpw", "gpw_sp", "gpw_sp_low", "gpw_sp_sr", "unigram", "kgw"]

# All attacks (excluding paraphrase initially due to memory issues)
STANDARD_ATTACKS = ["clean", "synonym_30", "swap_20", "typo_10", "copypaste_50"]

# Memory requirements
MEMORY_REQS = {
    "opt-1.3b": "32G",
    "gpt2": "24G",
    "qwen-7b": "48G",
}

SAMPLE_COUNTS = {
    "opt-1.3b": 200,
    "gpt2": 200,
    "qwen-7b": 100,
}


def submit_job(job_name: str, cmd: str, mem: str, dry_run: bool = False) -> str:
    """Submit a SLURM job."""
    os.makedirs(LOGDIR, exist_ok=True)

    log_file = f"{LOGDIR}/{job_name}_%j.out"
    err_file = f"{LOGDIR}/{job_name}_%j.err"

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
        print(f"[DRY RUN] {job_name}: {cmd[:80]}...")
        return "DRY"

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


def submit_phase1(model: str, dry_run: bool = False):
    """Submit Phase 1: Text generation jobs."""
    print("\n" + "=" * 70)
    print(f"PHASE 1: TEXT GENERATION ({model})")
    print("=" * 70)

    mem = MEMORY_REQS.get(model, "32G")
    n_samples = SAMPLE_COUNTS.get(model, 200)

    job_ids = []
    for wm in ALL_WATERMARKERS:
        job_name = f"gen_{model}_{wm}"
        cmd = f"python experiment_scripts/generate_texts.py --model {model} --watermarker {wm} --n_samples {n_samples}"
        job_id = submit_job(job_name, cmd, mem, dry_run)
        if job_id:
            job_ids.append((wm, job_id))

    print(f"\nSubmitted {len(job_ids)} generation jobs")
    return job_ids


def submit_phase2(model: str, dry_run: bool = False):
    """Submit Phase 2: Attack jobs on generated texts."""
    print("\n" + "=" * 70)
    print(f"PHASE 2: ATTACK EXPERIMENTS ({model})")
    print("=" * 70)

    mem = MEMORY_REQS.get(model, "32G")

    # Find generated text files
    pattern = f"{GENERATED_DIR}/{model}_*.pkl"
    generated_files = glob.glob(pattern)

    if not generated_files:
        print(f"No generated text files found matching: {pattern}")
        print("Run Phase 1 first!")
        return []

    print(f"Found {len(generated_files)} generated text files")

    job_ids = []
    attacks_str = " ".join(STANDARD_ATTACKS)

    for gen_file in generated_files:
        # Extract watermarker name from filename
        basename = os.path.basename(gen_file)
        parts = basename.replace(".pkl", "").split("_")
        wm = parts[1] if len(parts) > 1 else "unknown"

        # For baseline (none), we need to run detection with each watermarker
        if wm == "none":
            # Run FPR tests with each detector
            for detector in ["gpw", "gpw_sp", "unigram", "kgw"]:
                job_name = f"attack_{model}_baseline_{detector}"
                cmd = f"python experiment_scripts/run_attacks_on_texts.py --input {gen_file} --attacks {attacks_str} --detector {detector}"
                job_id = submit_job(job_name, cmd, mem, dry_run)
                if job_id:
                    job_ids.append((f"baseline_{detector}", job_id))
        else:
            job_name = f"attack_{model}_{wm}"
            cmd = f"python experiment_scripts/run_attacks_on_texts.py --input {gen_file} --attacks {attacks_str}"
            job_id = submit_job(job_name, cmd, mem, dry_run)
            if job_id:
                job_ids.append((wm, job_id))

    print(f"\nSubmitted {len(job_ids)} attack jobs")
    return job_ids


def list_experiments(model: str):
    """List what experiments will be submitted."""
    print("\n" + "=" * 70)
    print(f"EXPERIMENT PLAN FOR {model.upper()}")
    print("=" * 70)

    print("\nPHASE 1: Text Generation")
    print("-" * 40)
    for wm in ALL_WATERMARKERS:
        label = "baseline (no watermark)" if wm == "none" else wm
        print(f"  - Generate {model} + {label}")

    print(f"\nTotal Phase 1 jobs: {len(ALL_WATERMARKERS)}")

    print("\nPHASE 2: Attack Experiments")
    print("-" * 40)
    for wm in ALL_WATERMARKERS:
        if wm == "none":
            print(f"  - Baseline FPR tests:")
            for detector in ["gpw", "gpw_sp", "unigram", "kgw"]:
                print(f"      * Detect with {detector}: {len(STANDARD_ATTACKS)} attacks")
        else:
            print(f"  - {wm}: {len(STANDARD_ATTACKS)} attacks")

    n_phase2 = (len(ALL_WATERMARKERS) - 1) + 4  # Non-baseline + 4 FPR tests
    print(f"\nTotal Phase 2 jobs: {n_phase2}")
    print(f"\nTotal jobs: {len(ALL_WATERMARKERS) + n_phase2}")


def main():
    parser = argparse.ArgumentParser(description="Submit Efficient Experiments")
    parser.add_argument("--list", type=str, metavar="MODEL", help="List experiments for model")
    parser.add_argument("--phase1", type=str, metavar="MODEL", help="Submit Phase 1 for model")
    parser.add_argument("--phase2", type=str, metavar="MODEL", help="Submit Phase 2 for model")
    parser.add_argument("--all", type=str, metavar="MODEL", help="Submit all phases for model")
    parser.add_argument("--dry-run", action="store_true", help="Don't submit, just print")
    args = parser.parse_args()

    if args.list:
        list_experiments(args.list)
    elif args.phase1:
        submit_phase1(args.phase1, args.dry_run)
    elif args.phase2:
        submit_phase2(args.phase2, args.dry_run)
    elif args.all:
        submit_phase1(args.all, args.dry_run)
        print("\nNote: Wait for Phase 1 to complete before Phase 2 can use the generated texts!")
    else:
        print("Use --list MODEL, --phase1 MODEL, --phase2 MODEL, or --all MODEL")
        print("Models: opt-1.3b, gpt2, qwen-7b")


if __name__ == "__main__":
    main()
