#!/usr/bin/env python3
"""
Check Experiment Status

Monitors submitted experiments and collects results.

Usage:
    python check_experiment_status.py --jobs          # Check SLURM job status
    python check_experiment_status.py --results       # Check available results
    python check_experiment_status.py --merge         # Merge all results into unified CSV
    python check_experiment_status.py --failed        # List failed jobs
"""

import os
import sys
import subprocess
import json
import csv
import glob
from datetime import datetime

WORKDIR = "/scratch/hkanpak21/Comp441GPW"
RESULTS_DIR = f"{WORKDIR}/results"
LOGS_DIR = f"{WORKDIR}/logs/experiments"


def check_slurm_jobs():
    """Check status of running SLURM jobs."""
    print("=" * 70)
    print("SLURM JOB STATUS")
    print("=" * 70)

    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", "hkanpak21"), "-o", "%.8i %.9P %.30j %.8u %.2t %.10M %.6D %R"],
            capture_output=True, text=True
        )
        if result.stdout.strip():
            # Filter for our experiment jobs
            lines = result.stdout.strip().split('\n')
            exp_jobs = [l for l in lines if 'exp_' in l or 'JOBID' in l]
            if exp_jobs:
                for line in exp_jobs:
                    print(line)
            else:
                print("No experiment jobs currently running")
        else:
            print("No jobs currently running")
    except Exception as e:
        print(f"Error checking jobs: {e}")


def check_results():
    """Check available experiment results."""
    print("=" * 70)
    print("AVAILABLE RESULTS")
    print("=" * 70)

    # Find all summary JSON files
    pattern = f"{RESULTS_DIR}/exp_*_summary.json"
    summary_files = sorted(glob.glob(pattern))

    if not summary_files:
        print("No experiment results found yet")
        return

    results_by_model = {}
    for sf in summary_files:
        try:
            with open(sf, 'r') as f:
                data = json.load(f)
            model = data.get("model", "unknown")
            wm = data.get("watermarker", "unknown")
            attack = data.get("attack", "unknown")
            det_rate = data.get("detection_rate", 0)
            mean_z = data.get("mean_z_score", 0)
            n_samples = data.get("n_samples", 0)

            if model not in results_by_model:
                results_by_model[model] = []
            results_by_model[model].append({
                "watermarker": wm,
                "attack": attack,
                "detection_rate": det_rate,
                "mean_z_score": mean_z,
                "n_samples": n_samples,
            })
        except Exception as e:
            print(f"Error reading {sf}: {e}")

    for model, results in sorted(results_by_model.items()):
        print(f"\n{model.upper()}:")
        print(f"{'Watermarker':<15} {'Attack':<15} {'Det%':<8} {'Z-Score':<10} {'Samples':<8}")
        print("-" * 60)
        for r in sorted(results, key=lambda x: (x["watermarker"], x["attack"])):
            print(f"{r['watermarker']:<15} {r['attack']:<15} {r['detection_rate']:<8.1f} {r['mean_z_score']:<10.2f} {r['n_samples']:<8}")


def check_failed_jobs():
    """Check for failed experiments by looking at error logs."""
    print("=" * 70)
    print("FAILED EXPERIMENTS")
    print("=" * 70)

    if not os.path.exists(LOGS_DIR):
        print("No log directory found")
        return

    err_files = glob.glob(f"{LOGS_DIR}/*.err")
    failed = []

    for ef in err_files:
        try:
            with open(ef, 'r') as f:
                content = f.read()
            if content.strip() and ('Error' in content or 'error' in content or 'Traceback' in content):
                job_name = os.path.basename(ef).replace('.err', '')
                failed.append((job_name, content[:200]))
        except:
            pass

    if not failed:
        print("No failed jobs found")
    else:
        print(f"Found {len(failed)} potentially failed jobs:")
        for name, snippet in failed:
            print(f"\n  {name}:")
            print(f"    {snippet[:100]}...")


def merge_results():
    """Merge all experiment results into unified CSV."""
    print("=" * 70)
    print("MERGING RESULTS")
    print("=" * 70)

    # Find all summary JSON files
    pattern = f"{RESULTS_DIR}/exp_*_summary.json"
    summary_files = sorted(glob.glob(pattern))

    if not summary_files:
        print("No experiment results to merge")
        return

    results = []
    for sf in summary_files:
        try:
            with open(sf, 'r') as f:
                data = json.load(f)
            results.append({
                "Model": data.get("model", "").upper().replace("-", " ").replace("1.3B", "-1.3B").replace("OPT 1.3B", "OPT-1.3B"),
                "Watermarker": data.get("watermarker", "").upper().replace("_", "-"),
                "Variant": data.get("watermarker", ""),
                "Alpha": 3.0 if "gpw" in data.get("watermarker", "") else 0.5,
                "Omega": 50.0 if "gpw" in data.get("watermarker", "") and "low" not in data.get("watermarker", "") else 2.0,
                "Z_Threshold": 4.0,
                "Attack": data.get("attack", ""),
                "Detection_Rate": round(data.get("detection_rate", 0), 1),
                "Detection_Count": int(data.get("n_samples", 0) * data.get("detection_rate", 0) / 100),
                "Total_Samples": data.get("n_samples", 0),
                "Mean_Z_Score": round(data.get("mean_z_score", 0), 2),
                "Perplexity": round(data.get("mean_perplexity", -1), 2) if data.get("mean_perplexity", -1) > 0 else "-",
                "Notes": f"From {os.path.basename(sf)}",
            })
        except Exception as e:
            print(f"Error reading {sf}: {e}")

    if not results:
        print("No valid results to merge")
        return

    # Write merged CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{RESULTS_DIR}/merged_results_{timestamp}.csv"

    fieldnames = ["Model", "Watermarker", "Variant", "Alpha", "Omega", "Z_Threshold",
                  "Attack", "Detection_Rate", "Detection_Count", "Total_Samples",
                  "Mean_Z_Score", "Perplexity", "Notes"]

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Merged {len(results)} results to: {output_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check Experiment Status")
    parser.add_argument("--jobs", action="store_true", help="Check SLURM job status")
    parser.add_argument("--results", action="store_true", help="Check available results")
    parser.add_argument("--merge", action="store_true", help="Merge results into unified CSV")
    parser.add_argument("--failed", action="store_true", help="List failed jobs")
    args = parser.parse_args()

    if args.jobs:
        check_slurm_jobs()
    elif args.results:
        check_results()
    elif args.merge:
        merge_results()
    elif args.failed:
        check_failed_jobs()
    else:
        check_slurm_jobs()
        check_results()


if __name__ == "__main__":
    main()
