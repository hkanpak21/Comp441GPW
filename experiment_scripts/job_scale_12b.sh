#!/bin/bash
#SBATCH --job-name=scale_12b
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=1:55:00
#SBATCH --output=/scratch/hkanpak21/Comp441GPW/results/scaling/scale_12b_%j.out
#SBATCH --error=/scratch/hkanpak21/Comp441GPW/results/scaling/scale_12b_%j.err

echo "=========================================="
echo "MODEL SCALING EXPERIMENT - PYTHIA 12B"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo ""

# Activate environment
source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate thor310

# Set offline mode (shared cache is read-only)
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Show GPU info
nvidia-smi

cd /scratch/hkanpak21/Comp441GPW

# Run pythia-12b only with 50 samples for time
python -c "
import os
import sys
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, '/scratch/hkanpak21/Comp441GPW')

from experiment_scripts.exp_model_scaling import *

# Override settings
SHARED_CACHE = '/datasets/NLP/huggingface/hub'
MODEL = ('pythia-12b', 12_000_000_000, f'{SHARED_CACHE}/models--EleutherAI--pythia-12b/snapshots/3fef353ace0849cccae3f4d5b45a4a962217be9d')
NUM_SAMPLES_LOCAL = 50  # Reduced for time

print('Running pythia-12b with 50 samples')
device = get_device()
prompts = load_prompts(NUM_SAMPLES_LOCAL)

model_name, num_params, model_path = MODEL
print(f'\\n{\"=\"*70}')
print(f'MODEL: {model_name} ({num_params/1e9:.2f}B)')
print(f'{\"=\"*70}')

results = run_experiment_for_model(model_name, num_params, model_path, prompts, device)

if results:
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'scaling_pythia_12b.csv', index=False)
    print(f'\\nSaved {len(results)} results')
    print(df.to_string())

print('\\nPythia-12B completed!')
"

echo ""
echo "End: $(date)"
