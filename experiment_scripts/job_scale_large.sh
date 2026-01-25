#!/bin/bash
#SBATCH --job-name=scale_large
#SBATCH --partition=ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=7:55:00
#SBATCH --output=/scratch/hkanpak21/Comp441GPW/results/scaling/scale_large_%j.out
#SBATCH --error=/scratch/hkanpak21/Comp441GPW/results/scaling/scale_large_%j.err

echo "=========================================="
echo "MODEL SCALING EXPERIMENT - LARGE MODELS"
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

# Run large models only (6.9B - 12B)
python -c "
import os
import sys
os.environ['HF_HOME'] = '/datasets/NLP/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/datasets/NLP/huggingface/hub'
sys.path.insert(0, '/scratch/hkanpak21/Comp441GPW')

# Modify the experiment to only run large models
from experiment_scripts.exp_model_scaling import *

# Override to only large models
SHARED_CACHE = '/datasets/NLP/huggingface/hub'
LARGE_MODELS = [
    ('pythia-6.9b', 6_900_000_000, f'{SHARED_CACHE}/models--EleutherAI--pythia-6.9b/snapshots/0b4fc522e9aeb35aeebbc44d05236cb68dd805cd'),
    ('pythia-12b', 12_000_000_000, f'{SHARED_CACHE}/models--EleutherAI--pythia-12b/snapshots/3fef353ace0849cccae3f4d5b45a4a962217be9d'),
]

print('Running large models: 6.9B - 12B (using 4 GPUs)')
device = get_device()
prompts = load_prompts(NUM_SAMPLES)

all_results = []
for model_name, num_params, model_path in LARGE_MODELS:
    print(f'\\n{\"=\"*70}')
    print(f'MODEL: {model_name} ({num_params/1e9:.2f}B)')
    print(f'{\"=\"*70}')
    results = run_experiment_for_model(model_name, num_params, model_path, prompts, device)
    all_results.extend(results)
    
    # Save intermediate
    if results:
        import pandas as pd
        df = pd.DataFrame(all_results)
        df.to_csv(RESULTS_DIR / 'scaling_large_models.csv', index=False)

print('\\nLarge models completed!')
print(f'Total results: {len(all_results)}')
"

echo ""
echo "End: $(date)"
