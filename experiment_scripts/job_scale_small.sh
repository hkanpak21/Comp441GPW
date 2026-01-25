#!/bin/bash
#SBATCH --job-name=scale_small
#SBATCH --partition=t4_ai
#SBATCH --qos=comx29
#SBATCH --account=comx29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:55:00
#SBATCH --output=/scratch/hkanpak21/Comp441GPW/results/scaling/scale_small_%j.out
#SBATCH --error=/scratch/hkanpak21/Comp441GPW/results/scaling/scale_small_%j.err

echo "=========================================="
echo "MODEL SCALING EXPERIMENT - SMALL MODELS"
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

# Run small models only (70M - 2.8B)
python -c "
import os
import sys
os.environ['HF_HOME'] = '/datasets/NLP/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/datasets/NLP/huggingface/hub'
sys.path.insert(0, '/scratch/hkanpak21/Comp441GPW')

# Modify the experiment to only run small models
from experiment_scripts.exp_model_scaling import *

# Override to only small models
SHARED_CACHE = '/datasets/NLP/huggingface/hub'
SMALL_MODELS = [
    ('pythia-70m', 70_000_000, f'{SHARED_CACHE}/models--EleutherAI--pythia-70m/snapshots/2ab25ed47af79376eed2baaf8bbb7a192a0c73ff'),
    ('pythia-160m', 160_000_000, f'{SHARED_CACHE}/models--EleutherAI--pythia-160m/snapshots/26b94a336e5683a217752ab7ae4bf3cbe5661365'),
    ('pythia-410m', 410_000_000, f'{SHARED_CACHE}/models--EleutherAI--pythia-410m/snapshots/33fc2fdda3ea75631397cc28cec556f0ef401ae7'),
    ('pythia-1b', 1_000_000_000, f'{SHARED_CACHE}/models--EleutherAI--pythia-1b/snapshots/4c96d81536e92f85f8c2a45b5397057ce83a8636'),
    ('pythia-1.4b', 1_400_000_000, f'{SHARED_CACHE}/models--EleutherAI--pythia-1.4b/snapshots/8f0af839162ddb466006494a08733ee9cfa2d338'),
    ('pythia-2.8b', 2_800_000_000, f'{SHARED_CACHE}/models--EleutherAI--pythia-2.8b/snapshots/f20536c52a97faea73b8997cc789bd913853f14a'),
]

print('Running small models: 70M - 2.8B')
device = get_device()
prompts = load_prompts(NUM_SAMPLES)

all_results = []
for model_name, num_params, model_path in SMALL_MODELS:
    print(f'\\n{\"=\"*70}')
    print(f'MODEL: {model_name} ({num_params/1e9:.2f}B)')
    print(f'{\"=\"*70}')
    results = run_experiment_for_model(model_name, num_params, model_path, prompts, device)
    all_results.extend(results)
    
    # Save intermediate
    if results:
        import pandas as pd
        df = pd.DataFrame(all_results)
        df.to_csv(RESULTS_DIR / 'scaling_small_models.csv', index=False)

print('\\nSmall models completed!')
print(f'Total results: {len(all_results)}')
"

echo ""
echo "End: $(date)"
