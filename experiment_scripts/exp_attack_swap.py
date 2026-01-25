#!/usr/bin/env python3
"""
Experiment: Swap Attack on Watermarked Text

Tests watermark detection after applying word swap attack.
Simple attack - submit second.

Usage:
    python experiment_scripts/exp_attack_swap.py --model opt-1.3b --watermark unigram --num_samples 200
"""

import sys
import os
import time
from datetime import datetime
import importlib.util

# Import from files directly
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Load config module
config_path = os.path.join(script_dir, 'config.py')
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

# Load utils module
utils_path = os.path.join(script_dir, 'utils.py')
spec = importlib.util.spec_from_file_location("utils", utils_path)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

# Load dataset_loader module
dataset_loader_path = os.path.join(script_dir, 'dataset_loader.py')
spec = importlib.util.spec_from_file_location("dataset_loader", dataset_loader_path)
dataset_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_loader)

import torch

# Import all config variables
MODEL_CONFIGS = config.MODEL_CONFIGS
WATERMARK_PARAMS = config.WATERMARK_PARAMS
GENERATION_PARAMS = config.GENERATION_PARAMS
DEVICE = config.DEVICE
DTYPE = config.DTYPE
SWAP_ATTACK_CONFIG = config.SWAP_ATTACK_CONFIG
RESULTS_DIR = config.RESULTS_DIR
SAMPLE_CSV_COLUMNS = config.SAMPLE_CSV_COLUMNS
SUMMARY_CSV_COLUMNS = config.SUMMARY_CSV_COLUMNS

# Import functions
create_experiment_parser = utils.create_experiment_parser
load_model_and_tokenizer = utils.load_model_and_tokenizer
load_watermarker = utils.load_watermarker
load_attack = utils.load_attack
setup_csv_logger = utils.setup_csv_logger
log_sample_result = utils.log_sample_result
save_aggregate_metrics = utils.save_aggregate_metrics
load_c4_dataset = dataset_loader.load_c4_dataset
split_human_ai_data = dataset_loader.split_human_ai_data
load_c4_dataset = dataset_loader.load_c4_dataset
split_human_ai_data = dataset_loader.split_human_ai_data


def main():
    parser = create_experiment_parser("Swap Attack Experiment")
    parser.add_argument('--swap_rate', type=float, default=0.1,
                        help='Word swap rate')
    args = parser.parse_args()
    
    print("=" * 80)
    print("EXPERIMENT: Swap Attack on Watermarked Text")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Watermark: {args.watermark}")
    print(f"Num samples: {args.num_samples}")
    print(f"Swap rate: {args.swap_rate}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Load dataset
    print("\n[1/6] Loading C4 dataset...")
    c4_data = load_c4_dataset(num_samples=args.num_samples, seed=args.seed)
    _, prompts = split_human_ai_data(c4_data, split_ratio=0.0)
    print(f"✓ Loaded {len(prompts)} prompts")
    
    # Load model
    print(f"\n[2/6] Loading model: {args.model}...")
    model_config = MODEL_CONFIGS[args.model]
    model, tokenizer, device = load_model_and_tokenizer(model_config, args.device)
    
    # Load watermarker
    print(f"\n[3/6] Loading watermarker: {args.watermark}...")
    watermark_params = WATERMARK_PARAMS[args.watermark]
    watermarker = load_watermarker(args.watermark, model, tokenizer, watermark_params, device)
    print(f"✓ Watermarker loaded: {watermarker.__class__.__name__}")
    
    # Load attack
    print(f"\n[4/6] Loading attack: Swap...")
    attack_params = {"edit_rate": args.swap_rate}
    attack = load_attack("swap", attack_params)
    print(f"✓ Attack loaded: {attack.__class__.__name__}")
    
    # Setup CSV logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(args.output_dir, 
                              f"exp_attack_swap_{args.model}_{args.watermark}_{timestamp}.csv")
    
    fieldnames = [
        'sample_id', 'prompt', 'original_text', 'attacked_text',
        'z_score_original', 'z_score_attacked',
        'is_watermarked_original', 'is_watermarked_attacked',
        'num_tokens', 'generation_time', 'detection_time_original', 'detection_time_attacked'
    ]
    
    csv_writer, csv_file = setup_csv_logger(output_csv, fieldnames)
    print(f"✓ CSV logger setup: {output_csv}")
    
    # Run experiment
    print(f"\n[5/6] Generating and attacking {len(prompts)} samples...")
    results = []
    
    start_exp_time = time.time()
    
    for idx, prompt in enumerate(prompts):
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx + 1}/{len(prompts)}")
        
        try:
            # Generate watermarked text
            gen_start = time.time()
            original_text = watermarker.generate(prompt, **GENERATION_PARAMS)
            gen_time = time.time() - gen_start
            
            # Detect original
            detect_start = time.time()
            result_original = watermarker.detect(original_text)
            detect_time_original = time.time() - detect_start
            
            # Apply attack
            attacked_text = attack(original_text)
            
            # Detect attacked
            detect_start = time.time()
            result_attacked = watermarker.detect(attacked_text)
            detect_time_attacked = time.time() - detect_start
            
            # Log result
            row = {
                'sample_id': idx,
                'prompt': prompt[:100],
                'original_text': original_text[:200],
                'attacked_text': attacked_text[:200],
                'z_score_original': result_original.z_score,
                'z_score_attacked': result_attacked.z_score,
                'is_watermarked_original': result_original.is_watermarked,
                'is_watermarked_attacked': result_attacked.is_watermarked,
                'num_tokens': result_original.num_tokens_scored,
                'generation_time': gen_time,
                'detection_time_original': detect_time_original,
                'detection_time_attacked': detect_time_attacked,
            }
            
            log_sample_result(csv_writer, row)
            csv_file.flush()
            results.append(row)
            
        except Exception as e:
            print(f"  ✗ Error on sample {idx}: {e}")
            continue
    
    total_exp_time = time.time() - start_exp_time
    
    # Compute aggregate metrics
    print(f"\n[6/6] Computing aggregate metrics...")
    
    original_z_scores = [r['z_score_original'] for r in results]
    attacked_z_scores = [r['z_score_attacked'] for r in results]
    
    metrics = {
        'experiment': 'swap_attack',
        'model': args.model,
        'watermark': args.watermark,
        'swap_rate': args.swap_rate,
        'num_samples': len(results),
        'total_time': total_exp_time,
        
        'original_mean_z_score': sum(original_z_scores) / len(original_z_scores),
        'original_detection_rate': sum(1 for r in results if r['is_watermarked_original']) / len(results),
        
        'attacked_mean_z_score': sum(attacked_z_scores) / len(attacked_z_scores),
        'attacked_detection_rate': sum(1 for r in results if r['is_watermarked_attacked']) / len(results),
        
        'z_score_drop': (sum(original_z_scores) - sum(attacked_z_scores)) / len(results),
        'detection_rate_drop': (sum(1 for r in results if r['is_watermarked_original']) - 
                               sum(1 for r in results if r['is_watermarked_attacked'])) / len(results),
    }
    
    summary_json = output_csv.replace('.csv', '_summary.json')
    save_aggregate_metrics(metrics, summary_json)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Total samples: {metrics['num_samples']}")
    print(f"Total time: {metrics['total_time']:.1f}s")
    print(f"\nOriginal Detection Rate: {metrics['original_detection_rate']:.1%}")
    print(f"Attacked Detection Rate: {metrics['attacked_detection_rate']:.1%}")
    print(f"Detection Rate Drop: {metrics['detection_rate_drop']:.1%}")
    print(f"\nResults saved to:")
    print(f"  - {output_csv}")
    print(f"  - {summary_json}")
    print("=" * 80)
    
    csv_file.close()


if __name__ == "__main__":
    main()
