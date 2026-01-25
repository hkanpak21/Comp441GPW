#!/usr/bin/env python3
"""
Experiment: Detection without Attack - AI Generated without Watermark

Tests detection performance on unwatermarked AI-generated text (baseline FPR for AI text).

Expected: Low z-scores, similar to human text
"""

import sys
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/scratch/hkanpak21/Comp441GPW')

from exp_utils import *
from config import *


def main():
    """Run unwatermarked AI text detection experiment."""
    exp_name = "no_attack_unwatermarked"
    print_experiment_header(exp_name)
    
    # Set seed for reproducibility
    set_seed(RANDOM_SEED)
    
    # Load dataset
    print("Step 1: Loading C4 dataset...")
    c4_data = load_c4_dataset()
    prompts = [item["prompt"] for item in c4_data]
    print(f"✓ Loaded {len(prompts)} prompts")
    
    # Load model
    print("\nStep 2: Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    
    # Initialize watermarkers (for detection only)
    print("\nStep 3: Initializing watermarkers...")
    watermarkers = initialize_watermarkers(model, tokenizer)
    
    # Generate UNWATERMARKED texts (vanilla model)
    print("\nStep 4: Generating unwatermarked texts...")
    print("Using vanilla model (no watermark)...")
    
    generated_texts = []
    generation_times = []
    
    for i, prompt in enumerate(prompts):
        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{len(prompts)}...")
        
        gen_start = time.time()
        try:
            # Vanilla generation (no watermark)
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            outputs = model.generate(
                **inputs,
                max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                temperature=GENERATION_CONFIG["temperature"],
                top_k=GENERATION_CONFIG["top_k"],
                top_p=GENERATION_CONFIG["top_p"],
                do_sample=GENERATION_CONFIG["do_sample"],
            )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            gen_time = time.time() - gen_start
            
            generated_texts.append(text)
            generation_times.append(gen_time)
        except Exception as e:
            print(f"  ⚠ Generation failed for sample {i}: {e}")
            generated_texts.append(prompt)
            generation_times.append(0.0)
    
    print(f"✓ Generated {len(generated_texts)} unwatermarked texts")
    print(f"  Mean generation time: {np.mean(generation_times):.2f}s")
    
    # Storage for results
    all_samples = []
    all_summary = []
    
    start_time = time.time()
    
    # Run detection with each watermarker
    for wm_name, watermarker in watermarkers.items():
        print(f"\n{'=' * 80}")
        print(f"Testing Watermarker: {wm_name.upper()}")
        print("=" * 80)
        
        wm_start = time.time()
        
        # Detect on unwatermarked texts
        print(f"Detecting on unwatermarked texts...")
        z_scores = []
        detection_times = []
        
        for i, text in enumerate(generated_texts):
            if (i + 1) % 20 == 0:
                print(f"  Detected {i + 1}/{len(generated_texts)}...")
            
            det_start = time.time()
            try:
                result = watermarker.detect(text)
                det_time = time.time() - det_start
                
                z_scores.append(result.z_score)
                detection_times.append(det_time)
                
                # Save per-sample result
                sample_record = {
                    "sample_id": i,
                    "prompt": prompts[i][:100],
                    "generated_text": text[:200],
                    "watermarker": wm_name,
                    "variant": watermarker.get_config().get("variant", wm_name),
                    "z_score": result.z_score,
                    "p_value": result.p_value,
                    "is_detected": result.is_watermarked,
                    "num_tokens": result.num_tokens_scored,
                    "green_fraction": result.green_fraction,
                    "attack": "none",
                    "attack_params": "{}",
                    "generation_time": generation_times[i],
                    "detection_time": det_time,
                }
                all_samples.append(sample_record)
                
            except Exception as e:
                print(f"  ⚠ Detection failed for sample {i}: {e}")
                z_scores.append(0.0)
                detection_times.append(0.0)
        
        print(f"✓ Detected {len(z_scores)} texts")
        print(f"  Mean z-score: {np.mean(z_scores):.2f} ± {np.std(z_scores):.2f}")
        print(f"  False positive rate: {np.mean([z > 4.0 for z in z_scores]) * 100:.1f}%")
        print(f"  Mean detection time: {np.mean(detection_times):.3f}s")
        
        wm_elapsed = time.time() - wm_start
        
        # Compute summary metrics
        summary_record = {
            "watermarker": wm_name,
            "variant": watermarker.get_config().get("variant", wm_name),
            "num_samples": len(z_scores),
            "mean_z_score": float(np.mean(z_scores)),
            "std_z_score": float(np.std(z_scores)),
            "median_z_score": float(np.median(z_scores)),
            "tpr": None,  # This is unwatermarked (negative samples)
            "fpr": float(np.mean([z > 4.0 for z in z_scores])),
            "auc_roc": None,
            "tpr_at_fpr_1pct": None,
            "tpr_at_fpr_5pct": None,
            "mean_detection_time": float(np.mean(detection_times)),
            "total_time": float(wm_elapsed),
        }
        all_summary.append(summary_record)
    
    total_time = time.time() - start_time
    
    # Save results
    print(f"\n{'=' * 80}")
    print("Saving results...")
    print("=" * 80)
    
    samples_file = os.path.join(RESULTS_DIR, f"{exp_name}_samples.csv")
    summary_file = os.path.join(RESULTS_DIR, f"{exp_name}_summary.csv")
    
    save_samples_csv(samples_file, all_samples)
    save_summary_csv(summary_file, all_summary)
    
    # Print summary
    print_experiment_summary(
        num_samples=len(prompts),
        watermarkers=list(watermarkers.keys()),
        elapsed_time=total_time
    )


if __name__ == "__main__":
    main()
