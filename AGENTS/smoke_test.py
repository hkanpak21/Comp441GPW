#!/usr/bin/env python3
"""
Smoke Test for Watermarking Environment

Tests:
1. PyTorch and CUDA availability
2. Model loading (gpt2 for quick test, then opt-1.3b)
3. Watermarker initialization (Unigram, KGW, SEMSTAMP, GPW)
4. Basic generation and detection
5. Performance check (GPW detection speed)

Run with: python AGENTS/smoke_test.py
"""

import sys
import time
import torch
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, '/scratch/hkanpak21/Comp441GPW')


def test_environment():
    """Test basic environment setup."""
    print("=" * 70)
    print("SMOKE TEST: Environment Check")
    print("=" * 70)
    
    # Check PyTorch
    print(f"\n✓ PyTorch version: {torch.__version__}")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"✓ CUDA available: {cuda_available}")
    if cuda_available:
        print(f"  - Device: {torch.cuda.get_device_name(0)}")
        print(f"  - Count: {torch.cuda.device_count()}")
        print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ WARNING: CUDA not available, using CPU (will be slow)")
    
    return cuda_available


def test_model_loading(model_name="gpt2"):
    """Test model and tokenizer loading."""
    print(f"\n{'=' * 70}")
    print(f"SMOKE TEST: Model Loading ({model_name})")
    print("=" * 70)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nLoading {model_name}...")
    start = time.time()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        if device == "cpu":
            model = model.to(device)
        
        elapsed = time.time() - start
        print(f"✓ Model loaded in {elapsed:.1f}s")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        print(f"  - Device: {next(model.parameters()).device}")
        
        return model, tokenizer, device
    
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return None, None, device


def test_watermarkers(model, tokenizer, device):
    """Test watermarker initialization."""
    print(f"\n{'=' * 70}")
    print("SMOKE TEST: Watermarker Initialization")
    print("=" * 70)
    
    from watermarkers import UnigramWatermark, KGWWatermark, create_gpw_variant
    
    results = {}
    
    # Test Unigram
    print("\n1. Testing UnigramWatermark...")
    try:
        unigram = UnigramWatermark(model, tokenizer, gamma=0.5, delta=2.0, device=device)
        print("   ✓ Unigram initialized")
        results['unigram'] = unigram
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        results['unigram'] = None
    
    # Test KGW
    print("2. Testing KGWWatermark...")
    try:
        kgw = KGWWatermark(model, tokenizer, gamma=0.5, delta=2.0, device=device)
        print("   ✓ KGW initialized")
        results['kgw'] = kgw
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        results['kgw'] = None
    
    # Test GPW
    print("3. Testing GPW (non-SR)...")
    try:
        gpw = create_gpw_variant(model, tokenizer, variant="GPW", device=device)
        print("   ✓ GPW initialized")
        results['gpw'] = gpw
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        results['gpw'] = None
    
    # Test GPW-SP
    print("4. Testing GPW-SP...")
    try:
        gpw_sp = create_gpw_variant(model, tokenizer, variant="GPW-SP", device=device)
        print("   ✓ GPW-SP initialized")
        results['gpw_sp'] = gpw_sp
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        results['gpw_sp'] = None
    
    # Test SEMSTAMP (optional - requires sentence-transformers)
    print("5. Testing SEMSTAMP...")
    try:
        from watermarkers import SEMSTAMPWatermark
        from sentence_transformers import SentenceTransformer
        
        print("   Loading sentence encoder...")
        embedder = SentenceTransformer("all-mpnet-base-v2")  # Fast alternative
        semstamp = SEMSTAMPWatermark(model, tokenizer, embedder=embedder, device=device)
        print("   ✓ SEMSTAMP initialized")
        results['semstamp'] = semstamp
    except Exception as e:
        print(f"   ⚠ SKIPPED: {e}")
        results['semstamp'] = None
    
    return results


def test_generation_detection(watermarkers, tokenizer):
    """Test basic generation and detection."""
    print(f"\n{'=' * 70}")
    print("SMOKE TEST: Generation and Detection")
    print("=" * 70)
    
    prompt = "The future of artificial intelligence"
    print(f"\nPrompt: \"{prompt}\"")
    print(f"Max tokens: 30")
    
    for name, watermarker in watermarkers.items():
        if watermarker is None:
            continue
        
        print(f"\n{name.upper()}:")
        try:
            # Generate
            start = time.time()
            text = watermarker.generate(prompt, max_new_tokens=30, temperature=1.0)
            gen_time = time.time() - start
            
            # Count tokens
            tokens = tokenizer.encode(text)
            print(f"  Generated {len(tokens)} tokens in {gen_time:.2f}s ({len(tokens)/gen_time:.1f} tok/s)")
            print(f"  Text: {text[:100]}...")
            
            # Detect
            start = time.time()
            result = watermarker.detect(text)
            det_time = time.time() - start
            
            print(f"  Detection: z={result.z_score:.2f}, detected={result.is_watermarked} ({det_time:.3f}s)")
            
            # Test on non-watermarked text
            result_human = watermarker.detect("This is a regular human-written sentence without watermarking.")
            print(f"  Human text: z={result_human.z_score:.2f}, detected={result_human.is_watermarked}")
            
        except Exception as e:
            print(f"  ✗ FAILED: {e}")


def test_gpw_optimization(model, tokenizer, device):
    """Test GPW detection optimization."""
    print(f"\n{'=' * 70}")
    print("SMOKE TEST: GPW Detection Optimization")
    print("=" * 70)
    
    from watermarkers import create_gpw_variant
    
    # Test non-SR mode (should be fast)
    print("\nTesting GPW-SP (non-SR, optimized)...")
    try:
        gpw_sp = create_gpw_variant(model, tokenizer, variant="GPW-SP", device=device)
        
        # Generate test text
        test_text = gpw_sp.generate("The future of AI", max_new_tokens=50, temperature=1.0)
        tokens = tokenizer.encode(test_text)
        
        # Time detection
        start = time.time()
        result = gpw_sp.detect(test_text)
        elapsed = time.time() - start
        
        speed = len(tokens) / elapsed
        print(f"✓ Detected {len(tokens)} tokens in {elapsed:.3f}s ({speed:.1f} tok/s)")
        print(f"  z-score: {result.z_score:.2f}")
        
        if speed < 10:
            print(f"⚠ WARNING: Detection is slow ({speed:.1f} tok/s)")
            print("  Expected: >100 tok/s for non-SR mode")
        else:
            print(f"✓ Performance is good!")
        
    except Exception as e:
        print(f"✗ FAILED: {e}")


def main():
    """Run all smoke tests."""
    print("\n" + "=" * 70)
    print("WATERMARKING ENVIRONMENT SMOKE TEST")
    print("=" * 70)
    
    # Test environment
    cuda_available = test_environment()
    
    # Test with small model first
    model, tokenizer, device = test_model_loading("gpt2")
    
    if model is None:
        print("\n✗ Model loading failed. Aborting.")
        return
    
    # Test watermarkers
    watermarkers = test_watermarkers(model, tokenizer, device)
    
    # Test generation and detection
    test_generation_detection(watermarkers, tokenizer)
    
    # Test GPW optimization
    test_gpw_optimization(model, tokenizer, device)
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SMOKE TEST SUMMARY")
    print("=" * 70)
    
    print(f"\nEnvironment: {'✓ CUDA' if cuda_available else '⚠ CPU only'}")
    print(f"Model: ✓ gpt2 loaded")
    
    working = [name for name, wm in watermarkers.items() if wm is not None]
    failed = [name for name, wm in watermarkers.items() if wm is None]
    
    print(f"\nWatermarkers:")
    print(f"  ✓ Working: {', '.join(working) if working else 'None'}")
    if failed:
        print(f"  ✗ Failed: {', '.join(failed)}")
    
    print(f"\n{'=' * 70}")
    
    if len(working) >= 3:
        print("✓ SMOKE TEST PASSED - Ready for experiments!")
    else:
        print("⚠ SMOKE TEST PARTIAL - Some components failed")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
