#!/usr/bin/env python3
"""
Functionality Test Script

Tests all watermarkers and attacks to verify they work correctly.
Run this before submitting batch experiments.

Usage:
    python test_functionality.py --test all
    python test_functionality.py --test watermarkers
    python test_functionality.py --test attacks
"""

import os
import sys
import argparse
import torch
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test results storage
TEST_RESULTS = {}

def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def print_result(name: str, success: bool, message: str = ""):
    status = "PASS" if success else "FAIL"
    color = "\033[92m" if success else "\033[91m"
    reset = "\033[0m"
    print(f"  [{color}{status}{reset}] {name}: {message}")
    TEST_RESULTS[name] = {"success": success, "message": message}

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def test_imports():
    """Test all required imports work."""
    print_header("Testing Imports")

    tests = [
        ("torch", lambda: __import__("torch")),
        ("transformers", lambda: __import__("transformers")),
        ("sentence_transformers", lambda: __import__("sentence_transformers")),
        ("nltk", lambda: __import__("nltk")),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
            print_result(f"import_{name}", True, "imported successfully")
        except Exception as e:
            print_result(f"import_{name}", False, str(e))

    # Test internal imports
    internal_imports = [
        ("watermarkers.UnigramWatermark", lambda: __import__("watermarkers", fromlist=["UnigramWatermark"]).UnigramWatermark),
        ("watermarkers.KGWWatermark", lambda: __import__("watermarkers", fromlist=["KGWWatermark"]).KGWWatermark),
        ("watermarkers.gpw.GPWWatermark", lambda: __import__("watermarkers.gpw", fromlist=["GPWWatermark"]).GPWWatermark),
        ("watermarkers.semstamp.SEMSTAMPWatermark", lambda: __import__("watermarkers.semstamp", fromlist=["SEMSTAMPWatermark"]).SEMSTAMPWatermark),
        ("attacks.SynonymAttack", lambda: __import__("attacks", fromlist=["SynonymAttack"]).SynonymAttack),
        ("attacks.SwapAttack", lambda: __import__("attacks", fromlist=["SwapAttack"]).SwapAttack),
        ("attacks.TypoAttack", lambda: __import__("attacks", fromlist=["TypoAttack"]).TypoAttack),
        ("attacks.CopyPasteAttack", lambda: __import__("attacks", fromlist=["CopyPasteAttack"]).CopyPasteAttack),
        ("attacks.paraphrase.PegasusAttack", lambda: __import__("attacks.paraphrase", fromlist=["PegasusAttack"]).PegasusAttack),
        ("data_loaders.load_c4", lambda: __import__("data_loaders", fromlist=["load_c4"]).load_c4),
    ]

    for name, test_fn in internal_imports:
        try:
            test_fn()
            print_result(f"import_{name}", True, "imported successfully")
        except Exception as e:
            print_result(f"import_{name}", False, str(e))


def test_model_loading(model_name: str = "gpt2"):
    """Test model loading."""
    print_header(f"Testing Model Loading: {model_name}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print_result("tokenizer_load", True, f"loaded {model_name} tokenizer")
    except Exception as e:
        print_result("tokenizer_load", False, str(e))
        return None, None

    try:
        device = get_device()
        dtype = torch.float32 if model_name == "gpt2" else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print_result("model_load", True, f"loaded {model_name} on {device}")
        return model, tokenizer
    except Exception as e:
        print_result("model_load", False, str(e))
        return None, None


def test_watermarkers(model, tokenizer):
    """Test all watermarker implementations."""
    print_header("Testing Watermarkers")

    if model is None:
        print_result("watermarkers", False, "No model loaded")
        return

    device = get_device()
    test_prompt = "The future of artificial intelligence is"

    # Test Unigram
    try:
        from watermarkers import UnigramWatermark
        wm = UnigramWatermark(
            model=model, tokenizer=tokenizer,
            gamma=0.5, delta=2.0, z_threshold=4.0, device=device
        )
        text = wm.generate(test_prompt, max_new_tokens=50)
        result = wm.detect(text)
        print_result("unigram_generate", True, f"generated {len(text.split())} words")
        print_result("unigram_detect", True, f"z={result.z_score:.2f}, detected={result.is_watermarked}")
    except Exception as e:
        print_result("unigram", False, str(e))

    # Test KGW
    try:
        from watermarkers import KGWWatermark
        wm = KGWWatermark(
            model=model, tokenizer=tokenizer,
            gamma=0.5, delta=2.0, z_threshold=4.0,
            seeding_scheme="simple_1", device=device
        )
        text = wm.generate(test_prompt, max_new_tokens=50)
        result = wm.detect(text)
        print_result("kgw_generate", True, f"generated {len(text.split())} words")
        print_result("kgw_detect", True, f"z={result.z_score:.2f}, detected={result.is_watermarked}")
    except Exception as e:
        print_result("kgw", False, str(e))

    # Test GPW
    try:
        from watermarkers.gpw import GPWWatermark, GPWConfig, SRConfig
        import hashlib

        gpw_cfg = GPWConfig(alpha=3.0, omega=50.0, salted=False)
        sr_cfg = SRConfig(enabled=False)
        wm = GPWWatermark(
            model=model, tokenizer=tokenizer,
            gpw_cfg=gpw_cfg, sr_cfg=sr_cfg,
            hash_key=hashlib.sha256(b"test").digest(),
            device=device
        )
        wm.z_threshold = 4.0
        text = wm.generate(test_prompt, max_new_tokens=50)
        result = wm.detect(text)
        print_result("gpw_generate", True, f"generated {len(text.split())} words")
        print_result("gpw_detect", True, f"z={result.z_score:.2f}, detected={result.is_watermarked}")
    except Exception as e:
        print_result("gpw", False, str(e))

    # Test GPW-SP
    try:
        from watermarkers.gpw import GPWWatermark, GPWConfig, SRConfig
        import hashlib

        gpw_cfg = GPWConfig(alpha=3.0, omega=50.0, salted=True, ctx_mode="ngram", ngram=4)
        sr_cfg = SRConfig(enabled=False)
        wm = GPWWatermark(
            model=model, tokenizer=tokenizer,
            gpw_cfg=gpw_cfg, sr_cfg=sr_cfg,
            hash_key=hashlib.sha256(b"test").digest(),
            device=device
        )
        wm.z_threshold = 4.0
        text = wm.generate(test_prompt, max_new_tokens=50)
        result = wm.detect(text)
        print_result("gpw_sp_generate", True, f"generated {len(text.split())} words")
        print_result("gpw_sp_detect", True, f"z={result.z_score:.2f}, detected={result.is_watermarked}")
    except Exception as e:
        print_result("gpw_sp", False, str(e))

    # Test GPW-SP+SR
    try:
        from watermarkers.gpw import GPWWatermark, GPWConfig, SRConfig
        import hashlib

        gpw_cfg = GPWConfig(alpha=3.0, omega=50.0, salted=True, ctx_mode="ngram", ngram=4)
        sr_cfg = SRConfig(enabled=True, lambda_couple=0.1, rank=16)
        wm = GPWWatermark(
            model=model, tokenizer=tokenizer,
            gpw_cfg=gpw_cfg, sr_cfg=sr_cfg,
            hash_key=hashlib.sha256(b"test").digest(),
            device=device
        )
        wm.z_threshold = 4.0
        text = wm.generate(test_prompt, max_new_tokens=50)
        result = wm.detect(text)
        print_result("gpw_sp_sr_generate", True, f"generated {len(text.split())} words")
        print_result("gpw_sp_sr_detect", True, f"z={result.z_score:.2f}, detected={result.is_watermarked}")
    except Exception as e:
        print_result("gpw_sp_sr", False, str(e))

    # Test SEMSTAMP
    try:
        from watermarkers.semstamp import SEMSTAMPWatermark
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer("all-mpnet-base-v2")
        embedder.to(device)

        wm = SEMSTAMPWatermark(
            model=model, tokenizer=tokenizer,
            embedder=embedder,
            lsh_dim=3, margin=0.0, z_threshold=4.0,
            max_rejections=2, device=device
        )
        text = wm.generate(test_prompt, max_new_tokens=50)
        result = wm.detect(text)
        print_result("semstamp_generate", True, f"generated {len(text.split())} words")
        print_result("semstamp_detect", True, f"z={result.z_score:.2f}, detected={result.is_watermarked}")
    except Exception as e:
        print_result("semstamp", False, str(e))


def test_attacks():
    """Test all attack implementations."""
    print_header("Testing Attacks")

    test_text = "The quick brown fox jumps over the lazy dog. This is a test sentence for watermark attacks. We need enough text to properly test the attack mechanisms."
    human_text = "Human written text that is different from the watermarked text. This will be used for copy paste attacks."

    # Test Synonym Attack
    try:
        from attacks import SynonymAttack
        attack = SynonymAttack(edit_rate=0.3)
        result = attack.attack(test_text)
        attacked_text = result.attacked_text if hasattr(result, 'attacked_text') else str(result)
        print_result("synonym_attack", True, f"changed {len(test_text) - len(attacked_text)} chars")
    except Exception as e:
        print_result("synonym_attack", False, str(e))

    # Test Swap Attack
    try:
        from attacks import SwapAttack
        attack = SwapAttack(edit_rate=0.2)
        result = attack.attack(test_text)
        attacked_text = result.attacked_text if hasattr(result, 'attacked_text') else str(result)
        print_result("swap_attack", True, f"swapped words successfully")
    except Exception as e:
        print_result("swap_attack", False, str(e))

    # Test Typo Attack
    try:
        from attacks import TypoAttack
        attack = TypoAttack(edit_rate=0.1)
        result = attack.attack(test_text)
        attacked_text = result.attacked_text if hasattr(result, 'attacked_text') else str(result)
        print_result("typo_attack", True, f"introduced typos successfully")
    except Exception as e:
        print_result("typo_attack", False, str(e))

    # Test CopyPaste Attack
    try:
        from attacks import CopyPasteAttack
        attack = CopyPasteAttack(n_segments=3, watermark_ratio=0.5)
        result = attack.attack(test_text, human_text=human_text)
        attacked_text = result.attacked_text if hasattr(result, 'attacked_text') else str(result)
        print_result("copypaste_attack", True, f"mixed text: {len(attacked_text)} chars")
    except Exception as e:
        print_result("copypaste_attack", False, str(e))

    # Test Paraphrase Attack (Pegasus)
    try:
        from attacks.paraphrase import PegasusAttack
        attack = PegasusAttack()
        # Note: This loads a large model, may be slow
        result = attack.attack(test_text[:100])  # Short text for speed
        attacked_text = result.attacked_text if hasattr(result, 'attacked_text') else str(result)
        print_result("paraphrase_attack", True, f"paraphrased successfully")
    except Exception as e:
        print_result("paraphrase_attack", False, str(e))


def test_dataset():
    """Test dataset loading."""
    print_header("Testing Dataset Loading")

    try:
        from data_loaders import load_c4
        data = load_c4(num_samples=5)
        print_result("c4_dataset", True, f"loaded {len(data)} samples")
        if data:
            sample = data[0]
            has_prompt = 'prompt' in sample
            has_text = 'text' in sample
            print_result("c4_format", has_prompt and has_text,
                        f"prompt={has_prompt}, text={has_text}")
    except Exception as e:
        print_result("c4_dataset", False, str(e))


def print_summary():
    """Print test summary."""
    print_header("TEST SUMMARY")

    passed = sum(1 for r in TEST_RESULTS.values() if r["success"])
    failed = sum(1 for r in TEST_RESULTS.values() if not r["success"])
    total = len(TEST_RESULTS)

    print(f"\n  Total tests: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    if failed > 0:
        print("\n  Failed tests:")
        for name, result in TEST_RESULTS.items():
            if not result["success"]:
                print(f"    - {name}: {result['message']}")

    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Test Watermarking Framework Functionality")
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "imports", "model", "watermarkers", "attacks", "dataset"])
    parser.add_argument("--model", type=str, default="gpt2")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  WATERMARKING FRAMEWORK FUNCTIONALITY TEST")
    print("="*60)
    print(f"  Device: {get_device()}")
    print(f"  Model: {args.model}")

    model, tokenizer = None, None

    if args.test in ["all", "imports"]:
        test_imports()

    if args.test in ["all", "model", "watermarkers"]:
        model, tokenizer = test_model_loading(args.model)

    if args.test in ["all", "watermarkers"]:
        test_watermarkers(model, tokenizer)

    if args.test in ["all", "attacks"]:
        test_attacks()

    if args.test in ["all", "dataset"]:
        test_dataset()

    success = print_summary()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
