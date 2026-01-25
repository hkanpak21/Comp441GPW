#!/usr/bin/env python3
"""
Phase 1: Generate paraphrased texts from watermarked texts.
Only loads Pegasus model, saves paraphrased texts to disk.
"""

import os
import sys
import pickle
import argparse
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def paraphrase_texts(input_pickle: str, output_pickle: str, max_samples: int = None):
    """Load texts, paraphrase them, save to new pickle."""
    import torch
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer

    # Load input texts
    print(f"Loading texts from: {input_pickle}")
    with open(input_pickle, 'rb') as f:
        data = pickle.load(f)

    texts = data.get('texts', [])
    if max_samples and len(texts) > max_samples:
        texts = texts[:max_samples]
        print(f"Limited to {max_samples} samples")

    print(f"Total texts to paraphrase: {len(texts)}")

    # Load Pegasus model
    print("Loading Pegasus model for paraphrasing...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "tuner007/pegasus_paraphrase"

    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()

    def paraphrase_text(text: str) -> str:
        """Paraphrase a text by paraphrasing each sentence."""
        import re

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        paraphrased_sentences = []

        for sentence in sentences:
            if not sentence.strip():
                continue

            # Truncate very long sentences
            if len(sentence) > 400:
                sentence = sentence[:400]

            # Clean the sentence
            sentence = sentence.encode('ascii', 'ignore').decode('ascii').strip()
            if not sentence:
                continue

            try:
                inputs = tokenizer(
                    sentence,
                    return_tensors="pt",
                    max_length=100,
                    truncation=True
                ).to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=100,
                        num_beams=4,
                        do_sample=False,
                    )

                paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)
                paraphrased_sentences.append(paraphrased)
            except Exception as e:
                # Return original on failure
                paraphrased_sentences.append(sentence)

        return ' '.join(paraphrased_sentences)

    # Paraphrase all texts
    paraphrased_texts = []
    success_count = 0

    for i, text in enumerate(tqdm(texts, desc="Paraphrasing")):
        try:
            paraphrased = paraphrase_text(text)
            paraphrased_texts.append({
                'original': text,
                'paraphrased': paraphrased,
                'success': True
            })
            success_count += 1
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            paraphrased_texts.append({
                'original': text,
                'paraphrased': text,  # Use original if paraphrase fails
                'success': False
            })

    # Save results
    output_data = {
        'model': data.get('model', 'unknown'),
        'watermarker': data.get('watermarker', 'unknown'),
        'original_file': input_pickle,
        'paraphrased_texts': paraphrased_texts,
        'total': len(texts),
        'success_count': success_count,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }

    with open(output_pickle, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"\nParaphrase complete: {success_count}/{len(texts)} successful")
    print(f"Saved to: {output_pickle}")

    # Clear GPU memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return output_pickle


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Generate paraphrased texts")
    parser.add_argument("--input", required=True, help="Input pickle file with generated texts")
    parser.add_argument("--output", required=True, help="Output pickle file for paraphrased texts")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process")

    args = parser.parse_args()

    paraphrase_texts(args.input, args.output, args.max_samples)


if __name__ == "__main__":
    main()
