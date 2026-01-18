"""
Unigram-Watermark Implementation

Based on: "Provable Robust Watermarking for AI-Generated Text" (Zhao et al., 2023)
Paper: https://arxiv.org/abs/2306.17439
GitHub: https://github.com/XuandongZhao/Unigram-Watermark

Key Features:
- Context-free (unigram) green/red list split
- Fixed green list determined only by secret hash key
- 2x more robust to editing than context-dependent methods
- Provable theoretical guarantees

Note:
    Original paper used GPT2-XL, OPT-1.3B, LLaMA-7B.
    This implementation uses Qwen2.5-14B-Instruct by default for better results.
    Set model parameter to use original models for replication.
"""

import torch
import scipy.stats
from typing import Optional, List, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer

from .base import TokenLevelWatermarker, DetectionResult


class UnigramWatermark(TokenLevelWatermarker):
    """Unigram (context-free) watermarking method.
    
    Unlike KGW which uses context-dependent green lists, Unigram uses
    a fixed green/red split determined only by the secret key. This makes
    it more robust to text editing attacks.
    
    Args:
        model: Language model for generation
        tokenizer: Tokenizer for the model
        gamma: Green list ratio (default: 0.5)
        delta: Watermark strength - logit bias added to green tokens (default: 2.0)
        z_threshold: Detection threshold (default: 4.0)
        hash_key: Secret key for green list generation
        use_unique_detector: Use "Unique" detector that counts unique green tokens
        
    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
        >>> # Can also use: "facebook/opt-1.3b", "decapoda-research/llama-7b-hf"  
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
        >>> watermarker = UnigramWatermark(model, tokenizer)
        >>> text = watermarker.generate("The future of AI is")
        >>> result = watermarker.detect(text)
        >>> print(f"Watermarked: {result.is_watermarked}, Z-score: {result.z_score:.2f}")
    
    References:
        [1] Zhao et al., "Provable Robust Watermarking for AI-Generated Text", ICLR 2024
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        gamma: float = 0.5,
        delta: float = 2.0,
        z_threshold: float = 4.0,
        hash_key: int = 15485863,
        use_unique_detector: bool = False,
        device: str = "cuda",
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            gamma=gamma,
            delta=delta,
            z_threshold=z_threshold,
            hash_key=hash_key,
            device=device,
        )
        self.use_unique_detector = use_unique_detector
        
        # Get actual model vocabulary size from model's output logits
        # This may differ from tokenizer.vocab_size
        # Try multiple approaches to get the correct vocab size
        actual_vocab_size = None
        
        # Method 1: Check model's config if available
        if hasattr(model, 'config') and hasattr(model.config, 'vocab_size'):
            actual_vocab_size = model.config.vocab_size
        
        # Method 2: Run a dummy forward pass
        if actual_vocab_size is None:
            try:
                with torch.no_grad():
                    # Use a valid token ID
                    test_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else (tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)
                    dummy_input = torch.tensor([[test_token_id]], device=device)
                    dummy_output = model(dummy_input)
                    actual_vocab_size = dummy_output.logits.shape[-1]
            except Exception:
                pass
        
        # Method 3: Check embedding layer size
        if actual_vocab_size is None and hasattr(model, 'get_input_embeddings'):
            try:
                embedding_layer = model.get_input_embeddings()
                if hasattr(embedding_layer, 'num_embeddings'):
                    actual_vocab_size = embedding_layer.num_embeddings
            except Exception:
                pass
        
        # Fallback to tokenizer vocab size if all else fails
        if actual_vocab_size is None:
            actual_vocab_size = tokenizer.vocab_size
            print(f"Warning: Could not detect model vocab size, using tokenizer vocab size: {actual_vocab_size}")
        else:
            print(f"Detected model vocab size: {actual_vocab_size} (tokenizer vocab size: {tokenizer.vocab_size})")
        
        # Use the actual model vocab size instead of tokenizer vocab size
        self.vocab_size = actual_vocab_size
        self.vocab = list(range(self.vocab_size))
        
        # Generate fixed green list (context-free)
        self.rng = torch.Generator(device=device)
        self._generate_fixed_greenlist()
    
    def _generate_fixed_greenlist(self):
        """Generate the fixed green list using hash key."""
        self.rng.manual_seed(self.hash_key)
        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(
            self.vocab_size, 
            device=self.device, 
            generator=self.rng
        )
        self.greenlist_ids = set(vocab_permutation[:greenlist_size].tolist())
        self.greenlist_mask = torch.zeros(self.vocab_size, device=self.device)
        self.greenlist_mask[list(self.greenlist_ids)] = 1.0
    
    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        """No-op for Unigram - uses fixed green list."""
        pass  # Unigram doesn't reseed based on context
    
    def _get_greenlist_ids(self, input_ids: torch.LongTensor = None) -> List[int]:
        """Return fixed green list (context-independent)."""
        return list(self.greenlist_ids)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """Generate watermarked text using Unigram method.
        
        The generation process adds delta to logits of green list tokens
        before sampling, biasing generation toward the green list.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling (0 = disabled)
            do_sample: Whether to sample (False = greedy)
            
        Returns:
            Generated watermarked text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        generated_ids = input_ids.clone()
        
        # Flag to ensure we only fix vocab size once per generation call
        vocab_size_fixed = False
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits[:, -1, :]
                actual_logits_vocab_size = logits.shape[-1]
                
                # Detect and fix vocab size mismatch on first step
                # This handles cases where the model was switched (e.g., OPT-1.3B -> Qwen-7B)
                # or when initial vocab size detection was incorrect
                if step == 0 and self.greenlist_mask.shape[0] != actual_logits_vocab_size:
                    print(f"Detected vocab size mismatch: initial={self.vocab_size}, mask={self.greenlist_mask.shape[0]}, actual_logits={actual_logits_vocab_size}")
                    print(f"Auto-fixing: regenerating greenlist_mask with correct size...")
                    self.vocab_size = actual_logits_vocab_size
                    self.vocab = list(range(self.vocab_size))
                    self._generate_fixed_greenlist()
                    vocab_size_fixed = True
                    print(f"Fixed! New vocab_size={self.vocab_size}, greenlist_mask size={self.greenlist_mask.shape[0]}")
                
                # Double-check: ensure mask matches logits (safety check)
                if self.greenlist_mask.shape[0] != logits.shape[-1]:
                    # This should not happen if fix above worked, but handle it anyway
                    if self.greenlist_mask.shape[0] > logits.shape[-1]:
                        greenlist_mask = self.greenlist_mask[:logits.shape[-1]]
                    else:
                        greenlist_mask = torch.zeros(logits.shape[-1], device=self.device)
                        greenlist_mask[:self.greenlist_mask.shape[0]] = self.greenlist_mask
                else:
                    greenlist_mask = self.greenlist_mask
                
                # Apply watermark: add delta to green list tokens
                logits = logits + self.delta * greenlist_mask
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                # Sample or greedy
                if do_sample:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Stop if EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((1, 1), device=self.device)
                ], dim=-1)
        
        # Decode only the generated part
        generated_text = self.tokenizer.decode(
            generated_ids[0, input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return generated_text
    
    def detect(
        self,
        text: str,
        return_scores: bool = True,
        **kwargs
    ) -> DetectionResult:
        """Detect watermark using Unigram method.
        
        Counts the fraction of tokens in the fixed green list and
        computes a z-score to determine if watermarked.
        
        Args:
            text: Text to analyze
            return_scores: Whether to return detailed scores
            
        Returns:
            DetectionResult with detection outcome
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) == 0:
            return DetectionResult(
                is_watermarked=False,
                z_score=0.0,
                p_value=1.0,
                green_fraction=0.0,
                num_tokens_scored=0,
            )
        
        if self.use_unique_detector:
            # "Unique" detector: count unique green tokens
            unique_tokens = set(tokens)
            num_green = sum(1 for t in unique_tokens if t in self.greenlist_ids)
            total = len(unique_tokens)
        else:
            # Standard detector: count all occurrences
            num_green = sum(1 for t in tokens if t in self.greenlist_ids)
            total = len(tokens)
        
        green_fraction = num_green / total if total > 0 else 0.0
        z_score = self._compute_z_score(num_green, total)
        p_value = scipy.stats.norm.sf(z_score)
        
        is_watermarked = z_score > self.z_threshold
        confidence = 1 - p_value if is_watermarked else 0.0
        
        return DetectionResult(
            is_watermarked=is_watermarked,
            z_score=z_score,
            p_value=p_value,
            green_fraction=green_fraction,
            num_tokens_scored=total,
            confidence=confidence,
            metadata={
                "num_green_tokens": num_green,
                "detector_type": "unique" if self.use_unique_detector else "standard",
            }
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for reproducibility."""
        config = super().get_config()
        config.update({
            "use_unique_detector": self.use_unique_detector,
            "greenlist_size": len(self.greenlist_ids),
        })
        return config
