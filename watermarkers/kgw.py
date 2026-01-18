"""
KGW/TGRL Watermark Implementation

Based on: "A Watermark for Large Language Models" (Kirchenbauer et al., 2023)
Paper: https://arxiv.org/abs/2301.10226
GitHub: https://github.com/jwkirchenbauer/lm-watermarking

Key Features:
- Context-dependent green/red list based on previous token(s)
- Soft watermark with adjustable bias (delta)
- Statistical detection using z-score
- Supports both hard (reject red) and soft (bias green) variants

Note:
    Original paper used OPT-1.3B, OPT-2.7B, OPT-6.7B.
    This implementation uses Qwen2.5-14B-Instruct by default for better results.
    Set model parameter to use original models for replication.
"""

import torch
import scipy.stats
import collections
from typing import Optional, List, Dict, Any, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from nltk.util import ngrams

from .base import TokenLevelWatermarker, DetectionResult


class KGWWatermark(TokenLevelWatermarker):
    """KGW (Kirchenbauer-Geiping-Wen) context-dependent watermarking.
    
    This method creates a green/red token split at each generation step
    based on the hash of the previous token(s). The logits for green tokens
    are increased by delta, biasing the model to produce more green tokens.
    
    Args:
        model: Language model for generation
        tokenizer: Tokenizer for the model
        gamma: Green list ratio (default: 0.5)
        delta: Watermark strength - logit bias (default: 2.0)
        z_threshold: Detection threshold (default: 4.0)
        hash_key: Secret key for hashing
        context_width: Number of previous tokens to use for seeding (h in paper)
                       1 = simple seeding (default), higher = more context-dependent
        seeding_scheme: How to compute seed from context
                       "simple_1": Use only previous token (default)
                       "selfhash": Use token's own hash
        ignore_repeated_bigrams: Only count unique bigrams once in detection
        hard_watermark: If True, use hard watermark (only allow green tokens)
        
    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
        >>> # Can also use: "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b"
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
        >>> watermarker = KGWWatermark(model, tokenizer, delta=2.0)
        >>> text = watermarker.generate("The future of AI is")
        >>> result = watermarker.detect(text)
        >>> print(f"Watermarked: {result.is_watermarked}, Z-score: {result.z_score:.2f}")
    
    References:
        [1] Kirchenbauer et al., "A Watermark for Large Language Models", ICML 2023
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        gamma: float = 0.5,
        delta: float = 2.0,
        z_threshold: float = 4.0,
        hash_key: int = 15485863,
        context_width: int = 1,
        seeding_scheme: str = "simple_1",
        ignore_repeated_bigrams: bool = True,
        hard_watermark: bool = False,
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
        self.context_width = context_width
        self.seeding_scheme = seeding_scheme
        self.ignore_repeated_bigrams = ignore_repeated_bigrams
        self.hard_watermark = hard_watermark
        self.rng = torch.Generator(device=device)
    
    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed RNG based on context tokens.
        
        Args:
            input_ids: Previous token IDs for context
        """
        if self.seeding_scheme == "simple_1":
            # Use only the last token
            if len(input_ids) >= 1:
                prev_token = input_ids[-1].item()
                self.rng.manual_seed(self.hash_key * prev_token)
            else:
                self.rng.manual_seed(self.hash_key)
        elif self.seeding_scheme == "lefthash":
            # Use hash of all context tokens
            if len(input_ids) >= self.context_width:
                context = input_ids[-self.context_width:].tolist()
                seed = hash(tuple([self.hash_key] + context)) % (2**32)
                self.rng.manual_seed(seed)
            else:
                self.rng.manual_seed(self.hash_key)
        else:
            raise ValueError(f"Unknown seeding scheme: {self.seeding_scheme}")
    
    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> List[int]:
        """Get green list token IDs for current context.
        
        Args:
            input_ids: Context token IDs
            
        Returns:
            List of green list token IDs
        """
        self._seed_rng(input_ids)
        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(
            self.vocab_size, 
            device=self.device, 
            generator=self.rng
        )
        return vocab_permutation[:greenlist_size].tolist()
    
    def _calc_greenlist_mask(
        self, 
        scores: torch.FloatTensor, 
        greenlist_token_ids: List[int]
    ) -> torch.BoolTensor:
        """Create a mask for green list tokens.
        
        Args:
            scores: Logit scores from model
            greenlist_token_ids: List of green token IDs
            
        Returns:
            Boolean mask with True for green tokens
        """
        green_tokens_mask = torch.zeros_like(scores, dtype=torch.bool)
        green_tokens_mask[greenlist_token_ids] = True
        return green_tokens_mask
    
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
        """Generate watermarked text using KGW method.
        
        At each step, computes green list based on previous token(s),
        then adds delta to green token logits before sampling.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
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
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits[:, -1, :].squeeze(0)
                
                # Get green list based on context
                greenlist_ids = self._get_greenlist_ids(generated_ids[0])
                greenlist_mask = self._calc_greenlist_mask(logits, greenlist_ids)
                
                if self.hard_watermark:
                    # Hard watermark: set red list to -inf
                    logits[~greenlist_mask] = float('-inf')
                else:
                    # Soft watermark: add delta to green list
                    logits[greenlist_mask] = logits[greenlist_mask] + self.delta
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][-1]
                    logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        0, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                # Sample or greedy
                if do_sample:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs.unsqueeze(0), num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True).unsqueeze(0)
                
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
        """Detect watermark using KGW method.
        
        Iterates through tokens, computing green list for each based on
        its prefix context, and counts how many fall in their green list.
        
        Args:
            text: Text to analyze
            return_scores: Whether to return detailed scores
            
        Returns:
            DetectionResult with detection outcome
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) < 2:
            return DetectionResult(
                is_watermarked=False,
                z_score=0.0,
                p_value=1.0,
                green_fraction=0.0,
                num_tokens_scored=0,
            )
        
        tokens_tensor = torch.tensor(tokens, device=self.device)
        
        if self.ignore_repeated_bigrams:
            # Count unique bigrams only
            bigram_table = {}
            token_bigram_generator = ngrams(tokens, 2)
            freq = collections.Counter(token_bigram_generator)
            
            for bigram in freq.keys():
                prefix = torch.tensor([bigram[0]], device=self.device)
                greenlist_ids = self._get_greenlist_ids(prefix)
                bigram_table[bigram] = bigram[1] in greenlist_ids
            
            num_green = sum(bigram_table.values())
            total = len(bigram_table)
        else:
            # Standard detection
            num_green = 0
            for idx in range(1, len(tokens)):
                prefix = tokens_tensor[:idx]
                greenlist_ids = self._get_greenlist_ids(prefix)
                if tokens[idx] in greenlist_ids:
                    num_green += 1
            total = len(tokens) - 1
        
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
                "context_width": self.context_width,
                "ignore_repeated_bigrams": self.ignore_repeated_bigrams,
            }
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for reproducibility."""
        config = super().get_config()
        config.update({
            "context_width": self.context_width,
            "seeding_scheme": self.seeding_scheme,
            "ignore_repeated_bigrams": self.ignore_repeated_bigrams,
            "hard_watermark": self.hard_watermark,
        })
        return config
