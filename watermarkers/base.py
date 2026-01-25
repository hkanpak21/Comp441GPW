"""
Base classes for watermarking implementations.

This module provides abstract base classes that define the interface
for all watermarking methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class DetectionResult:
    """Result of watermark detection.
    
    Attributes:
        is_watermarked: Boolean prediction of whether text is watermarked
        z_score: Z-score statistic for detection
        p_value: P-value for hypothesis test
        green_fraction: Fraction of tokens in green list (token-level methods)
        num_tokens_scored: Number of tokens used in detection
        confidence: Detection confidence (1 - p_value if watermarked)
        metadata: Additional method-specific information
    """
    is_watermarked: bool
    z_score: float
    p_value: float
    green_fraction: float = 0.0
    num_tokens_scored: int = 0
    confidence: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    
    def __repr__(self):
        return (f"DetectionResult(is_watermarked={self.is_watermarked}, "
                f"z_score={self.z_score:.3f}, p_value={self.p_value:.4e})")


class BaseWatermarker(ABC):
    """Abstract base class for all watermarking methods.
    
    This class defines the interface that all watermarking implementations
    must follow, enabling fair comparison across methods.
    
    Attributes:
        model: The language model used for generation
        tokenizer: Tokenizer for the language model
        device: Device to run computations on (cuda/cpu)
        gamma: Green list ratio (typically 0.5 for token-level methods)
        delta: Watermark strength parameter
        z_threshold: Z-score threshold for detection (typically 4.0)
    
    Note:
        This implementation uses Qwen2.5-14B-Instruct by default for 
        state-of-the-art results. The original papers used:
        - OPT-1.3B, OPT-2.7B, OPT-6.7B (Kirchenbauer et al., 2023)
        - GPT2-XL, OPT-1.3B, LLaMA-7B (Zhao et al., 2023)
        - OPT-1.3B (Hou et al., 2024)
        
        You can substitute the model parameter to use these models for
        replication purposes.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
        gamma: float = 0.5,
        delta: float = 2.0,
        z_threshold: float = 4.0,
        hash_key: int = 15485863,  # Large prime for seeding
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.gamma = gamma
        self.delta = delta
        self.z_threshold = z_threshold
        self.hash_key = hash_key
        
        # Get vocabulary
        self.vocab_size = tokenizer.vocab_size
        self.vocab = list(range(self.vocab_size))
        
        # Move model to device only if not using device_map (accelerate)
        if model is not None:
            # Check if model uses device_map (accelerate offloading)
            if hasattr(model, 'hf_device_map') and model.hf_device_map:
                # Model already distributed, don't move
                pass
            else:
                try:
                    self.model = model.to(device)
                except Exception:
                    # Model may be on meta device or offloaded
                    pass
            self.model.eval()
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """Generate watermarked text.
        
        Args:
            prompt: Input prompt for generation
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability threshold
            **kwargs: Additional generation parameters
            
        Returns:
            Generated watermarked text
        """
        pass
    
    @abstractmethod
    def detect(
        self,
        text: str,
        return_scores: bool = True,
        **kwargs
    ) -> DetectionResult:
        """Detect watermark in text.
        
        Args:
            text: Text to check for watermark
            return_scores: Whether to return detailed scores
            **kwargs: Additional detection parameters
            
        Returns:
            DetectionResult with detection outcome and statistics
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get watermarker configuration for reproducibility."""
        return {
            "method": self.__class__.__name__,
            "gamma": self.gamma,
            "delta": self.delta,
            "z_threshold": self.z_threshold,
            "hash_key": self.hash_key,
            "vocab_size": self.vocab_size,
        }


class TokenLevelWatermarker(BaseWatermarker):
    """Base class for token-level watermarking methods.
    
    Token-level methods (Unigram, KGW) operate by biasing the probability
    distribution during generation to favor "green list" tokens.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = None
    
    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed RNG based on context tokens.
        
        Override in subclasses for different seeding strategies.
        """
        raise NotImplementedError
    
    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> List[int]:
        """Get green list token IDs for current context.
        
        Args:
            input_ids: Context token IDs for seeding
            
        Returns:
            List of token IDs in the green list
        """
        self._seed_rng(input_ids)
        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(
            self.vocab_size, 
            device=self.device, 
            generator=self.rng
        )
        greenlist_ids = vocab_permutation[:greenlist_size]
        return greenlist_ids.tolist()
    
    def _compute_z_score(self, num_green: int, total: int) -> float:
        """Compute z-score for detection.
        
        Args:
            num_green: Number of green tokens observed
            total: Total number of tokens scored
            
        Returns:
            Z-score statistic
        """
        if total == 0:
            return 0.0
        expected = self.gamma * total
        std = (total * self.gamma * (1 - self.gamma)) ** 0.5
        if std == 0:
            return 0.0
        z = (num_green - expected) / std
        return z


class SentenceLevelWatermarker(BaseWatermarker):
    """Base class for sentence-level watermarking methods.
    
    Sentence-level methods (SEMSTAMP) operate on semantic embeddings
    of sentences rather than individual tokens.
    """
    
    def __init__(self, *args, embedder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedder = embedder
    
    def _get_sentence_embedding(self, sentence: str) -> torch.Tensor:
        """Get embedding for a sentence.
        
        Args:
            sentence: Input sentence
            
        Returns:
            Embedding tensor
        """
        if self.embedder is None:
            raise ValueError("Embedder not provided")
        return self.embedder.encode(sentence, convert_to_tensor=True)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        import nltk
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download('punkt')
            sentences = nltk.sent_tokenize(text)
        return sentences
