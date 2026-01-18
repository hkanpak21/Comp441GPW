"""
Text Mixing Attacks (Copy-Paste)

These attacks mix watermarked text with human-written text
to dilute the watermark signal.

Based on attacks described in:
- Kirchenbauer et al., 2023 (KGW)
- Liang et al., 2025 (WaterPark)
"""

import random
from typing import List, Optional

from .base import BaseAttack, AttackResult


class CopyPasteAttack(BaseAttack):
    """Copy-paste attack mixing watermarked and human text.
    
    This attack simulates scenarios where a user copies watermarked
    text and pastes it into a document with human-written content,
    or vice versa.
    
    Described in KGW paper as "span replacement" attack.
    
    Args:
        n_segments: Number of segments to split text into
        watermark_ratio: Fraction of segments that are watermarked (0-1)
        
    Note:
        WaterPark notation: CP-n-m means n segments, m% watermarked
        E.g., CP-3-50 = 3 segments, 50% watermarked
        
    Example:
        >>> attack = CopyPasteAttack(n_segments=3, watermark_ratio=0.5)
        >>> result = attack.attack(watermarked_text, human_text=human_text)
    """
    
    def __init__(
        self,
        n_segments: int = 3,
        watermark_ratio: float = 0.5,
    ):
        super().__init__("CopyPasteAttack")
        self.n_segments = n_segments
        self.watermark_ratio = watermark_ratio
    
    def _split_into_segments(self, text: str, n: int) -> List[str]:
        """Split text into n roughly equal segments."""
        words = text.split()
        if len(words) < n:
            return [text]
        
        segment_size = len(words) // n
        segments = []
        
        for i in range(n):
            start = i * segment_size
            end = start + segment_size if i < n - 1 else len(words)
            segments.append(' '.join(words[start:end]))
        
        return segments
    
    def attack(
        self, 
        text: str, 
        human_text: Optional[str] = None,
        **kwargs
    ) -> AttackResult:
        """Apply copy-paste attack.
        
        Args:
            text: Watermarked text
            human_text: Human-written text to mix with (required)
            
        Returns:
            AttackResult with mixed text
        """
        if human_text is None:
            # If no human text provided, just return original
            return AttackResult(
                original_text=text,
                attacked_text=text,
                metadata={"error": "No human text provided"}
            )
        
        n_segments = kwargs.get('n_segments', self.n_segments)
        watermark_ratio = kwargs.get('watermark_ratio', self.watermark_ratio)
        
        # Split both texts into segments
        wm_segments = self._split_into_segments(text, n_segments)
        human_segments = self._split_into_segments(human_text, n_segments)
        
        # Determine which segments to use from watermarked text
        n_wm = max(1, int(n_segments * watermark_ratio))
        wm_indices = set(random.sample(range(n_segments), n_wm))
        
        # Build mixed text
        mixed_segments = []
        for i in range(n_segments):
            if i in wm_indices:
                if i < len(wm_segments):
                    mixed_segments.append(wm_segments[i])
            else:
                if i < len(human_segments):
                    mixed_segments.append(human_segments[i])
        
        mixed_text = ' '.join(mixed_segments)
        
        return AttackResult(
            original_text=text,
            attacked_text=mixed_text,
            metadata={
                "n_segments": n_segments,
                "watermark_ratio": watermark_ratio,
                "watermarked_segments": list(wm_indices),
            }
        )


class SentenceMixingAttack(BaseAttack):
    """Sentence-level mixing attack.
    
    Alternates sentences between watermarked and human text.
    More fine-grained than segment-based copy-paste.
    
    Args:
        watermark_ratio: Fraction of sentences from watermarked text
        
    Example:
        >>> attack = SentenceMixingAttack(watermark_ratio=0.5)
        >>> result = attack.attack(watermarked_text, human_text=human_text)
    """
    
    def __init__(self, watermark_ratio: float = 0.5):
        super().__init__("SentenceMixingAttack")
        self.watermark_ratio = watermark_ratio
    
    def attack(
        self,
        text: str,
        human_text: Optional[str] = None,
        **kwargs
    ) -> AttackResult:
        """Apply sentence mixing attack."""
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        if human_text is None:
            return AttackResult(text, text, metadata={"error": "No human text"})
        
        wm_sentences = nltk.sent_tokenize(text)
        human_sentences = nltk.sent_tokenize(human_text)
        
        watermark_ratio = kwargs.get('watermark_ratio', self.watermark_ratio)
        
        # Determine number of sentences to take from each
        total = min(len(wm_sentences), len(human_sentences)) * 2
        n_wm = int(total * watermark_ratio)
        
        # Select random sentences from each source
        selected_wm = random.sample(wm_sentences, min(n_wm, len(wm_sentences)))
        selected_human = random.sample(human_sentences, min(total - n_wm, len(human_sentences)))
        
        # Combine and shuffle
        all_sentences = selected_wm + selected_human
        random.shuffle(all_sentences)
        
        mixed_text = ' '.join(all_sentences)
        
        return AttackResult(
            original_text=text,
            attacked_text=mixed_text,
            metadata={
                "total_sentences": len(all_sentences),
                "watermarked_sentences": len(selected_wm),
                "human_sentences": len(selected_human),
            }
        )
