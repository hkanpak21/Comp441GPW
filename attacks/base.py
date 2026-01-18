"""
Base class for watermark attacks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class AttackResult:
    """Result of applying an attack.
    
    Attributes:
        original_text: Original text before attack
        attacked_text: Text after attack applied
        edit_rate: Actual edit rate achieved
        similarity: Semantic similarity to original (0-1)
        metadata: Additional attack-specific information
    """
    original_text: str
    attacked_text: str
    edit_rate: float = 0.0
    similarity: float = 1.0
    metadata: Dict[str, Any] = None


class BaseAttack(ABC):
    """Abstract base class for watermark removal attacks.
    
    All attack implementations should inherit from this class
    and implement the attack() method.
    """
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def attack(self, text: str, **kwargs) -> AttackResult:
        """Apply attack to text.
        
        Args:
            text: Input text to attack
            **kwargs: Attack-specific parameters
            
        Returns:
            AttackResult with original and attacked text
        """
        pass
    
    def __call__(self, text: str, **kwargs) -> str:
        """Convenience method to apply attack and return attacked text."""
        result = self.attack(text, **kwargs)
        return result.attacked_text
    
    def get_config(self) -> Dict[str, Any]:
        """Get attack configuration for reproducibility."""
        return {
            "attack_name": self.name,
        }
