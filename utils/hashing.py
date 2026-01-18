"""
Hashing utilities for watermarking.

Hash functions used for:
- Green list generation (KGW, Unigram)
- LSH hyperplane seeding (SEMSTAMP)
- Secret key management
"""

import hashlib
from typing import List, Tuple, Optional
import numpy as np


class HashFunctions:
    """Collection of hash functions for watermarking.
    
    Provides consistent hashing across watermarking implementations.
    """
    
    @staticmethod
    def token_hash(token_id: int, key: int = 15485863) -> int:
        """Hash a single token ID for seeding.
        
        Args:
            token_id: Token ID to hash
            key: Secret key (large prime)
            
        Returns:
            Hash value for seeding RNG
        """
        return (key * token_id) % (2**32)
    
    @staticmethod
    def context_hash(token_ids: List[int], key: int = 15485863) -> int:
        """Hash a sequence of token IDs.
        
        Args:
            token_ids: List of token IDs
            key: Secret key
            
        Returns:
            Hash value
        """
        combined = tuple([key] + list(token_ids))
        return hash(combined) % (2**32)
    
    @staticmethod
    def string_hash(text: str, key: int = 15485863) -> int:
        """Hash a string.
        
        Args:
            text: Text to hash
            key: Secret key
            
        Returns:
            Hash value
        """
        combined = f"{key}:{text}"
        return int(hashlib.sha256(combined.encode()).hexdigest(), 16) % (2**32)
    
    @staticmethod
    def generate_random_hyperplanes(
        dim: int,
        n_planes: int,
        seed: int = 42
    ) -> np.ndarray:
        """Generate random hyperplanes for LSH.
        
        Args:
            dim: Embedding dimension
            n_planes: Number of hyperplanes
            seed: Random seed
            
        Returns:
            Array of shape (n_planes, dim) with normalized hyperplane normals
        """
        rng = np.random.RandomState(seed)
        planes = rng.randn(n_planes, dim)
        
        # Normalize each hyperplane
        norms = np.linalg.norm(planes, axis=1, keepdims=True)
        planes = planes / norms
        
        return planes
    
    @staticmethod
    def lsh_signature(
        embedding: np.ndarray,
        hyperplanes: np.ndarray
    ) -> Tuple[int, ...]:
        """Compute LSH signature for an embedding.
        
        Args:
            embedding: Embedding vector
            hyperplanes: Hyperplane normals (n_planes, dim)
            
        Returns:
            Tuple of bits representing the signature
        """
        dots = np.dot(hyperplanes, embedding)
        signature = tuple(1 if d > 0 else 0 for d in dots)
        return signature
    
    @staticmethod
    def hamming_distance(sig1: Tuple[int, ...], sig2: Tuple[int, ...]) -> int:
        """Compute Hamming distance between two signatures.
        
        Args:
            sig1: First signature
            sig2: Second signature
            
        Returns:
            Number of differing bits
        """
        return sum(b1 != b2 for b1, b2 in zip(sig1, sig2))
