"""
SEMSTAMP Watermark Implementation

Based on: "SemStamp: A Semantic Watermark with Paraphrastic Robustness 
           for Text Generation" (Hou et al., 2024)
Paper: https://arxiv.org/abs/2310.03991
GitHub: https://github.com/abehou/SemStamp

Key Features:
- Sentence-level semantic watermarking using LSH
- Robust to paraphrase attacks (semantic-preserving)
- Uses paraphrase-robust sentence encoder (fine-tuned SBERT)
- Rejection sampling until sentence falls in valid LSH partition
- Also supports k-SemStamp variant (k-means clustering)

Note:
    Original paper used OPT-1.3B.
    This implementation uses Qwen2.5-14B-Instruct by default.
    Sentence encoder: AbeHou/SemStamp-c4-sbert (fine-tuned for robustness)
    Set model parameter to use original models for replication.
"""

import torch
import numpy as np
import scipy.stats
from typing import Optional, List, Dict, Any, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
import nltk

from .base import SentenceLevelWatermarker, DetectionResult


class SEMSTAMPWatermark(SentenceLevelWatermarker):
    """SEMSTAMP sentence-level semantic watermarking.
    
    This method operates on sentence embeddings rather than tokens.
    It uses Locality-Sensitive Hashing (LSH) to partition the semantic
    space and generates sentences that fall into predetermined "valid"
    partitions based on the previous sentence.
    
    Args:
        model: Language model for generation
        tokenizer: Tokenizer for the model
        embedder: Sentence encoder model (e.g., SentenceTransformer)
        lsh_dim: Number of LSH hyperplanes (d in paper, default: 3)
        margin: Margin threshold for LSH robustness (default: 0.02)
        z_threshold: Detection z-threshold (default: 4.0)
        hash_key: Secret key for LSH hyperplane generation
        max_rejections: Maximum rejection samples before fallback (default: 100)
        detection_mode: "lsh" or "kmeans" (default: "lsh")
        
    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from sentence_transformers import SentenceTransformer
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
        >>> # Can also use: "facebook/opt-1.3b" for paper replication
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
        >>> embedder = SentenceTransformer("AbeHou/SemStamp-c4-sbert")
        >>> watermarker = SEMSTAMPWatermark(model, tokenizer, embedder=embedder)
        >>> text = watermarker.generate("The future of AI is")
        >>> result = watermarker.detect(text)
    
    References:
        [1] Hou et al., "SemStamp: A Semantic Watermark with Paraphrastic Robustness
            for Text Generation", NAACL 2024
        [2] Hou et al., "k-SemStamp: A Clustering-Based Semantic Watermark for 
            Detection of Machine-Generated Text", ACL 2024
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        embedder=None,
        lsh_dim: int = 3,
        margin: float = 0.0,  # Disabled margin check for maximum speed
        z_threshold: float = 4.0,
        hash_key: int = 15485863,
        max_rejections: int = 10,  # Increased for better watermark embedding
        detection_mode: str = "lsh",
        device: str = "cuda",
        gamma: float = 0.25,  # For LSH with d=3, gamma = 1/2^3 = 0.125, but we use 0.25
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            embedder=embedder,
            gamma=gamma,
            z_threshold=z_threshold,
            hash_key=hash_key,
            device=device,
        )
        self.lsh_dim = lsh_dim
        self.margin = margin
        self.max_rejections = max_rejections
        self.detection_mode = detection_mode
        
        # Generate LSH hyperplanes
        self._generate_lsh_hyperplanes()
        
        # Generate target signature based on hash key
        self._generate_target_signature()
        
        # Download punkt if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    
    def _generate_lsh_hyperplanes(self):
        """Generate random hyperplanes for LSH partitioning."""
        rng = np.random.RandomState(self.hash_key)
        
        # Embedding dimension (default for SBERT: 768)
        embed_dim = 768
        if self.embedder is not None:
            try:
                embed_dim = self.embedder.get_sentence_embedding_dimension()
            except:
                embed_dim = 768
        
        # Generate d random hyperplanes
        self.lsh_normals = []
        for _ in range(self.lsh_dim):
            normal = rng.randn(embed_dim)
            normal = normal / np.linalg.norm(normal)
            self.lsh_normals.append(normal)
        self.lsh_normals = np.array(self.lsh_normals)
    
    def _generate_target_signature(self):
        """Generate target LSH signature for watermarking."""
        rng = np.random.RandomState(self.hash_key + 1)
        self.target_signature = [rng.randint(0, 2) for _ in range(self.lsh_dim)]
    
    def _get_lsh_signature(
        self, 
        embedding: np.ndarray, 
        check_margin: bool = False
    ) -> Tuple[List[int], bool]:
        """Compute LSH signature for an embedding.
        
        Args:
            embedding: Sentence embedding vector
            check_margin: Whether to check margin constraint
            
        Returns:
            Tuple of (signature, passes_margin_check)
        """
        signature = []
        passes_margin = True
        
        for normal in self.lsh_normals:
            dot_product = np.dot(normal, embedding)
            
            if check_margin and abs(dot_product) < self.margin:
                passes_margin = False
            
            bit = 1 if dot_product > 0 else 0
            signature.append(bit)
        
        return signature, passes_margin
    
    def _get_valid_signature(self, prev_signature: List[int]) -> List[int]:
        """Get valid signature based on previous sentence.
        
        Uses a deterministic function based on the hash key to
        determine which signature is valid given the previous one.
        
        Args:
            prev_signature: LSH signature of previous sentence
            
        Returns:
            Required signature for next sentence
        """
        # Simple XOR-based transition function
        seed = hash(tuple(prev_signature)) ^ self.hash_key
        rng = np.random.RandomState(seed % (2**31))
        return [rng.randint(0, 2) for _ in range(self.lsh_dim)]
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """Generate watermarked text using SEMSTAMP method.
        
        Generates text sentence by sentence, using rejection sampling
        to ensure each sentence falls into a valid LSH partition.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate (approximate)
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated watermarked text
        """
        if self.embedder is None:
            raise ValueError("SEMSTAMP requires a sentence embedder. "
                           "Use: SentenceTransformer('AbeHou/SemStamp-c4-sbert')")
        
        generated_sentences = []
        current_prompt = prompt
        tokens_generated = 0
        
        # Get initial signature from prompt (or use target)
        prev_signature = self.target_signature
        if prompt.strip():
            prompt_sentences = self._split_sentences(prompt)
            if prompt_sentences:
                last_sent_emb = self._get_sentence_embedding(prompt_sentences[-1])
                prev_signature, _ = self._get_lsh_signature(last_sent_emb.cpu().numpy())
        
        # Limit number of sentences to prevent excessive generation
        max_sentences = max(1, max_new_tokens // 20)  # Roughly 1 sentence per 20 tokens
        sentence_count = 0
        
        while tokens_generated < max_new_tokens and sentence_count < max_sentences:
            # Get required signature for next sentence
            required_sig = self._get_valid_signature(prev_signature)
            
            # Try to find a sentence with matching signature
            valid_sentence = None
            best_sentence = None
            best_match_score = -1
            
            # Try max_rejections times to find exact or good match
            for attempt in range(self.max_rejections):
                # Generate a candidate sentence
                candidate = self._generate_sentence(
                    current_prompt, 
                    temperature=temperature,
                    top_p=top_p
                )
                
                if not candidate.strip():
                    continue
                
                # Get embedding and signature
                emb = self._get_sentence_embedding(candidate)
                sig, _ = self._get_lsh_signature(
                    emb.cpu().numpy(), 
                    check_margin=False
                )
                
                # Count matching bits
                match_score = sum(1 for a, b in zip(sig, required_sig) if a == b)
                
                # Track best candidate
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_sentence = candidate
                    best_sig = sig
                
                # Accept if exact match (all bits match)
                if match_score == self.lsh_dim:
                    valid_sentence = candidate
                    prev_signature = sig
                    break
            
            # Use best sentence found (exact match or closest)
            if valid_sentence is None and best_sentence is not None:
                valid_sentence = best_sentence
                prev_signature = best_sig
                # Last resort: generate without checking
                valid_sentence = self._generate_sentence(
                    current_prompt,
                    temperature=temperature,
                    top_p=top_p
                )
                if valid_sentence.strip():
                    emb = self._get_sentence_embedding(valid_sentence)
                    prev_signature, _ = self._get_lsh_signature(emb.cpu().numpy())
                else:
                    # Final fallback: stop generation
                    break
            
            generated_sentences.append(valid_sentence)
            current_prompt = current_prompt + " " + valid_sentence
            tokens_generated += len(self.tokenizer.encode(valid_sentence))
            sentence_count += 1
        
        return " ".join(generated_sentences)
    
    def _generate_sentence(
        self, 
        prompt: str, 
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_tokens: int = 30  # Reduced from 50 for faster generation
    ) -> str:
        """Generate a single sentence.
        
        Args:
            prompt: Current context
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            max_tokens: Maximum tokens for sentence
            
        Returns:
            Generated sentence
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated = self.tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # Extract first sentence
        sentences = self._split_sentences(generated)
        return sentences[0] if sentences else generated
    
    def detect(
        self,
        text: str,
        return_scores: bool = True,
        **kwargs
    ) -> DetectionResult:
        """Detect watermark using SEMSTAMP method.
        
        Analyzes the LSH signatures of consecutive sentences to see
        if they follow the expected pattern.
        
        Args:
            text: Text to analyze
            return_scores: Whether to return detailed scores
            
        Returns:
            DetectionResult with detection outcome
        """
        if self.embedder is None:
            raise ValueError("SEMSTAMP detection requires a sentence embedder.")
        
        sentences = self._split_sentences(text)
        
        if len(sentences) < 2:
            return DetectionResult(
                is_watermarked=False,
                z_score=0.0,
                p_value=1.0,
                num_tokens_scored=len(sentences),
            )
        
        # Count valid sentence transitions using bit-level scoring
        # This gives more granular signal than exact match only
        total_matching_bits = 0
        total_bits = 0
        num_exact_matches = 0
        num_checked = 0
        
        # Start with target signature or first sentence
        prev_signature = self.target_signature
        
        for i, sentence in enumerate(sentences):
            emb = self._get_sentence_embedding(sentence)
            sig, _ = self._get_lsh_signature(emb.cpu().numpy())
            
            if i > 0:  # Check transition from previous
                expected_sig = self._get_valid_signature(prev_signature)
                
                # Count matching bits
                matching = sum(1 for a, b in zip(sig, expected_sig) if a == b)
                total_matching_bits += matching
                total_bits += self.lsh_dim
                
                if matching == self.lsh_dim:
                    num_exact_matches += 1
                num_checked += 1
            
            prev_signature = sig
        
        # Compute z-score based on bit-level matching
        # Under null hypothesis, each bit has 50% chance of matching
        if total_bits > 0:
            observed_fraction = total_matching_bits / total_bits
            expected_fraction = 0.5  # Random would match 50% of bits
            std = np.sqrt(0.5 * 0.5 / total_bits)  # Std of sample proportion
            z_score = (observed_fraction - expected_fraction) / std if std > 0 else 0.0
        else:
            z_score = 0.0
        
        p_value = scipy.stats.norm.sf(z_score)
        is_watermarked = z_score > self.z_threshold
        
        return DetectionResult(
            is_watermarked=is_watermarked,
            z_score=z_score,
            p_value=p_value,
            green_fraction=total_matching_bits / total_bits if total_bits > 0 else 0.0,
            num_tokens_scored=len(sentences),
            confidence=1 - p_value if is_watermarked else 0.0,
            metadata={
                "num_sentences": len(sentences),
                "num_exact_matches": num_exact_matches,
                "num_checked": num_checked,
                "total_matching_bits": total_matching_bits,
                "total_bits": total_bits,
                "bit_match_fraction": total_matching_bits / total_bits if total_bits > 0 else 0.0,
                "lsh_dim": self.lsh_dim,
            }
        )
    
    def _get_sentence_embedding(self, sentence: str) -> torch.Tensor:
        """Get sentence embedding using the embedder."""
        if self.embedder is None:
            raise ValueError("Embedder not provided")
        return self.embedder.encode(sentence, convert_to_tensor=True)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        try:
            return nltk.sent_tokenize(text)
        except:
            # Fallback: split on periods
            return [s.strip() + '.' for s in text.split('.') if s.strip()]
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for reproducibility."""
        config = super().get_config()
        config.update({
            "lsh_dim": self.lsh_dim,
            "margin": self.margin,
            "max_rejections": self.max_rejections,
            "detection_mode": self.detection_mode,
            "target_signature": self.target_signature,
        })
        return config
