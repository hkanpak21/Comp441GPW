"""
Paraphrase Attacks

These attacks use neural paraphrasers to rewrite text while
preserving meaning, attempting to remove watermarks.

Based on attacks described in:
- Hou et al., 2024 (SEMSTAMP) - Bigram Attack
- Liang et al., 2025 (WaterPark) - DIPPER, PEGASUS
- Kirchenbauer et al., 2023 (KGW) - T5 paraphraser
"""

import random
from typing import List, Dict, Any, Optional
import nltk

from .base import BaseAttack, AttackResult


class DIPPERAttack(BaseAttack):
    """DIPPER paraphrase attack.
    
    DIPPER (Discourse-aware Inference for Paraphrasing with 
    Prose Recombination) allows control over lexical and 
    order diversity.
    
    Requires: pip install torch transformers
    Model: kalpeshk2011/dipper-paraphraser-xxl (11B params)
    
    Args:
        lexical_diversity: Lexical diversity (0-100, default: 40)
        order_diversity: Order diversity (0-100, default: 20)
        
    Note:
        Original WaterPark experiments used lex_div=40, order_div=20.
        Higher values = more paraphrasing but potentially less coherent.
        
    Example:
        >>> attack = DIPPERAttack(lexical_diversity=40, order_diversity=20)
        >>> result = attack.attack("The quick brown fox jumps over the lazy dog")
    """
    
    def __init__(
        self, 
        lexical_diversity: int = 40,
        order_diversity: int = 20,
        device: str = "cuda"
    ):
        super().__init__("DIPPERAttack")
        self.lexical_diversity = lexical_diversity
        self.order_diversity = order_diversity
        self.device = device
        self.model = None
        self.tokenizer = None
    
    def _load_model(self):
        """Lazy load the DIPPER model."""
        if self.model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            
            model_name = "kalpeshk2011/dipper-paraphraser-xxl"
            print(f"Loading DIPPER model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, 
                device_map="auto"
            )
    
    def attack(self, text: str, **kwargs) -> AttackResult:
        """Apply DIPPER paraphrase attack.
        
        Args:
            text: Input text
            lexical_diversity: Override default lexical diversity
            order_diversity: Override default order diversity
            
        Returns:
            AttackResult with paraphrased text
        """
        self._load_model()
        
        lex_div = kwargs.get('lexical_diversity', self.lexical_diversity)
        order_div = kwargs.get('order_diversity', self.order_diversity)
        
        # DIPPER uses control codes for diversity
        input_text = f"lexical = {lex_div}, order = {order_div} {text}"
        
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        
        paraphrased = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return AttackResult(
            original_text=text,
            attacked_text=paraphrased,
            metadata={
                "lexical_diversity": lex_div,
                "order_diversity": order_div,
                "model": "kalpeshk2011/dipper-paraphraser-xxl"
            }
        )


class PegasusAttack(BaseAttack):
    """PEGASUS paraphrase attack.
    
    Uses PEGASUS fine-tuned for paraphrasing.
    
    Requires: pip install torch transformers
    Model: tuner007/pegasus_paraphrase
    
    Args:
        num_beams: Number of beams for beam search
        temperature: Sampling temperature
        
    Note:
        SEMSTAMP paper uses PEGASUS for paraphrase attacks.
        
    Example:
        >>> attack = PegasusAttack(num_beams=5)
        >>> result = attack.attack("The quick brown fox jumps over the lazy dog")
    """
    
    def __init__(
        self,
        num_beams: int = 5,
        temperature: float = 1.5,
        device: str = "cuda"
    ):
        super().__init__("PegasusAttack")
        self.num_beams = num_beams
        self.temperature = temperature
        self.device = device
        self.model = None
        self.tokenizer = None
    
    def _load_model(self):
        """Lazy load the PEGASUS model."""
        if self.model is None:
            from transformers import PegasusForConditionalGeneration, PegasusTokenizer
            
            model_name = "tuner007/pegasus_paraphrase"
            print(f"Loading PEGASUS model: {model_name}")
            
            self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
            self.model = PegasusForConditionalGeneration.from_pretrained(
                model_name
            ).to(self.device)
    
    def _paraphrase_sentence(self, sentence: str) -> str:
        """Paraphrase a single sentence."""
        # Truncate very long sentences
        if len(sentence) > 500:
            sentence = sentence[:500]

        # Clean the sentence - remove problematic characters
        sentence = sentence.encode('ascii', 'ignore').decode('ascii')

        if not sentence.strip():
            return sentence

        try:
            inputs = self.tokenizer(
                sentence,
                return_tensors="pt",
                max_length=128,  # Shorter max length
                truncation=True
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=self.num_beams,
                temperature=self.temperature,
                do_sample=False,  # Deterministic for stability
            )

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            # Return original if paraphrase fails
            return sentence
    
    def attack(self, text: str, **kwargs) -> AttackResult:
        """Apply PEGASUS paraphrase attack.
        
        Paraphrases each sentence separately to preserve structure.
        
        Args:
            text: Input text
            
        Returns:
            AttackResult with paraphrased text
        """
        self._load_model()
        
        # Split into sentences and paraphrase each
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            nltk.download('punkt', quiet=True)
            sentences = nltk.sent_tokenize(text)
        
        paraphrased_sentences = [self._paraphrase_sentence(s) for s in sentences]
        paraphrased = ' '.join(paraphrased_sentences)
        
        return AttackResult(
            original_text=text,
            attacked_text=paraphrased,
            metadata={
                "num_sentences": len(sentences),
                "model": "tuner007/pegasus_paraphrase"
            }
        )


class ParrotAttack(BaseAttack):
    """Parrot paraphrase attack.
    
    Uses Parrot model for paraphrasing.
    
    Model: prithivida/parrot_paraphraser_on_T5
    
    Example:
        >>> attack = ParrotAttack()
        >>> result = attack.attack("The quick brown fox jumps over the lazy dog")
    """
    
    def __init__(self, device: str = "cuda"):
        super().__init__("ParrotAttack")
        self.device = device
        self.model = None
        self.tokenizer = None
    
    def _load_model(self):
        """Lazy load the Parrot model."""
        if self.model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            
            model_name = "prithivida/parrot_paraphraser_on_T5"
            print(f"Loading Parrot model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
    
    def attack(self, text: str, **kwargs) -> AttackResult:
        """Apply Parrot paraphrase attack."""
        self._load_model()
        
        input_text = f"paraphrase: {text} </s>"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=256,
            truncation=True
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=256,
            do_sample=True,
            num_return_sequences=1,
        )
        
        paraphrased = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return AttackResult(
            original_text=text,
            attacked_text=paraphrased,
            metadata={"model": "prithivida/parrot_paraphraser_on_T5"}
        )


class GPTParaphraseAttack(BaseAttack):
    """GPT-based paraphrase attack.
    
    Uses OpenAI GPT-3.5 or GPT-4 API for paraphrasing.
    
    Requires: pip install openai
    Environment: OPENAI_API_KEY
    
    Args:
        model: OpenAI model to use (default: "gpt-3.5-turbo")
        temperature: Sampling temperature
        
    Note:
        WaterPark paper uses GPT-3.5-turbo for advanced attacks.
        Costs money to use!
        
    Example:
        >>> attack = GPTParaphraseAttack(model="gpt-3.5-turbo")
        >>> result = attack.attack("The quick brown fox jumps over the lazy dog")
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 1.0,
    ):
        super().__init__("GPTParaphraseAttack")
        self.model = model
        self.temperature = temperature
        self.client = None
    
    def _load_client(self):
        """Lazy load OpenAI client."""
        if self.client is None:
            import openai
            self.client = openai.OpenAI()
    
    def attack(self, text: str, **kwargs) -> AttackResult:
        """Apply GPT paraphrase attack.
        
        Args:
            text: Input text
            prompt_type: "standard" or "bigram" (for SEMSTAMP bigram attack)
            
        Returns:
            AttackResult with paraphrased text
        """
        self._load_client()
        
        prompt_type = kwargs.get('prompt_type', 'standard')
        
        if prompt_type == 'bigram':
            system_prompt = (
                "You are a paraphrasing assistant. Paraphrase the following text "
                "while minimizing bigram overlap with the original. Change word "
                "order, use synonyms, and restructure sentences, but preserve meaning."
            )
        else:
            system_prompt = (
                "You are a paraphrasing assistant. Paraphrase the following text "
                "while preserving its exact meaning. Use different words and "
                "sentence structures where possible."
            )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=self.temperature,
        )
        
        paraphrased = response.choices[0].message.content
        
        return AttackResult(
            original_text=text,
            attacked_text=paraphrased,
            metadata={
                "model": self.model,
                "prompt_type": prompt_type,
                "tokens_used": response.usage.total_tokens
            }
        )


class BigramAttack(BaseAttack):
    """Bigram Paraphrase Attack (SEMSTAMP-specific).
    
    This attack specifically targets sentence-level watermarks by
    minimizing bigram overlap between original and paraphrased text.
    
    From SEMSTAMP paper (Hou et al., 2024):
    "We propose a novel 'bigram paraphrase' attack by prompting 
    paraphraser to minimize bigram overlap."
    
    Args:
        paraphraser: Underlying paraphraser to use ("pegasus", "gpt", "parrot")
        min_bigram_change: Minimum fraction of bigrams to change
        
    Example:
        >>> attack = BigramAttack(paraphraser="pegasus")
        >>> result = attack.attack("The quick brown fox jumps over the lazy dog")
    """
    
    def __init__(
        self,
        paraphraser: str = "pegasus",
        min_bigram_change: float = 0.5,
        device: str = "cuda"
    ):
        super().__init__("BigramAttack")
        self.paraphraser = paraphraser
        self.min_bigram_change = min_bigram_change
        self.device = device
        self._paraphraser = None
    
    def _get_bigrams(self, text: str) -> set:
        """Get set of bigrams from text."""
        words = text.lower().split()
        return set(zip(words[:-1], words[1:]))
    
    def _bigram_overlap(self, text1: str, text2: str) -> float:
        """Compute bigram overlap between two texts."""
        bigrams1 = self._get_bigrams(text1)
        bigrams2 = self._get_bigrams(text2)
        
        if not bigrams1:
            return 0.0
        
        overlap = len(bigrams1 & bigrams2)
        return overlap / len(bigrams1)
    
    def _get_paraphraser(self):
        """Get the underlying paraphraser."""
        if self._paraphraser is None:
            if self.paraphraser == "pegasus":
                self._paraphraser = PegasusAttack(device=self.device)
            elif self.paraphraser == "gpt":
                self._paraphraser = GPTParaphraseAttack()
            elif self.paraphraser == "parrot":
                self._paraphraser = ParrotAttack(device=self.device)
            else:
                raise ValueError(f"Unknown paraphraser: {self.paraphraser}")
        return self._paraphraser
    
    def attack(self, text: str, max_attempts: int = 5, **kwargs) -> AttackResult:
        """Apply bigram paraphrase attack.
        
        Repeatedly paraphrases until bigram overlap is below threshold.
        
        Args:
            text: Input text
            max_attempts: Maximum paraphrase attempts
            
        Returns:
            AttackResult with bigram-minimizing paraphrase
        """
        paraphraser = self._get_paraphraser()
        
        best_result = None
        best_overlap = 1.0
        
        current_text = text
        for attempt in range(max_attempts):
            # Use bigram prompt if GPT
            if self.paraphraser == "gpt":
                result = paraphraser.attack(current_text, prompt_type="bigram")
            else:
                result = paraphraser.attack(current_text)
            
            overlap = self._bigram_overlap(text, result.attacked_text)
            
            if overlap < best_overlap:
                best_overlap = overlap
                best_result = result
            
            # Check if meets threshold
            if overlap < (1 - self.min_bigram_change):
                break
            
            current_text = result.attacked_text
        
        if best_result is None:
            best_result = AttackResult(text, text, 0.0, 1.0)
        
        best_result.metadata = best_result.metadata or {}
        best_result.metadata.update({
            "bigram_overlap": best_overlap,
            "attempts": attempt + 1,
            "underlying_paraphraser": self.paraphraser
        })
        
        return best_result
