"""
Lexical Editing Attacks

These attacks modify text at the character or word level to try
to remove watermarks while preserving meaning.

Based on attacks described in:
- Zhao et al., 2023 (Unigram-Watermark)
- Kirchenbauer et al., 2023 (KGW)
- Liang et al., 2025 (WaterPark)
"""

import random
import string
from typing import List, Dict, Any, Optional
import nltk
from nltk.corpus import wordnet

from .base import BaseAttack, AttackResult


class SynonymAttack(BaseAttack):
    """Replace words with their synonyms.
    
    Uses WordNet to find synonyms for words and randomly
    replaces a fraction of words.
    
    Args:
        edit_rate: Fraction of words to replace (0-1)
        
    Example:
        >>> attack = SynonymAttack(edit_rate=0.3)
        >>> result = attack.attack("The quick brown fox jumps over the lazy dog")
    """
    
    def __init__(self, edit_rate: float = 0.3):
        super().__init__("SynonymAttack")
        self.edit_rate = edit_rate
        
        # Download WordNet if needed
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)
    
    def _get_wordnet_pos(self, tag: str) -> Optional[str]:
        """Convert NLTK POS tag to WordNet POS."""
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        return None
    
    def _get_synonyms(self, word: str, pos: Optional[str] = None) -> List[str]:
        """Get synonyms for a word from WordNet."""
        synonyms = set()
        synsets = wordnet.synsets(word, pos=pos)
        for synset in synsets:
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        return list(synonyms)
    
    def attack(self, text: str, **kwargs) -> AttackResult:
        """Apply synonym replacement attack.
        
        Args:
            text: Input text
            
        Returns:
            AttackResult with synonyms substituted
        """
        edit_rate = kwargs.get('edit_rate', self.edit_rate)
        
        words = text.split()
        if not words:
            return AttackResult(text, text, 0.0, 1.0)
        
        # Get POS tags for better synonym matching
        try:
            pos_tags = nltk.pos_tag(words)
        except:
            pos_tags = [(w, 'NN') for w in words]
        
        num_to_replace = max(1, int(len(words) * edit_rate))
        indices = random.sample(range(len(words)), min(num_to_replace, len(words)))
        
        replaced_count = 0
        for idx in indices:
            word, tag = pos_tags[idx]
            wn_pos = self._get_wordnet_pos(tag)
            synonyms = self._get_synonyms(word.lower(), wn_pos)
            
            if synonyms:
                # Preserve capitalization
                new_word = random.choice(synonyms)
                if word[0].isupper():
                    new_word = new_word.capitalize()
                words[idx] = new_word
                replaced_count += 1
        
        attacked_text = ' '.join(words)
        actual_rate = replaced_count / len(words) if words else 0.0
        
        return AttackResult(
            original_text=text,
            attacked_text=attacked_text,
            edit_rate=actual_rate,
            metadata={"words_replaced": replaced_count}
        )


class SwapAttack(BaseAttack):
    """Swap adjacent words randomly.
    
    Args:
        edit_rate: Fraction of positions to swap
        
    Example:
        >>> attack = SwapAttack(edit_rate=0.2)
        >>> result = attack.attack("The quick brown fox")
    """
    
    def __init__(self, edit_rate: float = 0.2):
        super().__init__("SwapAttack")
        self.edit_rate = edit_rate
    
    def attack(self, text: str, **kwargs) -> AttackResult:
        """Apply word swap attack."""
        edit_rate = kwargs.get('edit_rate', self.edit_rate)
        
        words = text.split()
        if len(words) < 2:
            return AttackResult(text, text, 0.0, 1.0)
        
        num_swaps = max(1, int(len(words) * edit_rate / 2))  # Divide by 2 since swap affects 2 words
        swap_positions = random.sample(range(len(words) - 1), min(num_swaps, len(words) - 1))
        
        for pos in swap_positions:
            words[pos], words[pos + 1] = words[pos + 1], words[pos]
        
        attacked_text = ' '.join(words)
        actual_rate = len(swap_positions) * 2 / len(words) if words else 0.0
        
        return AttackResult(
            original_text=text,
            attacked_text=attacked_text,
            edit_rate=actual_rate,
            metadata={"num_swaps": len(swap_positions)}
        )


class TypoAttack(BaseAttack):
    """Introduce typos into text.
    
    Randomly applies:
    - Character insertion
    - Character deletion
    - Character substitution
    - Adjacent character swap
    
    Args:
        edit_rate: Fraction of words to add typos to
        typo_types: List of typo types to use
        
    Example:
        >>> attack = TypoAttack(edit_rate=0.3)
        >>> result = attack.attack("The quick brown fox")
    """
    
    def __init__(
        self, 
        edit_rate: float = 0.3,
        typo_types: List[str] = None
    ):
        super().__init__("TypoAttack")
        self.edit_rate = edit_rate
        self.typo_types = typo_types or ['insert', 'delete', 'substitute', 'swap']
    
    def _insert_char(self, word: str) -> str:
        """Insert a random character."""
        if not word:
            return word
        pos = random.randint(0, len(word))
        char = random.choice(string.ascii_lowercase)
        return word[:pos] + char + word[pos:]
    
    def _delete_char(self, word: str) -> str:
        """Delete a random character."""
        if len(word) <= 1:
            return word
        pos = random.randint(0, len(word) - 1)
        return word[:pos] + word[pos + 1:]
    
    def _substitute_char(self, word: str) -> str:
        """Substitute a random character."""
        if not word:
            return word
        pos = random.randint(0, len(word) - 1)
        char = random.choice(string.ascii_lowercase)
        return word[:pos] + char + word[pos + 1:]
    
    def _swap_adjacent(self, word: str) -> str:
        """Swap adjacent characters."""
        if len(word) < 2:
            return word
        pos = random.randint(0, len(word) - 2)
        return word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
    
    def _add_typo(self, word: str) -> str:
        """Add a random typo to a word."""
        typo_type = random.choice(self.typo_types)
        
        if typo_type == 'insert':
            return self._insert_char(word)
        elif typo_type == 'delete':
            return self._delete_char(word)
        elif typo_type == 'substitute':
            return self._substitute_char(word)
        elif typo_type == 'swap':
            return self._swap_adjacent(word)
        return word
    
    def attack(self, text: str, **kwargs) -> AttackResult:
        """Apply typo attack."""
        edit_rate = kwargs.get('edit_rate', self.edit_rate)
        
        words = text.split()
        if not words:
            return AttackResult(text, text, 0.0, 1.0)
        
        num_to_modify = max(1, int(len(words) * edit_rate))
        indices = random.sample(range(len(words)), min(num_to_modify, len(words)))
        
        modified_count = 0
        for idx in indices:
            word = words[idx]
            # Only modify words with at least 3 characters
            if len(word) >= 3 and word.isalpha():
                words[idx] = self._add_typo(word)
                modified_count += 1
        
        attacked_text = ' '.join(words)
        actual_rate = modified_count / len(words) if words else 0.0
        
        return AttackResult(
            original_text=text,
            attacked_text=attacked_text,
            edit_rate=actual_rate,
            metadata={"words_modified": modified_count}
        )


class MisspellingAttack(BaseAttack):
    """Replace words with common misspellings.
    
    Uses a dictionary of common misspellings.
    
    Args:
        edit_rate: Fraction of words to misspell
        
    Example:
        >>> attack = MisspellingAttack(edit_rate=0.3)
        >>> result = attack.attack("The separate occurrence was definitely necessary")
    """
    
    # Common misspellings dictionary
    MISSPELLINGS = {
        "accommodate": "accomodate",
        "achieve": "acheive",
        "across": "accross",
        "address": "adress",
        "beginning": "begining",
        "believe": "beleive",
        "business": "buisness",
        "calendar": "calender",
        "category": "catagory",
        "committed": "comitted",
        "consciousness": "conciousness",
        "definitely": "definately",
        "disappoint": "dissapoint",
        "embarrass": "embarass",
        "environment": "enviornment",
        "existence": "existance",
        "experience": "experiance",
        "foreign": "foriegn",
        "government": "goverment",
        "guarantee": "garantee",
        "immediately": "immediatly",
        "independent": "independant",
        "intelligence": "inteligence",
        "knowledge": "knowlege",
        "maintenance": "maintainance",
        "necessary": "neccessary",
        "occurrence": "occurence",
        "parallel": "paralell",
        "particular": "perticular",
        "possession": "posession",
        "preferred": "prefered",
        "privilege": "priviledge",
        "professor": "proffessor",
        "receive": "recieve",
        "recommend": "recomend",
        "reference": "refrence",
        "schedule": "scheduel",
        "separate": "seperate",
        "similar": "similiar",
        "success": "sucess",
        "surprise": "suprise",
        "their": "thier",
        "tomorrow": "tommorrow",
        "truly": "truely",
        "until": "untill",
        "usually": "usally",
        "which": "wich",
        "weird": "wierd",
    }
    
    def __init__(self, edit_rate: float = 0.3):
        super().__init__("MisspellingAttack")
        self.edit_rate = edit_rate
    
    def attack(self, text: str, **kwargs) -> AttackResult:
        """Apply misspelling attack."""
        edit_rate = kwargs.get('edit_rate', self.edit_rate)
        
        words = text.split()
        if not words:
            return AttackResult(text, text, 0.0, 1.0)
        
        # Find words that can be misspelled
        misspellable_indices = []
        for i, word in enumerate(words):
            clean_word = word.lower().strip(string.punctuation)
            if clean_word in self.MISSPELLINGS:
                misspellable_indices.append(i)
        
        if not misspellable_indices:
            return AttackResult(text, text, 0.0, 1.0)
        
        num_to_modify = max(1, int(len(words) * edit_rate))
        indices = random.sample(
            misspellable_indices, 
            min(num_to_modify, len(misspellable_indices))
        )
        
        for idx in indices:
            word = words[idx]
            clean_word = word.lower().strip(string.punctuation)
            misspelling = self.MISSPELLINGS.get(clean_word, word)
            
            # Preserve capitalization and punctuation
            if word[0].isupper():
                misspelling = misspelling.capitalize()
            
            # Preserve trailing punctuation
            trailing = ''
            while word and word[-1] in string.punctuation:
                trailing = word[-1] + trailing
                word = word[:-1]
            
            words[idx] = misspelling + trailing
        
        attacked_text = ' '.join(words)
        actual_rate = len(indices) / len(words) if words else 0.0
        
        return AttackResult(
            original_text=text,
            attacked_text=attacked_text,
            edit_rate=actual_rate,
            metadata={"words_misspelled": len(indices)}
        )
