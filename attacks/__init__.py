"""
Attack implementations for watermark robustness evaluation.
"""

from .base import BaseAttack
from .lexical import SynonymAttack, SwapAttack, TypoAttack, MisspellingAttack
from .paraphrase import DIPPERAttack, PegasusAttack, ParrotAttack, GPTParaphraseAttack, BigramAttack
from .text_mixing import CopyPasteAttack

__all__ = [
    'BaseAttack',
    'SynonymAttack', 
    'SwapAttack', 
    'TypoAttack',
    'MisspellingAttack',
    'DIPPERAttack', 
    'PegasusAttack',
    'ParrotAttack',
    'GPTParaphraseAttack',
    'BigramAttack',
    'CopyPasteAttack',
]
