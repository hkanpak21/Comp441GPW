"""
LLM Watermarking Experiments Framework

This package implements watermarking methods from 4 seminal papers:
1. Unigram-Watermark (Zhao et al., 2023) - Provable Robust Watermarking
2. KGW/TGRL (Kirchenbauer et al., 2023) - Red-Green List Watermarking  
3. SEMSTAMP (Hou et al., 2024) - Semantic Watermark with LSH
4. WaterPark evaluation framework (Liang et al., 2025)

GitHub References:
- KGW: https://github.com/jwkirchenbauer/lm-watermarking
- Unigram: https://github.com/XuandongZhao/Unigram-Watermark
- SEMSTAMP: https://github.com/abehou/SemStamp
"""

__version__ = "1.0.0"
__author__ = "Comp441 Project Team"

from .watermarkers import UnigramWatermark, KGWWatermark, SEMSTAMPWatermark
from .attacks import (
    SynonymAttack, SwapAttack, TypoAttack, 
    DIPPERAttack, PegasusAttack, GPTParaphraseAttack,
    CopyPasteAttack, BigramAttack
)
from .metrics import (
    compute_z_score, compute_p_value, compute_tpr_fpr, compute_auc,
    compute_perplexity, compute_bertscore, compute_mauve
)
from .data_loaders import load_c4, load_opengen, load_lfqa
