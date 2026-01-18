"""
Watermarker implementations from prior work papers.
"""

from .base import BaseWatermarker, DetectionResult
from .unigram import UnigramWatermark
from .kgw import KGWWatermark
from .semstamp import SEMSTAMPWatermark
from .gpw import GPWWatermark, GPWConfig, SRConfig, create_gpw_variant

__all__ = [
    'BaseWatermarker', 
    'DetectionResult',
    'UnigramWatermark', 
    'KGWWatermark', 
    'SEMSTAMPWatermark',
    'GPWWatermark',
    'GPWConfig',
    'SRConfig',
    'create_gpw_variant',
]
