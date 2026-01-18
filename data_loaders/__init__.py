"""
Dataset loaders for watermark experiments.
"""

from .loaders import (
    load_c4,
    load_opengen,
    load_lfqa,
    load_realnews,
    load_booksum,
)

__all__ = [
    'load_c4',
    'load_opengen',
    'load_lfqa',
    'load_realnews',
    'load_booksum',
]
