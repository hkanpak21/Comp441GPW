"""
Metrics for evaluating watermark detection and text quality.
"""

from .detection import (
    compute_z_score, 
    compute_p_value, 
    compute_tpr_fpr,
    compute_auc,
    tpr_at_fpr,
    compute_detection_metrics,
)
from .quality import (
    compute_perplexity,
    compute_bertscore,
    compute_mauve,
    compute_diversity,
    compute_p_sp,
)

__all__ = [
    'compute_z_score',
    'compute_p_value', 
    'compute_tpr_fpr',
    'compute_auc',
    'tpr_at_fpr',
    'compute_detection_metrics',
    'compute_perplexity',
    'compute_bertscore',
    'compute_mauve',
    'compute_diversity',
    'compute_p_sp',
]
