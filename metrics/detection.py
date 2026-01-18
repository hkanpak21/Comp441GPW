"""
Detection Metrics for Watermark Evaluation

Metrics for evaluating watermark detection performance including
z-score, p-value, TPR, FPR, and AUC-ROC.

Based on evaluation methods from:
- Zhao et al., 2023 (Unigram-Watermark)
- Kirchenbauer et al., 2023 (KGW)
- Liang et al., 2025 (WaterPark)
"""

import numpy as np
import scipy.stats
from typing import List, Tuple, Dict, Any, Optional
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def compute_z_score(
    num_green: int, 
    total: int, 
    gamma: float = 0.5
) -> float:
    """Compute z-score for watermark detection.
    
    The z-score measures how many standard deviations the observed
    green token count is from the expected count under null hypothesis.
    
    Args:
        num_green: Number of green tokens observed
        total: Total number of tokens scored
        gamma: Expected fraction of green tokens (default: 0.5)
        
    Returns:
        Z-score statistic
        
    Note:
        z = (|G| - γT) / sqrt(T * γ * (1-γ))
        where |G| = green count, T = total, γ = gamma
        
    Example:
        >>> z = compute_z_score(num_green=150, total=200, gamma=0.5)
        >>> print(f"Z-score: {z:.2f}")
        Z-score: 7.07
    """
    if total == 0:
        return 0.0
    
    expected = gamma * total
    variance = total * gamma * (1 - gamma)
    std = np.sqrt(variance) if variance > 0 else 0.0
    
    if std == 0:
        return 0.0
    
    z = (num_green - expected) / std
    return float(z)


def compute_p_value(z_score: float, one_sided: bool = True) -> float:
    """Compute p-value from z-score.
    
    Args:
        z_score: Z-score statistic
        one_sided: If True, compute one-sided p-value (default)
                   If False, compute two-sided p-value
        
    Returns:
        P-value for hypothesis test
        
    Note:
        One-sided: P(Z > z) = 1 - Φ(z)
        Two-sided: 2 * P(Z > |z|) = 2 * (1 - Φ(|z|))
        
    Example:
        >>> p = compute_p_value(z_score=4.0)
        >>> print(f"P-value: {p:.2e}")
        P-value: 3.17e-05
    """
    if one_sided:
        return scipy.stats.norm.sf(z_score)  # sf = 1 - cdf
    else:
        return 2 * scipy.stats.norm.sf(abs(z_score))


def compute_tpr_fpr(
    watermarked_scores: List[float],
    human_scores: List[float],
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """Compute TPR and FPR at a given threshold.
    
    Args:
        watermarked_scores: Z-scores for watermarked texts
        human_scores: Z-scores for human-written texts
        threshold: Detection threshold (default: 4.0)
        
    Returns:
        Dictionary with TPR, FPR, threshold
        
    Note:
        TPR = True Positive Rate = TP / (TP + FN)
        FPR = False Positive Rate = FP / (FP + TN)
        
    Example:
        >>> wm_scores = [5.2, 6.1, 4.8, 5.5]
        >>> human_scores = [0.5, -0.3, 1.2, 0.8]
        >>> result = compute_tpr_fpr(wm_scores, human_scores, threshold=4.0)
        >>> print(f"TPR: {result['tpr']:.2f}, FPR: {result['fpr']:.2f}")
    """
    if threshold is None:
        threshold = 4.0
    
    watermarked_scores = np.array(watermarked_scores)
    human_scores = np.array(human_scores)
    
    # TPR: fraction of watermarked texts detected as watermarked
    tpr = np.mean(watermarked_scores > threshold) if len(watermarked_scores) > 0 else 0.0
    
    # FPR: fraction of human texts falsely detected as watermarked
    fpr = np.mean(human_scores > threshold) if len(human_scores) > 0 else 0.0
    
    return {
        "tpr": float(tpr),
        "fpr": float(fpr),
        "threshold": threshold,
    }


def compute_auc(
    watermarked_scores: List[float],
    human_scores: List[float]
) -> float:
    """Compute AUC-ROC score.
    
    Args:
        watermarked_scores: Z-scores for watermarked texts
        human_scores: Z-scores for human-written texts
        
    Returns:
        AUC-ROC score (0-1, higher is better)
        
    Note:
        AUC of 1.0 = perfect separation
        AUC of 0.5 = random guessing
        
    Example:
        >>> wm_scores = [5.2, 6.1, 4.8, 5.5]
        >>> human_scores = [0.5, -0.3, 1.2, 0.8]
        >>> auc_score = compute_auc(wm_scores, human_scores)
        >>> print(f"AUC: {auc_score:.3f}")
    """
    # Create labels: 1 for watermarked, 0 for human
    scores = np.concatenate([watermarked_scores, human_scores])
    labels = np.concatenate([
        np.ones(len(watermarked_scores)),
        np.zeros(len(human_scores))
    ])
    
    if len(np.unique(labels)) < 2:
        return 0.5
    
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(auc(fpr, tpr))


def tpr_at_fpr(
    watermarked_scores: List[float],
    human_scores: List[float],
    target_fpr: float = 0.01
) -> Tuple[float, float]:
    """Compute TPR at a specific FPR target.
    
    This is the primary metric used in WaterPark for comparing
    watermarking methods at a controlled false positive rate.
    
    Args:
        watermarked_scores: Z-scores for watermarked texts
        human_scores: Z-scores for human-written texts
        target_fpr: Target FPR (default: 0.01 = 1%)
        
    Returns:
        Tuple of (TPR at target FPR, threshold used)
        
    Note:
        WaterPark uses TPR@FPR=1% as primary metric.
        Unigram paper reports TPR@FPR=1% and TPR@FPR=5%.
        
    Example:
        >>> wm_scores = [5.2, 6.1, 4.8, 5.5, 3.9]
        >>> human_scores = [0.5, -0.3, 1.2, 0.8, 2.1]
        >>> tpr, threshold = tpr_at_fpr(wm_scores, human_scores, target_fpr=0.01)
        >>> print(f"TPR@FPR=1%: {tpr:.2f} (threshold={threshold:.2f})")
    """
    human_scores = np.array(human_scores)
    watermarked_scores = np.array(watermarked_scores)
    
    # Find threshold that gives target FPR on human scores
    # FPR = P(score > threshold | human)
    # We want FPR = target_fpr, so threshold = percentile(100 - target_fpr*100)
    threshold = np.percentile(human_scores, 100 * (1 - target_fpr))
    
    # Compute TPR at this threshold
    tpr = np.mean(watermarked_scores > threshold) if len(watermarked_scores) > 0 else 0.0
    
    return float(tpr), float(threshold)


def compute_detection_metrics(
    watermarked_scores: List[float],
    human_scores: List[float],
    thresholds: List[float] = None
) -> Dict[str, Any]:
    """Compute comprehensive detection metrics.
    
    Args:
        watermarked_scores: Z-scores for watermarked texts
        human_scores: Z-scores for human-written texts
        thresholds: List of thresholds to evaluate (default: [2, 3, 4, 5])
        
    Returns:
        Dictionary with:
        - auc: AUC-ROC score
        - tpr_at_fpr_1: TPR at FPR=1%
        - tpr_at_fpr_5: TPR at FPR=5%
        - tpr_at_fpr_10: TPR at FPR=10%
        - results_by_threshold: TPR/FPR at each threshold
        
    Example:
        >>> metrics = compute_detection_metrics(wm_scores, human_scores)
        >>> print(f"AUC: {metrics['auc']:.3f}")
        >>> print(f"TPR@FPR=1%: {metrics['tpr_at_fpr_1']:.3f}")
    """
    if thresholds is None:
        thresholds = [2.0, 3.0, 4.0, 5.0]
    
    results = {
        "auc": compute_auc(watermarked_scores, human_scores),
        "tpr_at_fpr_1": tpr_at_fpr(watermarked_scores, human_scores, 0.01)[0],
        "tpr_at_fpr_5": tpr_at_fpr(watermarked_scores, human_scores, 0.05)[0],
        "tpr_at_fpr_10": tpr_at_fpr(watermarked_scores, human_scores, 0.10)[0],
        "mean_wm_score": float(np.mean(watermarked_scores)),
        "mean_human_score": float(np.mean(human_scores)),
        "std_wm_score": float(np.std(watermarked_scores)),
        "std_human_score": float(np.std(human_scores)),
        "results_by_threshold": {}
    }
    
    for threshold in thresholds:
        results["results_by_threshold"][threshold] = compute_tpr_fpr(
            watermarked_scores, human_scores, threshold
        )
    
    return results
