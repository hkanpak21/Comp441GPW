# Large Language Model Watermarking Framework

This repository contains a framework for the implementation, evaluation, and analysis of Large Language Model (LLM) watermarking techniques. The codebase includes implementations of several established baseline methods and introduces Gaussian Pancakes Watermarking (GPW), a suite of algorithms designed for enhanced robustness and semantic alignment.

## Overview

The framework provides a modular architecture for research in LLM watermarking, specifically focusing on the following areas:
1.  **Watermark Generation**: Sampling-based and post-hoc watermarking implementations.
2.  **Detection**: Statistical and semantic detection algorithms.
3.  **Robustness Analysis**: Evaluation of watermarking schemes against lexical and paraphrasing attacks.
4.  **Performance Evaluation**: Standardized metrics for assessing detectability and text quality.

## Implemented Methodologies

### Baseline Methods
-   **Unigram-Watermark** ([Zhao et al., 2023](https://arxiv.org/abs/2307.13808)): A context-free watermarking method utilizing local green lists.
-   **KGW** ([Kirchenbauer et al., 2023](https://arxiv.org/abs/2301.10226)): A method utilizing pseudo-random hashing of preceding tokens to determine green-list distributions.
-   **SEMSTAMP** ([Hou et al., 2024](https://arxiv.org/abs/2402.15006)): A sentence-level watermark utilizing Locality Sensitive Hashing (LSH) for semantic partitioning.

### Gaussian Pancakes Watermarking (GPW) Suite
-   **GPW (Basic)**: A watermarking method based on static cosine scores.
-   **GPW-SP**: GPW with Salted Phase, utilizing context-keyed phase shifts for security.
-   **GPW-SP+SR**: GPW with Salted Phase and Semantic Representation Coupling, which incorporates hidden-state information to align the watermark with the model's internal representations.

## Project Structure

```text
├── attacks/             # Lexical, paraphrase, and text-mixing attack implementations
├── data_loaders/        # Data loading utilities for LLM datasets
├── datasets/            # Dataset interface definitions
├── metrics/             # Detection (ROC, AUC) and text quality metrics
├── utils/               # Hashing and general utility functions
├── watermarkers/        # Core implementations of watermarking algorithms
│   ├── base.py          # Abstract base classes for watermarking modules
│   ├── gpw.py           # Gaussian Pancakes Watermarking implementations
│   ├── kgw.py           # KGW implementation
│   ├── semstamp.py      # SEMSTAMP implementation
│   └── unigram.py       # Unigram implementation
├── gpw_sp_contextual_cluster_wm.py  # Research script for GPW experimentation
├── Prior_Work_Experiments_Complete.ipynb # Experimental evaluation notebook
└── PRIOR_WORK_EXPERIMENTS_DOCUMENTATION.md # Documentation of experimental setup
```

## Installation

Ensure that Python 3.9 or higher is installed. The required dependencies can be installed using the following command:

```bash
pip install torch transformers datasets accelerate sentencepiece evaluate scikit-learn sentence-transformers tqdm
```

## Usage Instructions

### Experimental Evaluation
The primary method for conducting experiments is through the provided Jupyter notebook:
-   [Prior_Work_Experiments_Complete.ipynb](Prior_Work_Experiments_Complete.ipynb)

The primary experiment script can also be executed directly:
```bash
python gpw_sp_contextual_cluster_wm.py
```

### Extending the Framework
New watermarking methods should inherit from the `BaseWatermarker` class defined in [watermarkers/base.py](watermarkers/base.py). Implementation of the `generate` and `detect` methods is required.

```python
from watermarkers.base import BaseWatermarker

class CustomWatermarker(BaseWatermarker):
    def generate(self, input_ids, **kwargs):
        # Generation implementation
        pass
        
    def detect(self, text, **kwargs):
        # Detection implementation
        pass
```

## Evaluation Metrics

The framework employs the following metrics for evaluation:
-   **Detection Accuracy**: z-score analysis, Area Under the ROC Curve (AUC-ROC), and True Positive Rate (TPR) at fixed False Positive Rates (FPR).
-   **Text Quality**: Perplexity measurements (via GPT-2 or Llama), Sentence-BERT cosine similarity, and edit distance.

## References

-   Kirchenbauer, J., et al. (2023). A Watermark for Large Language Models.
-   Zhao, X., et al. (2023). Provable Robust Watermarking for AI-Generated Text.
-   Hou, Y., et al. (2024). SemStamp: A Semantic Watermark with LSH.
-   Liang, Y., et al. (2025). WaterPark: A Unified Benchmark for LLM Watermarking.

---
*Developed as part of the Comp441 Fall 2025 Research Project.*
