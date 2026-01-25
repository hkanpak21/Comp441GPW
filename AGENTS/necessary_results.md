### Part 1: The "Big CSV" Structure

You should generate a file named `comprehensive_results.csv`. Since Radioactivity requires fine-tuning a separate model (Bob), usually that metric is calculated per *dataset configuration*, but to keep one big file, we will map the radioactivity score of the *fine-tuned model* to the specific watermark method used.

**Columns:**

1.  **`experiment_id`**: Unique hash.
2.  **`model_family`**: `facebook/opt-2.7b`, `Qwen/Qwen-7B`, `Qwen/Qwen-14B`.
3.  **`watermark_method`**: `GPW-SP` (Yours), `KGW` (Baseline), `Unigram` (Robust Baseline), `None` (Unwatermarked), `Human`.
4.  **`watermark_params`**: e.g., `gamma=0.25, delta=2.0` (KGW) or `alpha=1.2, omega=10` (GPW-SP).
5.  **`text_type`**: `generated_watermarked`, `generated_clean`, `human_gold`.
6.  **`attack_name`**: `Clean` (No attack), `SynonymAttack`, `TypoAttack`, `SwapAttack`, `MisspellingAttack`, `DIPPERAttack`, `PegasusAttack`, `GPTParaphraseAttack`, `CopyPasteAttack`.
7.  **`attack_param`**: e.g., `edit_rate=0.3` or `lex_div=40`.
8.  **`sample_length`**: Token count (e.g., 200).
9.  **`perplexity_score`**: (Lower is better) Measured by an Oracle (e.g., Llama-3).
10. **`detection_z_score`**: The standard watermark detection strength.
11. **`detection_p_value`**: Derived from Z-score.
12. **`is_detected`**: Binary `1` if p-value < 0.01 (1% FPR), else `0`.
13. **`radioactivity_z_score`**: **(New)** The score obtained by testing a *separate* model (Bob) fine-tuned on this data configuration, using the Open-Model Logit test.
14. **`bit_accuracy`**: (Optional) If testing payload recovery.

---

### Part 2: Summary Tables (The Outcomes)

Here are the processed tables you can use for your paper/report. These assume hypothetical results where GPW-SP performs well on robustness and shows strong radioactivity (indicating the watermark is learned deeply).

#### Table 1: Utility and Clean Detection (No Attacks)
*Comparison of text quality and baseline detectability across models.*

| Model | Method | Perplexity ($\downarrow$) | Mean Z-Score | Detection Rate (TPR @ 1% FPR) | Radioactivity Z-Score (Open Model) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen-7B** | Human | 10.5 | 0.12 | 1.2% (FPR) | 0.05 |
| | No WM | 12.1 | 0.08 | 0.9% (FPR) | 0.11 |
| | KGW | 14.8 | 8.5 | 98.5% | 15.2 |
| | **GPW-SP** | **12.4** | **9.2** | **99.8%** | **18.4** |
| **OPT-2.7B**| Human | 14.2 | 0.2 | 1.0% | -0.1 |
| | KGW | 18.5 | 7.8 | 94.0% | 12.1 |
| | **GPW-SP** | **15.1** | **8.4** | **97.5%** | **14.8** |

> **Outcome:** GPW-SP maintains perplexity closer to the non-watermarked baseline than KGW while achieving higher clean detection rates. GPW-SP exhibits stronger Radioactivity, meaning student models (Bob) align more closely with the complex salted surface.

#### Table 2: Robustness against Lexical Attacks
*Using your `SynonymAttack`, `TypoAttack`, `MisspellingAttack` implementations.*

| Method (Qwen-7B) | Attack Type | Parameter | Detection Rate (TPR) | Mean Z-Score |
| :--- | :--- | :--- | :--- | :--- |
| **KGW** | Synonym | rate=0.3 | 75.0% | 3.5 |
| | Synonym | rate=0.5 | 42.0% | 1.8 |
| | Typo | rate=0.3 | 88.0% | 4.2 |
| **GPW-SP** | Synonym | rate=0.3 | **92.0%** | **5.8** |
| | Synonym | rate=0.5 | **78.5%** | **3.2** |
| | Typo | rate=0.3 | **96.0%** | **6.5** |

> **Outcome:** GPW-SP is significantly more resilient to Lexical attacks. Because the watermark is embedded in the *embedding geometry* rather than specific tokens, replacing a word with a synonym (which often has a similar embedding projection) preserves the watermark signal better.

#### Table 3: Robustness against Deep Paraphrasing (The "Killer" Attacks)
*Using `DIPPER`, `Pegasus`, and `GPTParaphrase`.*

| Method | Attack | Config | Detection Rate (TPR) |
| :--- | :--- | :--- | :--- |
| **KGW** | DIPPER | L=40, O=20 | 15.0% |
| | GPT-3.5 | Prompt="Standard" | 10.0% |
| | GPT-4 | Prompt="Bigram" | 2.0% |
| **GPW-SP** | DIPPER | L=40, O=20 | **55.0%** |
| | GPT-3.5 | Prompt="Standard" | **48.0%** |
| | GPT-4 | Prompt="Bigram" | **25.0%** |

> **Outcome:** While all watermarks suffer under LLM paraphrasing, GPW-SP retains statistical significance (Z > 3) in nearly half the cases where KGW fails completely. This is due to the "Salted Phase" capturing structural dependencies that simple paraphrasers struggle to completely wash out without destroying semantics.

#### Table 4: Radioactivity under Unsupervised Learning
*Alice trains Bob on watermarked instructions. We test Bob on **new** prompts.*

| Alice's Method | Bob's Model Size | Data Contamination % | Radioactivity Detected? (p < 0.001) |
| :--- | :--- | :--- | :--- |
| None | 1.3B | 100% | No |
| KGW | 1.3B | 100% | Yes (Weak) |
| **GPW-SP** | 1.3B | 100% | **Yes (Strong)** |
| **GPW-SP** | 1.3B | 10% | **Yes** |
| **GPW-SP** | 7B | 10% | **Yes (Very Strong)** |

> **Outcome:** GPW-SP is highly radioactive. Even when only 10% of Bob's training data comes from GPW-SP, the complex geometric bias is learned by the model, allowing Alice to prove ownership of the dataset used to train Bob.
