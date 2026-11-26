# CSET-346: Natural Language Processing • Milestone-2 Progress Report
**Project ID:** PJ_03 • **Date:** November 25–26, 2026  
**Title:** Explainable & Alignment-Aware Hate Speech Detection Using Multilingual Transformers and Rationale-Guided Large Language Models  
**Team:** Aditya Singh & Team Members | **Venue:** P2-CUB-096 | **GitHub:** [github.com/adityasingh6533/hate-speech-xai-dpo](https://github.com/adityasingh6533/hate-speech-xai-dpo)

---

## Executive Summary
Effective content moderation requires both accurate detection and faithful, human-aligned explanations. For Milestone-2, we built and benchmarked a multi-stage NLP pipeline on the **HateXplain** benchmark (20,148 posts; 19,229 official split posts). 

Fine-tuning `xlm-roberta-base` substantially outperformed a balanced TF-IDF + Logistic Regression baseline:
- **Baseline (TF-IDF + LogReg):** 75.97% Macro-F1, 0.8381 AUROC
- **Fine-Tuned XLM-RoBERTa:** **78.06% Macro-F1**, **78.90% Accuracy**, and **0.8713 AUROC**

Post-hoc explainability via **SHAP** and **LIME** achieved **0.755–0.757 Token-F1** against human ground-truth rationales (compared to 0.327 for random top-$k$), exhibiting strong model faithfulness (**Comprehensiveness: 0.529**, **Sufficiency: -0.068**). 

Furthermore, we fine-tuned a 4-bit QLoRA `Qwen2.5-1.5B-Instruct` generative model (74.71% Macro-F1) that emits structured natural language rationales, confirmed custom Direct Preference Optimization (DPO) convergence (loss $0.2543 \to 0.0060$), trained a standalone DistilRoBERTa reward model (100% preference accuracy), and deployed an interactive Streamlit web dashboard with live token attribution, counterfactual fairness auditing, and multimodal CLIP meme classification.

| Metric | Score | Detail |
|---|---|---|
| **Best Test Macro-F1** | **78.06%** | Fine-tuned XLM-RoBERTa-Base |
| **Best Test AUROC** | **0.8713** | Fine-tuned XLM-RoBERTa-Base |
| **Token-F1 vs Human** | **0.7573** | LIME / SHAP Explainability ($N=100$) |
| **QLoRA Trainable Params** | **1.18%** | 18.46M / 1.5B params ($r=16, \alpha=32$) |

---

## End-to-End System Workflow & Milestone-2 Architecture

```
[Stage 1: DONE]               [Stage 2: DONE]                 [Stage 3: DONE]               [Stage 4: ACTIVE]
Dataset & TF-IDF Baseline ──► Multilingual Transformer ────► Post-Hoc Explainability ────► Rationale LLM & Extensions
• HateXplain JSON parsing     • Fine-tune XLM-RoBERTa-Base    • Whitespace SHAP & LIME      • 4-bit QLoRA SFT (Qwen2.5)
• Majority vote & binary map  • FP16, 3 epochs, lr=2e-5       • Token-F1, IoU plausibility  • TRL DPO + PPO RL alignment
• 50k (1,2)-gram + LogReg     • Best val Macro-F1 selection   • Comp & Suff faithfulness    • Hinglish, CDA & CLIP Memes
```

---

## Dataset Description & Statistics

### Why HateXplain?
We use HateXplain (*Mathew et al., AAAI 2021*; sourced from Twitter and Gab), which uniquely provides:
1. **3-annotator ground truth** (`normal`, `hatespeech`, `offensive`).
2. **Target community metadata** enabling demographic fairness and bias audits.
3. **Human token-level rationales** enabling quantitative explainability benchmarking.

### Split Breakdown
| Partition / Class | Posts | Share | Class Breakdown / Role |
|---|---|---|---|
| **Raw Corpus (Total)** | 20,148 | 100.0% | Normal: 8,153 (40.5%) • Hate: 6,234 (30.9%) • Offensive: 5,761 (28.6%) |
| **Hate / Offensive (Combined $y=1$)** | 11,995 | 59.5% | Hateful / abusive target class |
| **Official Train (80%)** | 15,383 | 80.0% | $y=1$: 9,132 • $y=0$: 6,251 |
| **Official Validation (10%)** | 1,922 | 10.0% | Checkpoint selection & early stopping |
| **Official Test (10%)** | 1,924 | 10.0% | $y=1$: 1,142 • $y=0$: 782 |
| **Test Human Rationales** | 1,141 | 99.9% | Human rationale subset of $y=1$ test ($N=100$ XAI benchmark) |

*Preprocessing note: 919 ambiguous posts with three-way tie annotator disagreements were discarded to prevent label noise.*

---

## Methodology & Implementation

### 1. Environment & Hardware
- **Stack:** PyTorch 2.x, Hugging Face `transformers`, `datasets`, `peft`, `bitsandbytes`, `trl`, `shap`, `lime`, `streamlit`.
- **Compute:** NVIDIA T4 GPU (12–15 GB VRAM) on Google Colab with Google Drive checkpoint persistence.

### 2. Lexical Baseline (TF-IDF + Balanced Logistic Regression)
- Word unigrams and bigrams (`ngram_range=(1,2)`, `max_features=50,000`).
- L2-regularized Logistic Regression with `class_weight='balanced'` to establish a transparent baseline against the 59:41 class ratio.

### 3. Fine-Tuned Multilingual Encoder (XLM-RoBERTa-Base)
- Fine-tuned `xlm-roberta-base` (278M parameters) with a 2-class classification head for 3 epochs (batch size 16, AdamW lr = $2\times 10^{-5}$, max sequence length 128, FP16 mixed precision).
- Best model selected strictly based on validation Macro-F1.

### 4. Post-Hoc Explainability (SHAP & LIME)
- Evaluated on 100 randomly sampled hateful test posts with valid human rationales (annotator mean $\ge 0.5$).
- Both `shap.maskers.Text` and `LimeTextExplainer(bow=False)` aligned on whitespace tokens with adaptive top-$k$ equal to human rationale length.
- Scored via:
  - **Plausibility:** Token-F1 and Intersection-over-Union (IoU) against human annotations.
  - **Faithfulness:** Comprehensiveness ($\Delta P$ removing top-$k$; higher is better) and Sufficiency ($\Delta P$ keeping only top-$k$; lower is better).

### 5. Generative Rationale LLM (4-bit QLoRA SFT & DPO / PPO)
- `Qwen2.5-1.5B-Instruct` quantized to 4-bit NF4 with LoRA adapters ($r=16, \alpha=32$ across all attention and MLP projection layers; 18.46M trainable parameters = 1.18%).
- Supervised fine-tuning to emit structured explanations: `Label: HATEFUL|NOT_HATEFUL` plus `Reason: targets <group>; key words: <spans>`.
- Aligned via TRL Direct Preference Optimization (DPO) and standalone DistilRoBERTa Reward Model + PPO reinforcement learning.

### 6. Exploratory & Multimodal Tracks
- **Hinglish Code-Mixed Benchmark:** $N=300$ unique code-mixed evaluation posts (Macro-F1 0.5409, Acc 63.67%, AUROC 0.6314).
- **Counterfactual Fairness Audit:** Counterfactual Data Augmentation (CDA) with 5,981 twins, reducing Counterfactual Flip Rate (CFR) from $0.1324 \to 0.0197$.
- **Multimodal CLIP:** `openai/clip-vit-base-patch32` fusion on 80 held-out memes (87.50% accuracy) with $7 \times 7$ Vision Transformer spatial patch attention heatmaps.

---

## Experimental Results & Analysis

### Table 1: Held-Out Test Classification Performance ($N = 1,924$)
| Model Architecture | Macro-F1 (↑) | Accuracy (↑) | AUROC (↑) |
|---|---|---|---|
| TF-IDF (1,2-gram) + Balanced LogReg | 0.7597 | 0.7656 | 0.8381 |
| **XLM-RoBERTa-Base (Fine-Tuned)** | **0.7806** | **0.7890** | **0.8713** |
| Qwen2.5-1.5B (4-bit QLoRA SFT) | 0.7471 | 0.7552 | 0.7915 |

*Analysis: Fine-tuned XLM-RoBERTa outperforms TF-IDF by +2.09 Macro-F1 and +3.32 AUROC points. Qwen2.5-1.5B SFT achieves competitive classification (0.7471 Macro-F1) while uniquely generating native human-readable rationales.*

### Table 2: Explainability Benchmark ($N = 100$ Hateful Test Posts)
| Method | Token-F1 (↑) | IoU (↑) | Comprehensiveness (↑) | Sufficiency (↓) |
|---|---|---|---|---|
| Random Top-$k$ Baseline | 0.3265 | 0.2803 | 0.1327 | 0.3428 |
| Qwen2.5 SFT (Generative Rationales) | 0.4348 | 0.3757 | — | — |
| **LIME (XLM-R)** | **0.7573** | 0.6855 | 0.5052 | **-0.0677** |
| **SHAP (XLM-R)** | 0.7554 | **0.6874** | **0.5291** | -0.0604 |

*Analysis: SHAP and LIME outperform random attribution by over +43 Token-F1 points. Negative Sufficiency confirms that the extracted tokens alone are sufficient to maintain or increase model toxicity confidence.*

---

## Challenges Encountered & Solutions
1. **GPU Session Resets & Memory Ceiling:** Solved 12 GB VRAM limitations using 4-bit NF4 QLoRA and Google Drive persistence checkpoints.
2. **Library API Breaking Changes:** TRL DPOTrainer version mismatch resolved by standardizing parameter signatures and gradient accumulation steps.
3. **Cross-Lingual Domain Shift:** Standard English models failed on Romanized Hinglish insults; resolved by building a code-mixed semantic layer and Hinglish benchmark.
4. **Interactive Deployment & Latency:** Built an asynchronous, cached Streamlit pipeline (`@st.cache_resource`) ensuring sub-second inference.

---

## GITHUB REPOSITORY & TEAM CONTRIBUTIONS

| Team Member | Core Responsibilities & Verified Deliverables |
| :--- | :--- |
| **Aditya Singh** | **Lead ML Architect & LLM Alignment:**<br>Colab GPU pipeline, Qwen2.5-1.5B 4-bit QLoRA SFT ($r=16, \alpha=32$), TRL DPO training loop (loss: $0.2543 \to 0.0060$), DistilRoBERTa Reward Model (100% acc) & PPO RL (+0.5430 mean reward). |
| **Member 2 [Edit]** | **Data Engineering & Counterfactual Fairness:**<br>HateXplain preprocessing, multi-annotator agreement rules, Counterfactual Data Augmentation (CDA with 5,981 twins; $N=21,364$), 10-group FPR disparity audit & CFR metric ($0.1324 \to 0.0197$). |
| **Member 3 [Edit]** | **Multilingual NLP & Hinglish Evaluation:**<br>XLM-RoBERTa cross-lingual fine-tuning, zero-duplicate Hinglish benchmark curation ($N=300$; Macro-F1: 0.5409, Acc: 63.67%), and code-mixed threat phrase / colloquial dampener modeling. |
| **Member 4 [Edit]** | **Explainable AI (XAI) & Interpretability:**<br>SHAP & LIME token-attribution pipelines, evaluation against HateXplain human rationales (Sufficiency, Comprehensiveness, AOPC), and Leave-One-Out marginal perturbation ($\Delta P$) attribution engine. |
| **Member 5 [Edit]** | **Multimodal Vision-Language & Web Deployment:**<br>CLIP ViT-B/32 cross-modal fusion on memes (held-out acc: 0.8750), Vision Transformer $7 \times 7$ spatial patch attention heatmap overlay, and interactive Streamlit web dashboard (`app.py`). |

> **Repo Structure:** `/notebooks/` (01_data_preprocessing, 02_xlmr_baseline, 03_explainability_shap_lime, 04_qlora_sft_dpo, 05_dpo_ppo_fairness_multimodal), `/src/` (inference_engine.py), `/results/` (CSVs, qualitative JSON & attention heatmaps), `app.py` (Interactive Streamlit App), `run_app.bat`, `/docs/` (Milestone-2 report & PPT).
