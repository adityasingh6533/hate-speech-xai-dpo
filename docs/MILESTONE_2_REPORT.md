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

Furthermore, we fine-tuned a 4-bit QLoRA `Qwen2.5-1.5B-Instruct` generative model (74.71% Macro-F1) emitting structured rationales, confirmed DPO loss convergence ($0.2543 \to 0.0060$), trained a standalone DistilRoBERTa reward model (100% preference accuracy), and deployed an interactive Streamlit web dashboard.

| Metric | Score | Detail |
|---|---|---|
| **Best Test Macro-F1** | **78.06%** | Fine-tuned XLM-RoBERTa-Base |
| **Best Test AUROC** | **0.8713** | Fine-tuned XLM-RoBERTa-Base |
| **Token-F1 vs Human** | **0.7573** | LIME / SHAP Explainability ($N=100$) |
| **QLoRA Trainable Params** | **1.18%** | 18.46M / 1.5B params ($r=16, \alpha=32$) |

---

## Dataset Description & Statistics

| Partition / Class | Posts | Share | Class Breakdown / Role |
|---|---|---|---|
| **Raw Corpus (Total)** | 20,148 | 100.0% | Normal: 8,153 (40.5%) • Hate: 6,234 (30.9%) • Offensive: 5,761 (28.6%) |
| **Hate / Offensive (Combined $y=1$)** | 11,995 | 59.5% | Hateful / abusive target class |
| **Official Train (80%)** | 15,383 | 80.0% | $y=1$: 9,132 • $y=0$: 6,251 |
| **Official Validation (10%)** | 1,922 | 10.0% | Checkpoint selection & early stopping |
| **Official Test (10%)** | 1,924 | 10.0% | $y=1$: 1,142 • $y=0$: 782 |
| **Test Human Rationales** | 1,141 | 99.9% | Human rationale subset of $y=1$ test ($N=100$ XAI benchmark) |

---

## Experimental Results & Analysis

### Table 1: Held-Out Test Classification Performance ($N = 1,924$)
| Model Architecture | Macro-F1 (↑) | Accuracy (↑) | AUROC (↑) |
|---|---|---|---|
| TF-IDF (1,2-gram) + Balanced LogReg | 0.7597 | 0.7656 | 0.8381 |
| **XLM-RoBERTa-Base (Fine-Tuned)** | **0.7806** | **0.7890** | **0.8713** |
| Qwen2.5-1.5B (4-bit QLoRA SFT) | 0.7471 | 0.7552 | 0.7915 |

### Table 2: Explainability Benchmark ($N = 100$ Hateful Test Posts)
| Method | Token-F1 (↑) | IoU (↑) | Comprehensiveness (↑) | Sufficiency (↓) |
|---|---|---|---|---|
| Random Top-$k$ Baseline | 0.3265 | 0.2803 | 0.1327 | 0.3428 |
| Qwen2.5 SFT (Generative Rationales) | 0.4348 | 0.3757 | — | — |
| **LIME (XLM-R)** | **0.7573** | 0.6855 | 0.5052 | **-0.0677** |
| **SHAP (XLM-R)** | 0.7554 | **0.6874** | **0.5291** | -0.0604 |

---

## GITHUB REPOSITORY & TEAM CONTRIBUTIONS

| Team Member | Core Responsibilities & Verified Deliverables |
| :--- | :--- |
| **Aditya Singh** | QLoRA 4-bit SFT (Qwen2.5-1.5B), TRL DPO training loop, and Reward Model + PPO. |
| **Member 2 [Edit]** | HateXplain data processing, Counterfactual Data Augmentation (CDA), and fairness audit. |
| **Member 3 [Edit]** | XLM-RoBERTa fine-tuning, Hinglish benchmark curation ($N=300$), and threat modeling. |
| **Member 4 [Edit]** | SHAP & LIME XAI pipelines, Token-F1 benchmarking, and real-time token attribution. |
| **Member 5 [Edit]** | CLIP ViT-B/32 multimodal meme pipeline, patch attention heatmaps, and Streamlit app. |

> **Repo Structure:** `/notebooks/` (01 to 05), `/src/` (inference_engine.py), `/results/` (CSVs & heatmaps), `app.py` (Streamlit UI), `run_app.bat`.
