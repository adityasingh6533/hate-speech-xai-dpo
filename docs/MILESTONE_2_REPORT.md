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

## GITHUB REPOSITORY & TEAM CONTRIBUTIONS

| Team Member | Core Responsibilities & Verified Deliverables |
| :--- | :--- |
| **Aditya Singh** | QLoRA 4-bit SFT (Qwen2.5-1.5B), TRL DPO training loop, and Reward Model + PPO. |
| **Member 2 [Edit]** | HateXplain data processing, Counterfactual Data Augmentation (CDA), and fairness audit. |
| **Member 3 [Edit]** | XLM-RoBERTa fine-tuning, Hinglish benchmark curation ($N=300$), and threat modeling. |
| **Member 4 [Edit]** | SHAP & LIME XAI pipelines, Token-F1 benchmarking, and real-time token attribution. |
| **Member 5 [Edit]** | CLIP ViT-B/32 multimodal meme pipeline, patch attention heatmaps, and Streamlit app. |

> **Repo Structure:** `/notebooks/` (01 to 05), `/src/` (inference_engine.py), `/results/` (CSVs & heatmaps), `app.py` (Streamlit UI), `run_app.bat`.
