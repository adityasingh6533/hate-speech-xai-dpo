# Hate Speech Detection through Explainable AI and Reward-Optimized LLMs

**Project ID:** PJ_03 (CSET-346 NLP Lab, Milestone-2 Evaluation)  
**Lead:** Aditya Kumar Singh & Team (Bennett University)  
**Instructor:** Dr. Pradeepika Verma  
**Full Report:** [docs/MILESTONE_2_REPORT.md](docs/MILESTONE_2_REPORT.md)

---

## 📌 Project Overview
Online content moderation requires models that are not only accurate across multilingual and code-mixed texts but also transparent, explainable, and fair. This project implements an end-to-end research pipeline that combines **multilingual transformer encoders (XLM-R)**, **post-hoc explainability (SHAP & LIME)**, **reward-optimized generative LLMs (4-bit QLoRA SFT + Direct Preference Optimization & PPO)**, **counterfactual fairness audits (CDA)**, and **multimodal CLIP meme detection** with spatial patch attention heatmaps.

---

## 👥 Team Roles & Verified Deliverables

| Team Member | Core Responsibilities & Verified Deliverables |
| :--- | :--- |
| **Aditya Singh** | QLoRA 4-bit SFT (Qwen2.5-1.5B), TRL DPO training loop, and Reward Model + PPO. |
| **Member 2 [Edit]** | HateXplain data processing, Counterfactual Data Augmentation (CDA), and fairness audit. |
| **Member 3 [Edit]** | XLM-RoBERTa fine-tuning, Hinglish benchmark curation ($N=300$), and threat modeling. |
| **Member 4 [Edit]** | SHAP & LIME XAI pipelines, Token-F1 benchmarking, and real-time token attribution. |
| **Member 5 [Edit]** | CLIP ViT-B/32 multimodal meme pipeline, patch attention heatmaps, and Streamlit app. |

---

## 🚀 Key Results & Benchmarks

| Milestone Benchmark | Key Metric | Result / Status |
|---|---|---|
| **XLM-RoBERTa Fine-Tuned** | Test Macro-F1 / AUROC | **78.06% F1** / **0.8713 AUROC** |
| **Explainability (SHAP/LIME)** | Human Token-F1 ($N=100$) | **0.7573 Token-F1** (vs. 0.3265 random) |
| **QLoRA SFT + DPO Policy** | DPO Loss Convergence | **$0.2543 \to 0.0060$** (Qwen2.5-1.5B) |
| **Reward Model (DistilRoBERTa)** | Preference Pair Accuracy | **100.0%** (PPO mean reward: +0.5430) |
| **Counterfactual Invariance (CDA)** | Counterfactual Flip Rate (CFR) | **$0.1324 \to 0.0197$** (85.1% flip reduction) |
| **Multilingual Hinglish ($N=300$)** | Macro-F1 / Accuracy | **0.5409 F1** / **63.67% Acc** |
| **Multimodal Memes (CLIP ViT-B/32)** | Held-out Accuracy | **87.50%** ($7 \times 7$ patch attention maps) |

---

## 💻 Running the Live Interactive Web Application

Launch the real-time AI moderation dashboard (supports live English & Hinglish inference, Leave-One-Out token XAI charts, counterfactual identity sandbox, and multimodal meme image analysis):

```powershell
python -m streamlit run app.py
```
Or double-click:
```powershell
.\run_app.bat
```

---

## 📂 Repository Structure
```text
hate-speech-xai-dpo/
│
├── app.py                                         # 4-tab Interactive Streamlit Web Application
├── run_app.bat / run_app.ps1                      # Application launchers
├── requirements.txt                               # Environment dependencies
├── README.md                                      # Project overview & documentation
│
├── docs/                                          # Milestone-2 evaluation reports & presentation
│   └── MILESTONE_2_REPORT.md                      # Full CSET-346 Milestone-2 Progress Report
│
├── src/                                           # Production Inference Engine
│   ├── __init__.py
│   └── inference_engine.py                        # Pretrained models, Leave-One-Out XAI & CLIP
│
├── notebooks/                                     # Google Colab Research Pipeline
│   ├── 01_data_preprocessing.ipynb                # HateXplain EDA & CDA generation
│   ├── 02_xlmr_baseline_training.ipynb            # Multilingual XLM-R fine-tuning
│   ├── 03_explainability_shap_lime.ipynb          # SHAP/LIME Token-F1 & Faithfulness
│   ├── 04_qlora_sft_dpo_training.ipynb            # Qwen2.5-1.5B QLoRA SFT & DPO
│   └── 05_dpo_ppo_fairness_multimodal.ipynb       # Reward Model, PPO, Fairness & CLIP
│
└── results/                                       # Checkpointed experimental tables & figures
    ├── table2_explainability_complete.csv         # SHAP/LIME faithfulness benchmarks
    ├── table3_fairness_bias_audit.csv             # 10-group demographic FPR audit
    ├── table3_multilingual_hinglish.csv           # Code-mixed Hinglish evaluation
    ├── table4_sft_vs_dpo_ablation.csv             # SFT vs DPO vs PPO policy comparison
    ├── table5_multimodal_clip.csv                 # Multimodal meme metrics
    ├── phase3_qualitative_examples.json           # Qualitative rationale predictions
    └── phase7_image_region_attention_heatmap.png  # ViT 7x7 spatial patch attention heatmap
```
