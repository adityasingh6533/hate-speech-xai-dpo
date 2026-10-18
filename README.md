# Hate Speech Detection through Explainable AI and Reward-Optimized LLMs

**Project ID:** PJ_03 (NLP Lab)  
**Author:** Aditya Kumar Singh (Bennett University)  
**Instructor:** Dr. Pradeepika Verma  

---

## 📌 Project Overview
Online content moderation requires models that are not only accurate across multilingual and code-mixed texts but also transparent, explainable, and fair. This project implements an end-to-end research pipeline that combines **multilingual transformer encoders (XLM-R)**, **post-hoc explainability (SHAP & LIME)**, and **reward-optimized generative LLMs (QLoRA SFT + Direct Preference Optimization)** to detect hate speech while providing faithful, human-aligned rationales.

---

## 🚀 Key Objectives & Architecture
1. **Multilingual Detection:** Robust hate speech and offensive language classification using `xlm-roberta-base`.
2. **Explainability Module:** Post-hoc attribution analysis via **SHAP** and **LIME**, evaluated for human plausibility (Token-F1, IoU) and model faithfulness (Comprehensiveness, Sufficiency).
3. **Reward-Optimized LLM Alignment:** Fine-tuning `Qwen/Qwen2.5-1.5B-Instruct` using 4-bit QLoRA and optimizing policy outputs via **Direct Preference Optimization (DPO)** using a multi-objective reward function (label accuracy, rationale Jaccard overlap, and anti-hacking length penalty).
4. **Multilingual & Fairness Audit:** Zero-shot cross-lingual evaluation on code-mixed Hinglish benchmarks and demographic bias audits measuring group-wise False Positive Rate (FPR) disparities across identity mentions.

---

## 📂 Repository Structure
```text
hate-speech-detection-xai-dpo/
│
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
│
├── notebooks/                 # Step-by-step Google Colab notebooks
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_xlmr_baseline_training.ipynb
│   ├── 03_explainability_shap_lime.ipynb
│   └── 04_qlora_sft_dpo_training.ipynb
│
└── results/                   # Saved experimental metrics & tables
    ├── table2_explainability_complete.csv
    ├── table4_sft_vs_dpo_ablation.csv
    ├── table3_multilingual_hinglish.csv
    └── table3_fairness_bias_audit.csv