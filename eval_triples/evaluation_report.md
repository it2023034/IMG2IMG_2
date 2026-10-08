# VLM Triplet Extraction Benchmark

## 1. Model Comparison

| Model | GT | Pred | TP | FP | FN | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gemma 3 12B** | 21 | 18 | 18 | 0 | 3 | **1.0000** | **0.8571** | **0.9231** |
| **LLava 13B** | 21 | 13 | 6 | 7 | 15 | **0.4615** | **0.2857** | **0.3529** |

---

## 2. Component Analysis

| Model | Component | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: |
| **Gemma 3 12B** | Entity Extraction | 1.0000 | 0.8889 | **0.9412** |
| **Gemma 3 12B** | Predicate Classification | 1.0000 | 0.9444 | **0.9714** |
| **LLava 13B** | Entity Extraction | 0.7059 | 0.4444 | **0.5455** |
| **LLava 13B** | Predicate Classification | 0.6923 | 0.5000 | **0.5806** |