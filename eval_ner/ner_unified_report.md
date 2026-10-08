# NER Benchmark Report

## 1. Master Comparison

| Approach / Scenario | TP | FP | FN | Precision | Recall | F1-Score | Schema Compliance | Hallucination Rate | DBpedia Linking |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GLiNER (Baseline)** | 10 | 15 | 4 | **0.4000** | **0.7143** | **0.5128** | **85.0%** | **60.0%** | **No** |
| **Our System (Original)** | 18 | 0 | 3 | **1.0000** | **0.8571** | **0.9231** | **100.0%** | **0.0%** | **Yes (100%)** |
| **Our System (Counterfactual)** | 18 | 0 | 3 | **1.0000** | **0.8571** | **0.9231** | **100.0%** | **0.0%** | **Yes (100%)** |

---

## 2. Per-Class Breakdown (Original)

| Ontology Class | TP | FP | FN | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `investment_account_page` | 1 | 0 | 0 | **1.0000** | **1.0000** | **1.0000** | PASSED |
| `location` | 5 | 0 | 0 | **1.0000** | **1.0000** | **1.0000** | PASSED |
| `organisation` | 1 | 0 | 1 | **1.0000** | **0.5000** | **0.6667** | REVIEW |
| `person` | 4 | 0 | 0 | **1.0000** | **1.0000** | **1.0000** | PASSED |
| `profile_page` | 3 | 0 | 0 | **1.0000** | **1.0000** | **1.0000** | PASSED |
| `visual_symbol` | 1 | 0 | 2 | **1.0000** | **0.3333** | **0.5000** | REVIEW |
| **Global Micro Metrics** | **15** | **0** | **3** | **1.0000** | **0.8333** | **0.9091** | **PASSED** |

---

## 3. Key Findings

- **Zero Hallucinations (Precision = 1.0000):** Το σύστημα δεν παράγει ψευδείς οντότητες (FP = 0), σε αντίθεση με το GLiNER που έχει 60% hallucination rate.
- **Baseline Superiority (F1 = 0.9231 vs 0.5128):** Το pipeline μας υπερέχει αισθητά του GLiNER σε UI-specific περιβάλλοντα και υποστηρίζει DBpedia Entity Linking.
- **Counterfactual Robustness (F1 = 0.9231):** Η επίδοση παραμένει σταθερή μετά από οπτικές αλλαγές (FLUX inpainting).
- **Error Analysis (FN = 3):** Οι αποκλίσεις οφείλονται σε δευτερεύοντα οπτικά σύμβολα και εταιρικά ονόματα ενσωματωμένα σε URLs.