# Counterfactual UI Ontology Evaluation Report

## Per-Scenario Triplet Results (Counterfactuals)

| Scenario / Image | Ground Truth | Predicted | Correct (TP) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `fb_profile` | 10 | 8 | 8 | 1.0000 | 0.8000 | **0.8889** |
| `viber_1` | 3 | 3 | 3 | 1.0000 | 1.0000 | **1.0000** |
| `viber_2` | 3 | 3 | 3 | 1.0000 | 1.0000 | **1.0000** |
| `dashboard` | 5 | 4 | 4 | 1.0000 | 0.8000 | **0.8889** |

## Core Academic Metrics (Counterfactuals)

### Metric 1: Overall Triplet Extraction Performance (Micro-Average)
- **Precision:** 1.0000
- **Recall:** 0.8571
- **Overall F1-Score:** **0.9231**

### Metric 2: Disaggregated Component F1-Scores
| Component | Precision | Recall | F1-Score | Focus Area |
| :--- | :---: | :---: | :---: | :--- |
| **Entity Extraction** | 1.0000 | 0.8889 | **0.9412** | Identification of UI Nodes (Subject/Object) |
| **Predicate Classification** | 1.0000 | 0.9444 | **0.9714** | Correct Relationship Mapping (Edges) |

---

## Evaluation Notes & Cheat Sheet (Counterfactual Analysis)

### Basic Variables
- **GT (Ground Truth):** Ο αριθμός των πραγματικών τριπλετών που υπάρχουν στο counterfactual dataset.
- **Pred (Predicted):** Ο αριθμός των τριπλετών που παρήγαγε το μοντέλο στις τροποποιημένες εικόνες.
- **TP (True Positives):** Οι σωστές τριπλέτες που πρόβλεψε το μοντέλο και ταυτίζονται 100% με το Counterfactual GT.

### Core Equations & Meanings
- **Precision (TP / Pred):** Πόσες από τις τριπλέτες που έβγαλε το μοντέλο είναι πραγματικά σωστές. *Υψηλό Precision = Χαμηλά hallucinations.*
- **Recall (TP / GT):** Πόσες από τις πραγματικές τριπλέτες του Counterfactual GT κατάφερε να βρει το μοντέλο.
- **F1-Score (2 * P * R / (P + R)):** Ο αρμονικός μέσος Precision & Recall για τα counterfactual σενάρια.

### Counterfactual Purpose Sheet
- **Robustness Evaluation:** Αξιολογεί την ανθεκτικότητα του μοντέλου όταν αλλάζουν συγκεκριμένες τιμές (π.χ. ονόματα, τοποθεσίες) στις εικόνες.
- **Memorization Check:** Υψηλό F1-Score στα counterfactuals αποδεικνύει ότι το μοντέλο «διαβάζει» πραγματικά την εικόνα (visual extraction) και δεν έχει απλά απομνημονεύσει τα αρχικά δεδομένα.