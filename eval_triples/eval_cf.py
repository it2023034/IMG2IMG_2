"""
Counterfactual UI Ontology Evaluation Script with Embedded Explanatory Notes.
Generates evaluation_report_cf.md containing counterfactual metrics and reference notes.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
GT_PATH = BASE_DIR / "gt_ui_cf.json"
PRED_PATH = BASE_DIR / "results_cf.json"
REPORT_PATH = BASE_DIR / "evaluation_report_cf.md"


def extract_components(data):
    """Extracts entities, predicates, and full triplets as lowercased sets."""
    entities = set()
    predicates = set()
    triplets = set()

    for rel in data.get("relations", []):
        sub = str(rel.get("subject", "")).strip().lower()
        pred = str(rel.get("predicate", "")).strip().lower()
        obj = str(rel.get("object", "")).strip().lower()

        if sub and pred and obj:
            entities.add(sub)
            entities.add(obj)
            predicates.add(pred)
            triplets.add((sub, pred, obj))

    return entities, predicates, triplets


def calc_f1(tp, pred_count, gt_count):
    """Calculates Precision, Recall, and F1-Score."""
    p = tp / pred_count if pred_count > 0 else 0.0
    r = tp / gt_count if gt_count > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def main():
    if not GT_PATH.exists() or not PRED_PATH.exists():
        print(f"[ERROR] Missing {GT_PATH.name} or {PRED_PATH.name}!")
        return

    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    with open(PRED_PATH, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    tot_trip_tp, tot_trip_pred, tot_trip_gt = 0, 0, 0
    tot_ent_tp, tot_ent_pred, tot_ent_gt = 0, 0, 0
    tot_pred_tp, tot_pred_pred, tot_pred_gt = 0, 0, 0

    md_lines = [
        "# Counterfactual UI Ontology Evaluation Report\n",
        "## Per-Scenario Triplet Results (Counterfactuals)\n",
        "| Scenario / Image | Ground Truth | Predicted | Correct (TP) | Precision | Recall | F1-Score |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for img_key, gt_content in gt_data.items():
        gt_ent, gt_pred, gt_triplets = extract_components(gt_content)
        pred_ent, pred_pred, pred_triplets = extract_components(
            pred_data.get(img_key, {})
        )

        tp_trip = len(gt_triplets.intersection(pred_triplets))
        p_trip, r_trip, f1_trip = calc_f1(
            tp_trip, len(pred_triplets), len(gt_triplets)
        )

        tot_trip_tp += tp_trip
        tot_trip_pred += len(pred_triplets)
        tot_trip_gt += len(gt_triplets)

        tot_ent_tp += len(gt_ent.intersection(pred_ent))
        tot_ent_pred += len(pred_ent)
        tot_ent_gt += len(gt_ent)

        tot_pred_tp += len(gt_pred.intersection(pred_pred))
        tot_pred_pred += len(pred_pred)
        tot_pred_gt += len(gt_pred)

        md_lines.append(
            f"| `{img_key}` | {len(gt_triplets)} | {len(pred_triplets)} | {tp_trip} | {p_trip:.4f} | {r_trip:.4f} | **{f1_trip:.4f}** |"
        )

    micro_p, micro_r, micro_f1 = calc_f1(
        tot_trip_tp, tot_trip_pred, tot_trip_gt
    )
    ent_p, ent_r, ent_f1 = calc_f1(tot_ent_tp, tot_ent_pred, tot_ent_gt)
    pred_p, pred_r, pred_f1 = calc_f1(tot_pred_tp, tot_pred_pred, tot_pred_gt)

    md_lines.extend(
        [
            "\n## Core Academic Metrics (Counterfactuals)\n",
            "### Metric 1: Overall Triplet Extraction Performance (Micro-Average)",
            f"- **Precision:** {micro_p:.4f}",
            f"- **Recall:** {micro_r:.4f}",
            f"- **Overall F1-Score:** **{micro_f1:.4f}**\n",
            "### Metric 2: Disaggregated Component F1-Scores",
            "| Component | Precision | Recall | F1-Score | Focus Area |",
            "| :--- | :---: | :---: | :---: | :--- |",
            f"| **Entity Extraction** | {ent_p:.4f} | {ent_r:.4f} | **{ent_f1:.4f}** | Identification of UI Nodes (Subject/Object) |",
            f"| **Predicate Classification** | {pred_p:.4f} | {pred_r:.4f} | **{pred_f1:.4f}** | Correct Relationship Mapping (Edges) |",
            "\n---\n",
            "## Evaluation Notes & Cheat Sheet (Counterfactual Analysis)\n",
            "### Basic Variables",
            "- **GT (Ground Truth):** Ο αριθμός των πραγματικών τριπλετών που υπάρχουν στο counterfactual dataset.",
            "- **Pred (Predicted):** Ο αριθμός των τριπλετών που παρήγαγε το μοντέλο στις τροποποιημένες εικόνες.",
            "- **TP (True Positives):** Οι σωστές τριπλέτες που πρόβλεψε το μοντέλο και ταυτίζονται 100% με το Counterfactual GT.\n",
            "### Core Equations & Meanings",
            "- **Precision (TP / Pred):** Πόσες από τις τριπλέτες που έβγαλε το μοντέλο είναι πραγματικά σωστές. *Υψηλό Precision = Χαμηλά hallucinations.*",
            "- **Recall (TP / GT):** Πόσες από τις πραγματικές τριπλέτες του Counterfactual GT κατάφερε να βρει το μοντέλο.",
            "- **F1-Score (2 * P * R / (P + R)):** Ο αρμονικός μέσος Precision & Recall για τα counterfactual σενάρια.\n",
            "### Counterfactual Purpose Sheet",
            "- **Robustness Evaluation:** Αξιολογεί την ανθεκτικότητα του μοντέλου όταν αλλάζουν συγκεκριμένες τιμές (π.χ. ονόματα, τοποθεσίες) στις εικόνες.",
            "- **Memorization Check:** Υψηλό F1-Score στα counterfactuals αποδεικνύει ότι το μοντέλο «διαβάζει» πραγματικά την εικόνα (visual extraction) και δεν έχει απλά απομνημονεύσει τα αρχικά δεδομένα.",
        ]
    )

    full_report = "\n".join(md_lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(full_report)
    print(f"\n[SUCCESS] Counterfactual report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()