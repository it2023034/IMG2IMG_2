"""
UI Ontology Evaluation Script.
Evaluates Triplet Knowledge Extraction across VLM Backbones.
Generates evaluation_report.md.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
GT_PATH = BASE_DIR / "gt_ui.json"
RUNS_DIR = BASE_DIR / "runs"
REPORT_PATH = BASE_DIR / "evaluation_report.md"

# VLM Μοντέλα προς σύγκριση
MODELS_TO_EVAL = {
    "Gemma 3 12B": RUNS_DIR / "results_gemma.json",
    "LLava 13B": RUNS_DIR / "results_llava.json",
}


def extract_components(data):
    """Extracts entities, predicates, and full triplets as lowercased sets."""
    entities = set()
    predicates = set()
    triplets = set()

    relations = data.get("relations", []) if isinstance(data, dict) else []

    for rel in relations:
        if not isinstance(rel, dict):
            continue

        sub = str(rel.get("subject", "")).strip().lower()
        pred = str(rel.get("predicate", "")).strip().lower()
        obj = str(rel.get("object", "")).strip().lower()

        if sub and pred and obj:
            entities.add(sub)
            entities.add(obj)
            predicates.add(pred)
            triplets.add((sub, pred, obj))

    return entities, predicates, triplets


def calc_metrics(tp, fp, fn):
    """Calculates Precision, Recall, and F1-Score."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def evaluate_single_model(gt_data, pred_data):
    """Evaluates a model output against Ground Truth."""
    tot_trip_tp, tot_trip_fp, tot_trip_fn = 0, 0, 0
    tot_ent_tp, tot_ent_fp, tot_ent_fn = 0, 0, 0
    tot_pred_tp, tot_pred_fp, tot_pred_fn = 0, 0, 0

    for img_key, gt_content in gt_data.items():
        gt_ent, gt_pred, gt_triplets = extract_components(gt_content)
        pred_content = pred_data.get(img_key, {})
        pred_ent, pred_pred, pred_triplets = extract_components(pred_content)

        # Triplets TP, FP, FN
        tp_trip = len(gt_triplets.intersection(pred_triplets))
        fp_trip = len(pred_triplets - gt_triplets)
        fn_trip = len(gt_triplets - pred_triplets)

        tot_trip_tp += tp_trip
        tot_trip_fp += fp_trip
        tot_trip_fn += fn_trip

        # Entities TP, FP, FN
        tot_ent_tp += len(gt_ent.intersection(pred_ent))
        tot_ent_fp += len(pred_ent - gt_ent)
        tot_ent_fn += len(gt_ent - pred_ent)

        # Predicates TP, FP, FN
        tot_pred_tp += len(gt_pred.intersection(pred_pred))
        tot_pred_fp += len(pred_pred - gt_pred)
        tot_pred_fn += len(gt_pred - pred_pred)

    micro_p, micro_r, micro_f1 = calc_metrics(
        tot_trip_tp, tot_trip_fp, tot_trip_fn
    )
    ent_p, ent_r, ent_f1 = calc_metrics(tot_ent_tp, tot_ent_fp, tot_ent_fn)
    pred_p, pred_r, pred_f1 = calc_metrics(
        tot_pred_tp, tot_pred_fp, tot_pred_fn
    )

    return {
        "overall": {
            "tp": tot_trip_tp,
            "fp": tot_trip_fp,
            "fn": tot_trip_fn,
            "pred": tot_trip_tp + tot_trip_fp,
            "gt": tot_trip_tp + tot_trip_fn,
            "precision": micro_p,
            "recall": micro_r,
            "f1": micro_f1,
        },
        "entities": {"precision": ent_p, "recall": ent_r, "f1": ent_f1},
        "predicates": {"precision": pred_p, "recall": pred_r, "f1": pred_f1},
    }


def main():
    if not GT_PATH.exists():
        print("[ERROR] Missing gt_ui.json!")
        return

    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    md_lines = [
        "# VLM Triplet Extraction Benchmark\n",
        "## 1. Model Comparison\n",
        "| Model | GT | Pred | TP | FP | FN | Precision | Recall | F1-Score |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    model_results = {}

    for model_name, res_path in MODELS_TO_EVAL.items():
        if not res_path.exists():
            print(f"[WARNING] Results file not found for {model_name}: {res_path}")
            continue

        with open(res_path, "r", encoding="utf-8") as f:
            pred_data = json.load(f)

        metrics = evaluate_single_model(gt_data, pred_data)
        model_results[model_name] = metrics

        ov = metrics["overall"]
        md_lines.append(
            f"| **{model_name}** | {ov['gt']} | {ov['pred']} | {ov['tp']} | {ov['fp']} | {ov['fn']} | **{ov['precision']:.4f}** | **{ov['recall']:.4f}** | **{ov['f1']:.4f}** |"
        )

    if model_results:
        md_lines.extend(
            [
                "\n---\n",
                "## 2. Component Analysis\n",
                "| Model | Component | Precision | Recall | F1-Score |",
                "| :--- | :--- | :---: | :---: | :---: |",
            ]
        )
        for m_name, m_metrics in model_results.items():
            md_lines.append(
                f"| **{m_name}** | Entity Extraction | {m_metrics['entities']['precision']:.4f} | {m_metrics['entities']['recall']:.4f} | **{m_metrics['entities']['f1']:.4f}** |"
            )
            md_lines.append(
                f"| **{m_name}** | Predicate Classification | {m_metrics['predicates']['precision']:.4f} | {m_metrics['predicates']['recall']:.4f} | **{m_metrics['predicates']['f1']:.4f}** |"
            )

    full_report = "\n".join(md_lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(full_report)
    print(f"\n[SUCCESS] Updated report generated at: {REPORT_PATH}")


if __name__ == "__main__":
    main()