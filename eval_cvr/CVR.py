"""
Counterfactual Validity Rate (CVR) Evaluator
Targeted verification of modified entities between original and counterfactual predictions.
Outputs results directly to cvr_report.md.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_RESULTS_PATH = BASE_DIR / "results.json"
CF_RESULTS_PATH = BASE_DIR / "results_cf.json"
REPORT_PATH = BASE_DIR / "cvr_report.md"

# Στοχευμένες αλλαγές ανά σενάριο
EXPECTED_CHANGES = {
    "dashboard": {
        "type": "Person Label",
        "from": "mary pappa",
        "to": "mia patel",
    },
    "fb_profile": {
        "type": "Location Entity",
        "from": "syria",
        "to": "spain",
    },
    "viber_1": {
        "type": "Person Label",
        "from": "michael anderson",
        "to": "matthew archer",
    },
    "viber_2": {
        "type": "Person Label",
        "from": "richard coleman",
        "to": "raymond caldwell",
    },
}


def get_target_entity_values(data):
    """Extracts entity labels and DBpedia resource names (lowercased)."""
    extracted_values = []

    # Extraction from entities
    for ent in data.get("entities", []):
        obj = str(ent.get("object", "")).strip().lower()
        if "http://dbpedia.org/resource/" in obj:
            extracted_values.append(obj.split("/")[-1])
        else:
            extracted_values.append(obj)

    # Extraction from literal relations (excluding full URL strings)
    for rel in data.get("relations", []):
        obj = str(rel.get("object", "")).strip().lower()
        if not obj.startswith("http"):
            extracted_values.append(obj)

    return extracted_values


def main():
    # Fallback search if results.json is in parent dir or named differently
    orig_path = (
        ORIGINAL_RESULTS_PATH
        if ORIGINAL_RESULTS_PATH.exists()
        else BASE_DIR.parent / "results.json"
    )
    cf_path = (
        CF_RESULTS_PATH
        if CF_RESULTS_PATH.exists()
        else BASE_DIR.parent / "results_cf.json"
    )

    if not orig_path.exists() or not cf_path.exists():
        print("[ERROR] Missing results.json or results_cf.json!")
        return

    with open(orig_path, "r", encoding="utf-8") as f:
        orig_data = json.load(f)
    with open(cf_path, "r", encoding="utf-8") as f:
        cf_data = json.load(f)

    total_edits = len(EXPECTED_CHANGES)
    successful_updates = 0

    md_lines = [
        "# Counterfactual Validity Rate (CVR) Report\n",
        "## Per-Scenario Modification Tracking\n",
        "| Scenario / Image | Target Edit Type | Original Value | Target Value | Graph Updated? | Status |",
        "| :--- | :--- | :--- | :--- | :---: | :---: |",
    ]

    for scenario, edit_info in EXPECTED_CHANGES.items():
        old_val = edit_info["from"]
        new_val = edit_info["to"]

        cf_scenario_vals = get_target_entity_values(cf_data.get(scenario, {}))

        # Verification: New value is present AND old value is removed from entities
        has_new = new_val in cf_scenario_vals
        has_old = old_val in cf_scenario_vals

        updated = has_new and not has_old

        if updated:
            successful_updates += 1
            status = "PASSED"
        else:
            status = "FAILED"

        md_lines.append(
            f"| `{scenario}` | {edit_info['type']} | `{old_val}` | `{new_val}` | {'Yes' if updated else 'No'} | {status} |"
        )

    cvr_score = (successful_updates / total_edits) * 100.0

    md_lines.extend(
        [
            "\n## Global CVR Metric Summary\n",
            f"- **Total Image Modifications Tested:** {total_edits}",
            f"- **Successful Dynamic Graph Updates:** {successful_updates}",
            f"- **Counterfactual Validity Rate (CVR):** **{cvr_score:.1f}%**",
            "- **Target Benchmark:** > 97.0%",
            f"- **Overall Assessment:** {'PASSED' if cvr_score >= 97.0 else 'NEEDS REVIEW'}\n",
            "---\n",
            "## Evaluation Notes\n",
            "- **Definition:** Το CVR αξιολογεί αν οι οπτικές/κειμενικές παρεμβάσεις (counterfactual perturbations) στην εικόνα μεταφράζονται με συνέπεια σε αντίστοιχες ενημερώσεις στον παραγόμενο Γράφο Γνώσης.",
            "- **Significance:** CVR 100% επιβεβαιώνει ότι το pipeline ανταποκρίνεται άμεσα και έγκυρα στις αλλαγές του περιεχομένου χωρίς να διατηρεί παλαιά 'απολιθώματα' πληροφορίας.",
        ]
    )

    full_report = "\n".join(md_lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(full_report)
    print(
        f"\n[SUCCESS] Updated CVR Report ({cvr_score:.1f}%) saved to: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()