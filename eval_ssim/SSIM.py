"""
UI Visual Structure Evaluation: SSIM Calculator
Compares images from 'images/' and 'counterfactual/' subdirectories.
Outputs metrics directly to ssim_report.md.
"""

from pathlib import Path
import cv2
from skimage.metrics import structural_similarity as ssim

BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_DIR = BASE_DIR / "images"
CF_DIR = BASE_DIR / "counterfactual"
REPORT_PATH = BASE_DIR / "ssim_report.md"

SCENARIOS = ["dashboard", "fb_profile", "viber_1", "viber_2"]


def calculate_ssim(img1_path, img2_path):
    img1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(img2_path), cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        return None

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    score, _ = ssim(img1, img2, full=True)
    return score


def main():
    scores = []
    md_lines = [
        "# UI Visual Structure Preservation Report (SSIM)\n",
        "## Per-Scenario SSIM Scores\n",
        "| Scenario / Image | Original Screenshot | Counterfactual Screenshot | SSIM Score | Status (Target > 0.90) |",
        "| :--- | :--- | :--- | :---: | :---: |",
    ]

    for scenario in SCENARIOS:
        orig_path = ORIGINAL_DIR / f"{scenario}.png"
        cf_path = CF_DIR / f"{scenario}_cf.png"

        score = calculate_ssim(orig_path, cf_path)

        if score is not None:
            scores.append(score)
            status = "PASSED" if score >= 0.90 else "FAILED"
            md_lines.append(
                f"| `{scenario}` | `images/{orig_path.name}` | `counterfactual/{cf_path.name}` | **{score:.4f}** | {status} |"
            )
            print(f"[{scenario}] SSIM: {score:.4f} ({status})")
        else:
            md_lines.append(
                f"| `{scenario}` | `images/{orig_path.name}` | `counterfactual/{cf_path.name}` | *N/A* | MISSING |"
            )
            print(f"[WARNING] Missing images for: {scenario}")

    avg_ssim = sum(scores) / len(scores) if scores else 0.0

    md_lines.extend(
        [
            "\n## Global Visual Fidelity Summary\n",
            f"- **Average Structural Similarity Index (SSIM):** **{avg_ssim:.4f}**",
            "- **Target Benchmark:** > 0.9000",
            f"- **Overall Assessment:** {'PASSED' if avg_ssim >= 0.90 else 'NEEDS REVIEW'}\n",
            "---\n",
            "## Evaluation Notes\n",
            "- **Purpose:** Μετράει τη διατήρηση του UI layout μετά το inpainting.",
            "- **Target:** Τιμές > 0.90 επιβεβαιώνουν στοχευμένες τοπικές αλλαγές χωρίς παραμόρφωση του υπόλοιπου UI.",
        ]
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n[SUCCESS] SSIM Report generated at: {REPORT_PATH}")


if __name__ == "__main__":
    main()