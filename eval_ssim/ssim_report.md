# UI Visual Structure Preservation Report (SSIM)

## Per-Scenario SSIM Scores

| Scenario / Image | Original Screenshot | Counterfactual Screenshot | SSIM Score | Status (Target > 0.90) |
| :--- | :--- | :--- | :---: | :---: |
| `dashboard` | `images/dashboard.png` | `counterfactual/dashboard_cf.png` | **0.9986** | PASSED |
| `fb_profile` | `images/fb_profile.png` | `counterfactual/fb_profile_cf.png` | **0.9996** | PASSED |
| `viber_1` | `images/viber_1.png` | `counterfactual/viber_1_cf.png` | **0.9918** | PASSED |
| `viber_2` | `images/viber_2.png` | `counterfactual/viber_2_cf.png` | **0.9867** | PASSED |

## Global Visual Fidelity Summary

- **Average Structural Similarity Index (SSIM):** **0.9942**
- **Target Benchmark:** > 0.9000
- **Overall Assessment:** PASSED

---

## Evaluation Notes

- **Purpose:** Μετράει τη διατήρηση του UI layout μετά το inpainting.
- **Target:** Τιμές > 0.90 επιβεβαιώνουν στοχευμένες τοπικές αλλαγές χωρίς παραμόρφωση του υπόλοιπου UI.