# Counterfactual Validity Rate (CVR) Report

## Per-Scenario Modification Tracking

| Scenario / Image | Target Edit Type | Original Value | Target Value | Graph Updated? | Status |
| :--- | :--- | :--- | :--- | :---: | :---: |
| `dashboard` | Person Label | `mary pappa` | `mia patel` | Yes | PASSED |
| `fb_profile` | Location Entity | `syria` | `spain` | Yes | PASSED |
| `viber_1` | Person Label | `michael anderson` | `matthew archer` | Yes | PASSED |
| `viber_2` | Person Label | `richard coleman` | `raymond caldwell` | Yes | PASSED |

## Global CVR Metric Summary

- **Total Image Modifications Tested:** 4
- **Successful Dynamic Graph Updates:** 4
- **Counterfactual Validity Rate (CVR):** **100.0%**
- **Target Benchmark:** > 97.0%
- **Overall Assessment:** PASSED

---

## Evaluation Notes

- **Definition:** Το CVR αξιολογεί αν οι οπτικές/κειμενικές παρεμβάσεις (counterfactual perturbations) στην εικόνα μεταφράζονται με συνέπεια σε αντίστοιχες ενημερώσεις στον παραγόμενο Γράφο Γνώσης.
- **Significance:** CVR 100% επιβεβαιώνει ότι το pipeline ανταποκρίνεται άμεσα και έγκυρα στις αλλαγές του περιεχομένου χωρίς να διατηρεί παλαιά 'απολιθώματα' πληροφορίας.