# Archive: Original v1 Study (Superseded)

These files represent the original 360-run study before the revisions
documented in the main README's "Revision history" section.

- `experiment_starter_v6_ORIGINAL.py` — the original experiment script.
  MobileNetV2 and ViT-Tiny were fine-tuned at learning rate 1e-3, later found
  to be a confound (see main README).
- `results_raw_ORIGINAL_v1.csv` — raw output of all 360 original runs.
- `results_summary_ORIGINAL_v1.csv` — mean/std summary of the same.

**These results are superseded and should not be cited as the paper's
findings.** They are kept here for transparency and reproducibility of the
revision history described in the main README. The current, correct results
are `results_final_merged.csv` (main grid) and `results_mismatch_v2.csv`
(mismatch/mitigation ablation) in the repository root.
