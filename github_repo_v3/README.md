# Diagnosing and Fixing Train/Test Degradation Mismatch in Lightweight Satellite Image Classifiers

This repository contains the code and results for a study on how raw-sensor
degradation and label scarcity affect lightweight image classifiers trained
for onboard satellite deployment — and a validated, low-cost fix for the
largest risk identified.

**Note on framing:** clean-trained models failing on corrupted test data is a
well-documented general phenomenon (Hendrycks & Dietterich, ICLR 2019;
introduced ImageNet-C), and noise-augmented training as a mitigation is
similarly established (Ford et al., 2019). This work's contribution is not
that general phenomenon, but quantifying its severity and confirming the
mitigation remains effective under constraints atypical of that literature —
lightweight architectures and extreme label scarcity — in the specific,
previously unquantified context of onboard satellite AI.

## Headline findings

**1. Train/test degradation mismatch is severe in this setting.** A model
trained on clean imagery and evaluated on severely degraded imagery it never
saw during training loses **53–89% of its accuracy**, collapsing toward
random-chance levels, regardless of architecture or label count. A model
trained *and* evaluated under matched severe degradation loses only a few
percent. Significant for 5 of 6 tested conditions after Bonferroni
correction (n=10 seeds).

**2. A standard mitigation remains effective here.** Training with a
randomly assigned degradation level per image, per epoch ("mixed" training)
closes 65–99% of the mismatch-induced accuracy gap, bringing accuracy within
a few points of matched-condition performance across all three architectures
tested — despite the label scarcity and lightweight-architecture constraints
atypical of the general robustness literature this technique comes from.
Significant for 5 of 6 conditions (p < 0.001).

**3. Nuance: the severity gradient is architecture-dependent.** MobileNetV2
shows an all-or-nothing cliff — partial train/test alignment barely helps.
SimpleCNN and ViT-Tiny degrade more gradually, benefiting meaningfully from
partial alignment even without an exact match.

**Secondary finding:** under matched conditions, the interaction between
label scarcity and degradation severity is architecture-dependent, not
universal — ViT-Tiny shows a significant "compounding" pattern, MobileNetV2
shows a significant but counterintuitive pattern apparently driven by noise
acting as implicit regularization under extreme scarcity, and SimpleCNN shows
no reliable interaction.

## Revision history (kept for transparency)

- **v1**: Original 360-run study claimed a universal "floor effect" —
  degradation's cost grows with label availability, holding across all three
  architectures. A review identified weak statistical framing, experimental
  design gaps, and overclaiming.
- **v2**: Fixed a learning-rate confound (MobileNetV2/ViT-Tiny were
  fine-tuned at 1e-3, too high for pretrained models). A new train/test
  decoupling ablation then revealed the mismatch catastrophe. However, this
  version's "5 of 6 significant" claim used **uncorrected** p-values;
  applying the same Bonferroni standard used elsewhere revealed only 3 of 6
  actually survived correction at n=5 seeds.
- **v3**: Reran the mismatch ablation at n=10 seeds (matching main grid
  rigor), recovering 5 of 6 conditions as significant with proper power.
  Added a severity-gradient test and a mitigation experiment (randomized
  training-time degradation), elevated to a co-headline result.
- **v4 (current)**: An external literature check found the core mismatch
  phenomenon is well-established (Hendrycks & Dietterich 2019; Ford et al.
  2019) and that reference [8] misattributed EuroSAT-C's origin (the actual
  source is GenFormer, Oehri et al. 2024, which also already explores
  training-time augmentation on EuroSAT-C). Added all three as citations and
  honestly reframed the paper's contribution from implied novel discovery to
  accurately scoped: severity quantification and mitigation validation under
  onboard-AI-specific constraints.

See `archive/` for the original v1 code, results, and writeup.

## Repository contents

| File | Description |
|---|---|
| `final_rerun.py` | Main grid: MobileNetV2 + ViT-Tiny at corrected lr=1e-4, all shot counts/degradation levels, 10 seeds |
| `ablation_v1.py` | Original Track A (LR-tuned pilot) + Track B (5-seed decoupling ablation, superseded by v2) |
| `mismatch_v2.py` | **Current mismatch/mitigation ablation**: 10 seeds, severity gradient (none/moderate/severe train × test), and "mixed" (randomized-degradation) mitigation training |
| `results_final_merged.csv` | Clean merged main-grid dataset: SimpleCNN (lr=1e-3) + MobileNetV2/ViT-Tiny (lr=1e-4) — used for Table I |
| `results_mismatch_v2.csv` | **Primary evidentiary basis for the paper's headline finding** — mismatch + severity gradient + mitigation, 10 seeds, 720 data points |
| `results_track_b_decoupling.csv` | Original 5-seed decoupling ablation (superseded by `results_mismatch_v2.csv`, kept for transparency) |
| `results_mobilenet_vit_final.csv` | Raw output of the full corrected MobileNetV2/ViT-Tiny main-grid rerun |
| `figure1_final.png` / `.pdf` | Primary figure: clean / matched-severe / mismatch / mixed-trained accuracy comparison |
| `figure1_accuracy_by_condition.png` / `.pdf` | Secondary figure: scarcity×degradation interaction by architecture |
| `paper_v4.docx` | Current paper draft (IEEE GRSL format, 5 pages) |
| `requirements.txt` | Exact package versions used |
| `archive/` | Original (v1, superseded) 360-run grid, analysis, and writeup |

## Reproducing the results

```bash
pip install -r requirements.txt
python final_rerun.py       # Main grid: MobileNetV2 + ViT-Tiny, corrected LR
python mismatch_v2.py       # Mismatch + severity gradient + mitigation ablation (720 datapoints)
```

Scripts support resuming from their respective CSVs if interrupted — they
skip any (model, shots, degradation, seed) combination already present.

## Environment

- PyTorch 2.11.0 (CUDA 12.8 build)
- timm 1.0.28
- scikit-learn 1.6.1
- scipy (Mann-Whitney U, bootstrap contrasts)
- Trained on Google Colab (mixed NVIDIA T4 GPU / CPU)

## Dataset

[EuroSAT](https://github.com/phelber/EuroSAT) (Helber et al., 2019), accessed
via `torchvision.datasets.EuroSAT`. Sentinel-2 satellite imagery, 10 land-use
and land-cover classes, 64×64 pixel patches.

## Related work this study builds on

- Hendrycks, D. and Dietterich, T. "Benchmarking Neural Network Robustness to
  Common Corruptions and Perturbations." ICLR, 2019.
- Ford, N., Gilmer, J., Carlini, N., and Cubuk, E. D. "Adversarial Examples
  Are a Natural Consequence of Test Error in Noise." arXiv:1901.10513, 2019.
- Oehri, S., Ebert, N., Abdullah, A., Stricker, D., and Wasenmüller, O.
  "GenFormer: Generated Images Are All You Need to Improve Robustness of
  Transformers on Small Datasets." ICPR, 2024.

## Citation

If you use this code or dataset splits, please cite:

```
[FILL IN once published — full paper citation]
```

## License

This code is released under the MIT License (see `LICENSE`).


