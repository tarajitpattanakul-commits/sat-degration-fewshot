# Compounding Effects of Label Scarcity and Raw-Sensor Degradation on Lightweight Satellite Image Classifiers

This repository contains the code and results for a study measuring how label
scarcity (few-shot training) and simulated raw-sensor degradation interact when
training lightweight image classifiers for onboard satellite deployment.

## Summary

We evaluate three architecturally distinct lightweight classifiers — a
convolutional network trained from scratch, a pretrained MobileNetV2, and a
pretrained ViT-Tiny — on the EuroSAT benchmark, across a full grid of:

- **Label availability:** 5-shot, 10-shot, 50-shot, full-data
- **Simulated degradation:** none, moderate, severe
- **10 random seeds** per condition (360 total training runs)

**Headline finding:** raw-sensor degradation's accuracy penalty grows with
label availability rather than compounding under scarcity. Under conservative
(Bonferroni-corrected) significance testing, no architecture shows a reliable
degradation effect under severe label scarcity (5-shot, 10-shot), while a
significant penalty emerges for at least one architecture at every high-label
condition tested (50-shot, full-data), with the remaining architectures
trending in the same direction.

## Repository contents

| File | Description |
|---|---|
| `train.py` | Full experiment pipeline: dataset loading, degradation simulation, few-shot sampling, model definitions, training/eval loop, and the 360-run grid |
| `results_raw.csv` | Every individual run (360 rows): model, shot count, degradation level, seed, accuracy, F1-score, train time, parameter count |
| `results_summary.csv` | Mean ± std per (model, shots, degradation) combination (36 rows) |
| `figure1_accuracy_by_condition.png` / `.pdf` | Three-panel accuracy comparison figure used in the paper |
| `requirements.txt` | Exact package versions used |

## Reproducing the results

```bash
pip install -r requirements.txt
python train.py
```

Note: the full grid (360 runs) took approximately [FILL IN: total wall-clock
time] on a mix of NVIDIA T4 GPU and CPU compute via Google Colab. The script
supports resuming from `results_raw.csv` if interrupted — it will skip any
(model, shots, degradation, seed) combination already present in the file.

## Environment

- PyTorch 2.11.0 (CUDA 12.8 build)
- timm 1.0.28
- scikit-learn 1.6.1
- Trained on Google Colab (mixed NVIDIA T4 GPU / CPU)

## Dataset

[EuroSAT](https://github.com/phelber/EuroSAT) (Helber et al., 2019), accessed
via `torchvision.datasets.EuroSAT`. Sentinel-2 satellite imagery, 10 land-use
and land-cover classes, 64×64 pixel patches.

## Citation

If you use this code or dataset splits, please cite:

```
[FILL IN once published — full paper citation]
```

## License

This code is released under the MIT License (see `LICENSE`).
