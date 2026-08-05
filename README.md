# Superpixel Segmentation Benchmark: SLIC vs SSN (Deep Learning)

Comparison of two superpixel segmentation methods — the classic **SLIC** algorithm and the deep-learning-based **SSN (Superpixel Sampling Network)** — evaluated on the **BSDS300** dataset using the **ASA (Achievable Segmentation Accuracy)** metric.

## Overview

For a range of superpixel counts (`50, 100, 150, 200, 250, 300, 400, 500`), the project:

1. generates SLIC and SSN segmentations for every test image of BSDS300;
2. saves the segmented images to disk;
3. computes the ASA score of each segmentation against the human ground-truth segmentations;
4. reports the mean ASA score per method and per superpixel count.

The **ASA score** measures how well a superpixel segmentation can match the ground truth: for each superpixel, only the pixels of its majority ground-truth label are counted as correct, and the total is normalized by the number of pixels (value in `[0, 1]`, higher is better).

## Project structure

```
.
├── main.py                     # Entry point: runs the full pipeline
├── save_segmented_images.py    # Generates and saves SLIC/SSN segmentations
├── compute_ASA_scores.py       # Computes ASA scores vs. human segmentations
├── init.sh                     # Creates the output directories
└── segmentation_methods/       # SLIC and SSN segmentation implementations
    ├── slic_segmentation.py
    └── deepSSn_segmentation.py
```

> The `BSDS300` dataset and a `BSDS` helper module (for reading ground-truth segmentations) are expected at the project root.

## Requirements

- Python 3
- `numpy`, `matplotlib`, `scikit-image`

```bash
pip install numpy matplotlib scikit-image
```

Segmented images are written to `segmented_images/slic/` and `segmented_images/deep_snn/`, and mean ASA scores are printed for each superpixel count.

## Context

Project developed as part of research exposure on superpixel image segmentation (LaBRI), comparing a traditional clustering approach (SLIC) against a learned approach (SSN).
