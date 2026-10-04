# Morph-Paste

Official implementation of **Morph-Paste: Morphology-Adaptive Poisson Blending for Bounding-Box-Annotated Industrial Surface Defect Detection**.

## Overview

Morph-Paste is an offline data augmentation method for industrial surface defect detection using bounding-box annotations.

The method includes:

1. Class-conditional instance scaling.
2. Low-gradient placement-region selection.
3. Local histogram matching.
4. Otsu thresholding and morphological dilation.
5. Elliptical-mask fallback for unreliable foreground masks.
6. Poisson image blending using OpenCV `NORMAL_CLONE`.
7. Automatic updating of YOLO-format bounding-box annotations.

The augmentation procedure does not require pixel-level segmentation masks and does not modify the detector architecture.

## Repository Structure

```text
Morph-Paste/
├── morph_paste.py                 # Main Morph-Paste implementation
├── train_multiseed.py             # YOLOv8 training and evaluation
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── ablation_c_histogram.py        # Histogram-matching ablation
├── ablation_d_adaptive_alpha.py   # Adaptive-mask and alpha-blending ablation
├── stable mask possion.py         # Fixed elliptical-mask Poisson comparison
├── alpha_blending.py              # Gaussian alpha-blending comparison
├── mixup_2_0x.py                  # MixUp comparison
└── Copy paste improved.py         # Scale and placement constrained copy-paste
```

The ablation and comparison scripts are optional and are included to support the experiments reported in the manuscript.

## Installation

Python 3.10 or later is recommended.

Install the required packages using:

```bash
pip install -r requirements.txt
```

The principal dependencies are PyTorch, Ultralytics, OpenCV, NumPy, scikit-image, and PyYAML.

## Datasets

### NEU-DET

The NEU surface defect dataset is available from:

https://faculty.neu.edu.cn/songkc/en/zdylm/263270/list/index.htm

An alternative repository is available at:

https://github.com/abin24/NEU-DET

### GC10-DET

The GC10-DET metallic surface defect dataset is available at:

https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets

The original datasets are not included in this repository. Users should download them from their respective official sources and convert the annotations to YOLO format when necessary.

## Dataset Organization

The expected dataset structure is:

```text
NEU-DET/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Each image must have a corresponding YOLO-format label file:

```text
class_id center_x center_y width height
```

The four bounding-box coordinates must be normalized to the range from 0 to 1.

## Running Morph-Paste

Before running the augmentation script, open `morph_paste.py` and set the dataset path:

```python
DATA_ROOT = r"path/to/NEU-DET"
```

The default augmentation settings are:

```python
RATIO = 2.0
SEED = 42
OUT_NAME = "train_cp_poisson_v2_2_0x"
```

Run the offline augmentation using:

```bash
python morph_paste.py
```

With `RATIO = 2.0`, the script retains the original training images and generates an equal number of augmented images.

The generated images and updated YOLO annotations are saved under:

```text
NEU-DET/train_cp_poisson_v2_2_0x/
├── images/
└── labels/
```

## Training and Evaluation

Open `train_multiseed.py` and configure the following paths:

```python
ROOT = Path(r"path/to/project")
DATA = ROOT / "data_ours_poisson.yaml"
MODEL = ROOT / "yolov8n.pt"
```

To reproduce the three-seed experiment, use:

```python
SEEDS = (0, 42, 3407)
```

Then run:

```bash
python train_multiseed.py
```

The manuscript uses the following principal detector settings:

```text
Model: YOLOv8n
Image size: 640 × 640
Batch size: 4
Maximum epochs: 300
Early-stopping patience: 50
Initial learning rate: 0.01
Momentum: 0.937
Weight decay: 0.0005
Mosaic: 1.0
MixUp: 0.0
Built-in copy-paste: 0.0
```

Validation and test images must remain unchanged. Offline augmentation must be applied only to the training set.

## Main Results

On NEU-DET, three-seed mean mAP@0.5 increased from `76.4 ± 0.9%` to `77.6 ± 0.6%`, while recall increased from `71.2 ± 2.9%` to `73.9 ± 0.3%`.

On GC10-DET, mAP@0.5 increased from `65.3%` to `66.7%`, and precision increased from `59.6%` to `65.9%`.

## Notes

- Dataset images and model weights are not included in this repository.
- The augmentation code expects YOLO-format bounding-box annotations.
- Random seeds should be fixed for reproducible data generation and detector training.
- Paths in the scripts must be updated according to the local directory structure.
- The code is provided to support verification and reproduction of the manuscript results.
