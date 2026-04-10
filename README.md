# KITTI 2D Object Detection with YOLO11

Fine-tuning **YOLO11m** for 2D object detection on the [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d). Detects **Cars**, **Pedestrians**, and **Cyclists** in autonomous driving scenes using monocular camera images.

## Table of Contents

- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Setup](#setup)
- [Pipeline](#pipeline)
- [Training Configuration](#training-configuration)
- [Evaluation Protocol](#evaluation-protocol)
- [Results](#results)
- [Training on CSC Mahti](#training-on-csc-mahti)
- [Project Structure](#project-structure)

---

## Introduction

This project implements a complete pipeline for 2D object detection on the KITTI benchmark:

1. **Dataset preparation** — converts KITTI annotations to YOLO format with an 80/20 train/val split
2. **Training** — fine-tunes YOLO11m from COCO-pretrained weights with driving-scene-optimized augmentation
3. **Inference** — generates predictions on the validation set in KITTI format
4. **Evaluation** — computes KITTI AP_40 metrics at three difficulty levels
5. **Visualization** — overlays ground truth and predicted bounding boxes for qualitative analysis

The KITTI 2D object detection benchmark contains **7,481 training images** with approximately **80,256 labeled objects**. Evaluation follows the KITTI protocol using Average Precision with the **40-point recall interpolation method (AP_40)**.

---

## Model Architecture

**YOLO11m** (medium variant) — the latest model in the Ultralytics YOLO family.

| Component | Description |
|-----------|-------------|
| Backbone | CSPDarknet with C3k2 blocks |
| Neck | Path Aggregation FPN (PAFPN) for multi-scale feature fusion |
| Head | Decoupled detection head with separate classification and regression branches |
| Design | Anchor-free — predicts bounding box centers directly |
| Parameters | ~20M |
| Input size | 640×640 |

The medium variant balances accuracy and overfitting risk on KITTI's relatively small dataset (~7.5K images). Training starts from **COCO-pretrained weights** (`yolo11m.pt`); COCO's 80 classes include direct analogues to KITTI's three target classes (car, person, bicycle).

---

## Dataset

### Source

[KITTI Object Detection Benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) — left color images (1242×375 px) with annotations in a 15-column format.

### Classes

| ID | Class | IoU Threshold |
|----|-------|--------------|
| 0 | Car | 0.7 |
| 1 | Pedestrian | 0.5 |
| 2 | Cyclist | 0.5 |

Objects labeled `DontCare`, `Van`, `Truck`, `Person_sitting`, `Tram`, and `Misc` are excluded from training. `DontCare` regions are still used during evaluation to avoid penalizing detections in ignored areas.

### Format Conversion

KITTI annotations are converted to YOLO format (normalized center coordinates):

```
class_id  x_center  y_center  width  height
```

All coordinates are normalized to [0, 1] relative to image dimensions.

### Train / Val Split

An 80/20 random split with fixed seed 42:
- **Train**: ~5,985 images
- **Val**: ~1,496 images

---

## Setup

### Requirements

- Python ≥ 3.12
- PyTorch 2.2.2
- Ultralytics ≥ 8.4.30

### 1. Clone the repository

```bash
git clone https://github.com/Valtterimj/Autonomous-driving-project.git
cd Autonomous-driving-project
```

### 2. Set up the environment

**Using uv (recommended):**
```bash
uv sync
uv pip install -e .
```

**Using pip:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Download KITTI data

Download from [KITTI 2D object detection](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d):
- Left color images of object data set (12 GB)
- Training labels of object data set

Place files in:
```
data/kitti/raw/
├── image_2/     # PNG images  (000000.png … 007480.png)
└── label_2/     # Label files (000000.txt … 007480.txt)
```

---

## Pipeline

Run the five steps in order:

### Step 1 — Preprocess dataset

Converts KITTI annotations to YOLO format, performs the train/val split, and generates the dataset YAML.

```bash
python -m kitti_object_detection.main
```

Output: `data/kitti/processed/{images,labels}/{train,val}/`

### Step 2 — Train YOLO11m

Fine-tunes YOLO11m from COCO-pretrained weights.

```bash
python -m kitti_object_detection.train
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `yolo11m.pt` | YOLO model name |
| `--epochs` | `150` | Number of training epochs |
| `--imgsz` | `640` | Input image size |
| `--batch` | `-1` | Batch size (`-1` = auto) |
| `--device` | `auto` | `0`, `cpu`, or `auto` |
| `--name` | `kitti_yolo11m` | Run name |
| `--resume` | — | Resume from last checkpoint |

Weights are saved to `runs/detect/kitti_yolo11m/weights/`.

### Step 3 — Run inference

Generates predictions on the validation set in KITTI format.

```bash
python -m kitti_object_detection.predict
```

| Option | Default | Description |
|--------|---------|-------------|
| `--weights` | `runs/detect/kitti_yolo11m/weights/best.pt` | Model weights |
| `--conf` | `0.001` | Confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |
| `--max-det` | `100` | Max detections per image |
| `--device` | `auto` | `0`, `cpu`, or `auto` |

Predictions are saved to `runs/predictions/` as per-image KITTI-format `.txt` files.

### Step 4 — Evaluate

Computes KITTI AP_40 metrics per class at Easy, Moderate, and Hard difficulty levels.

```bash
python -m kitti_object_detection.evaluate
```

| Option | Default | Description |
|--------|---------|-------------|
| `--pred-dir` | `runs/predictions/` | Prediction files |
| `--gt-dir` | `data/kitti/raw/label_2/` | Ground truth labels |
| `--data-dir` | `data/kitti/processed/` | Processed dataset root |

### Step 5 — Visualize results

Generates annotated images with ground truth (dashed) and predicted (solid) bounding boxes. Selects a diverse set of samples covering best detections, most false positives, and most missed objects.

```bash
python -m kitti_object_detection.visualize
```

| Option | Default | Description |
|--------|---------|-------------|
| `--num-samples` | `20` | Number of samples to visualize |
| `--conf` | `0.25` | Confidence threshold for display |

Output is saved to `runs/visualizations/{all,successes,failures}/`.

---

## Training Configuration

### Optimizer

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate (`lr0`) | 0.001 |
| LR schedule | Cosine annealing (`lrf=0.01`) |
| Warmup epochs | 3 |
| Weight decay | 0.0005 |
| Momentum | 0.937 |
| Epochs | 150 (early stopping patience=50) |
| Batch size | Auto (`-1`) |
| Mixed precision (AMP) | Enabled |

### Augmentation

Tailored for dashboard-camera driving scenes:

| Augmentation | Value | Rationale |
|---|---|---|
| Mosaic | 1.0 (off last 10 epochs) | Highest-impact multi-scale augmentation |
| Horizontal flip | 0.5 | Mirrored driving scenes are realistic |
| Vertical flip | 0.0 | Objects never appear upside-down |
| Scale | 0.5 | Simulates varying distances |
| Translation | 0.1 | Conservative spatial shift |
| Rotation | 0.0 | Roads are level |
| Mixup | 0.1 | Mild regularization |
| Copy-paste | 0.1 | Helps with rare classes (Cyclists) |
| HSV — hue | 0.015 | Slight color variation |
| HSV — saturation | 0.7 | Simulates varying lighting |
| HSV — value | 0.4 | Simulates varying brightness |

`close_mosaic=10` disables mosaic for the final 10 epochs, letting batch normalization statistics stabilize before convergence.

---

## Evaluation Protocol

### KITTI AP_40

The standard KITTI 2D object detection evaluation:

1. **40-point recall interpolation** — precision is sampled at 41 equally-spaced recall values [0, 1/40, …, 1]; maximum precision at recall ≥ r is used for each point, and the sum is divided by 40.
2. **Per-class IoU thresholds** — 0.7 for Car, 0.5 for Pedestrian and Cyclist.
3. **Greedy matching** — predictions sorted by confidence are matched to the highest-IoU unmatched ground truth box.
4. **DontCare handling** — detections overlapping DontCare regions (IoU > 0.5) are neither true positives nor false positives.

### Difficulty Levels

| Level | Min Height | Max Occlusion | Max Truncation |
|-------|-----------|---------------|----------------|
| Easy | 40 px | Fully visible (0) | 15% |
| **Moderate** (primary) | 25 px | Partly occluded (1) | 30% |
| Hard | 25 px | Largely occluded (2) | 50% |

---

## Results

Evaluated on 1,496 validation images (80/20 random split, seed=42).

### Detection Performance (AP_40, %)

| Class | Easy | Moderate | Hard |
|-------|------|----------|------|
| Car (IoU=0.7) | 97.32% | 97.37% | 92.35% |
| Pedestrian (IoU=0.5) | 96.62% | 90.81% | 85.59% |
| Cyclist (IoU=0.5) | 95.53% | 93.05% | 90.21% |
| **mAP** | **96.49%** | **93.74%** | **89.38%** |

### Qualitative Notes

**Strengths:**
- Nearby, unoccluded cars achieve near-perfect detection at Easy difficulty
- Real-time capable: YOLO11m processes images at 30–200+ FPS
- PAFPN neck handles objects at various distances effectively

**Challenges:**
- Small distant pedestrians/cyclists at Hard difficulty have lower recall
- The strict 0.7 IoU threshold for Cars makes precise localization critical
- Class imbalance (Cyclists are rare) is partially mitigated by copy-paste augmentation

---

## Training on CSC Mahti

Slurm job scripts for the [CSC Mahti](https://docs.csc.fi/computing/systems-mahti/) supercomputer (NVIDIA A100 40GB GPU) are provided in `scripts/`.

### Setup

1. Edit all scripts: replace `<project>` with your CSC project ID.
2. Ensure the dataset has been preprocessed locally before uploading to Mahti.

### Submit jobs

```bash
sbatch scripts/run_train.sh      # Fine-tune on A100 (up to 4h, NVMe fast I/O)
sbatch scripts/run_predict.sh    # Run inference on A100 (~30 min)
sbatch scripts/run_evaluate.sh   # Evaluate + visualize on CPU (~15 min)
```

Monitor with:
```bash
squeue -u $USER
```

The training script copies the processed dataset to local NVMe (`$LOCAL_SCRATCH`) before training to avoid shared filesystem bottlenecks.

---

## Project Structure

```
Autonomous-driving-project/
├── data/
│   └── kitti/
│       ├── raw/                     # Original KITTI data (download separately)
│       │   ├── image_2/
│       │   └── label_2/
│       └── processed/               # Generated by main.py
│           ├── images/{train,val}/
│           └── labels/{train,val}/
│
├── runs/                            # Generated by training/inference
│   ├── detect/kitti_yolo11m/        # Training outputs and weights
│   ├── predictions/                 # Per-image KITTI-format prediction files
│   └── visualizations/              # Annotated images
│
├── scripts/
│   ├── run_train.sh                 # Slurm: train on Mahti A100
│   ├── run_predict.sh               # Slurm: inference on Mahti
│   └── run_evaluate.sh              # Slurm: evaluate + visualize on Mahti
│
└── src/kitti_object_detection/
    ├── config.py                    # Paths, constants, dataset YAML generation
    ├── main.py                      # Data preprocessing pipeline
    ├── train.py                     # YOLO11 training
    ├── predict.py                   # Inference → KITTI-format predictions
    ├── evaluate.py                  # KITTI AP_40 evaluation
    ├── visualize.py                 # Detection visualization
    └── data/
        ├── kitti_reader.py          # KITTI sample discovery and loading
        ├── kitti_labels.py          # Label parsing and YOLO conversion
        ├── convert_kitti_to_yolo.py # Per-split format conversion
        ├── image_utils.py           # Image size utilities
        ├── splits.py                # Train/val splitting
        └── kitti_dataset.yaml       # Ultralytics dataset config (generated)
```
