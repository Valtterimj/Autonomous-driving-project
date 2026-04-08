
# KITTI 2D Object Detection with YOLO11

Fine-tuning YOLO11m for 2D object detection on the KITTI benchmark. Detects Cars, Pedestrians, and Cyclists in driving scenes.

## Set up

### 1. Clone repo

```bash
git clone https://github.com/Valtterimj/Autonomous-driving-project.git
cd Autonomous-driving-project
```

### 2. Set up environment

#### Using uv
```bash
uv sync
uv pip install -e .
```

#### Using pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Download KITTI data

Download from https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d :
- Left color images of object data set (12 GB)
- Training labels of object data set

Place files in:
```
data/kitti/raw/
    image_2/     # PNG images (000000.png ... 007480.png)
    label_2/     # Label files (000000.txt ... 007480.txt)
```

## Pipeline

### Step 1: Preprocess dataset

Converts KITTI annotations to YOLO format, splits into train/val sets, and generates the dataset YAML.

```bash
python -m kitti_object_detection.main
```

Output: `data/kitti/processed/{images,labels}/{train,val}/`

### Step 2: Train YOLO11m

Fine-tunes YOLO11m with COCO-pretrained weights on the KITTI dataset.

```bash
python -m kitti_object_detection.train
```

Options:
```
--model      Model name (default: yolo11m.pt)
--epochs     Number of epochs (default: 150)
--imgsz      Image size (default: 640)
--batch      Batch size, -1 for auto (default: -1)
--device     Device: 0, cpu, or auto (default: auto)
--resume     Resume from last checkpoint
```

### Step 3: Run inference

Generates predictions on the validation set in KITTI format.

```bash
python -m kitti_object_detection.predict
```

### Step 4: Evaluate

Computes KITTI AP_40 metrics per class (Car, Pedestrian, Cyclist) at Easy, Moderate, and Hard difficulty levels.

```bash
python -m kitti_object_detection.evaluate
```

### Step 5: Visualize results

Generates images with overlaid bounding boxes showing detections, false positives, and misses.

```bash
python -m kitti_object_detection.visualize
```

## Training on CSC Mahti

Slurm job scripts are provided in `scripts/` for running on CSC Mahti supercomputer.

1. Edit scripts to set your project account: replace `<project>` with your CSC project ID
2. Submit jobs:

```bash
sbatch scripts/run_train.sh      # Train on A100 GPU
sbatch scripts/run_predict.sh    # Run inference
sbatch scripts/run_evaluate.sh   # Evaluate + visualize
```

The training script copies data to local NVMe (`$LOCAL_SCRATCH`) for fast I/O.

## Project Structure

```
src/kitti_object_detection/
├── main.py          # Data preprocessing pipeline
├── config.py        # Centralized paths, constants, YAML generation
├── train.py         # YOLO11 training
├── predict.py       # Inference on validation set
├── evaluate.py      # KITTI AP_40 evaluation
├── visualize.py     # Detection visualization
└── data/
    ├── kitti_reader.py          # KITTI data reading
    ├── kitti_labels.py          # Label parsing and conversion
    ├── convert_kitti_to_yolo.py # Format conversion
    ├── image_utils.py           # Image utilities
    ├── splits.py                # Train/val splitting
    └── kitti_dataset.yaml       # Ultralytics dataset config

scripts/
├── run_train.sh     # Slurm: training on Mahti
├── run_predict.sh   # Slurm: inference on Mahti
└── run_evaluate.sh  # Slurm: evaluation on Mahti

report.md            # Technical report
```

## Evaluation Protocol

- **Metric**: Average Precision with 40-point recall interpolation (AP_40)
- **IoU thresholds**: Car = 0.7, Pedestrian = 0.5, Cyclist = 0.5
- **Difficulty levels**: Easy, Moderate (primary), Hard
