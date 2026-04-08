from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "kitti" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "kitti" / "processed"
DATASET_YAML_PATH = PROJECT_ROOT / "src" / "kitti_object_detection" / "data" / "kitti_dataset.yaml"

RUNS_DIR = PROJECT_ROOT / "runs"
DETECT_DIR = RUNS_DIR / "detect"
VISUALIZATIONS_DIR = RUNS_DIR / "visualizations"
PREDICTIONS_DIR = RUNS_DIR / "predictions"

CLASS_NAMES = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
ID_TO_CLASS = CLASS_NAMES
CLASS_TO_ID = {v: k for k, v in CLASS_NAMES.items()}
NUM_CLASSES = len(CLASS_NAMES)

KITTI_IOU_THRESHOLDS = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}

DIFFICULTY_LEVELS = {
    "Easy": {"min_height": 40, "max_occlusion": 0, "max_truncation": 0.15},
    "Moderate": {"min_height": 25, "max_occlusion": 1, "max_truncation": 0.30},
    "Hard": {"min_height": 25, "max_occlusion": 2, "max_truncation": 0.50},
}

DEFAULT_MODEL = "yolo11m.pt"
DEFAULT_TRAIN_NAME = "kitti_yolo11m"


def generate_dataset_yaml(data_dir: Path | None = None, output_path: Path | None = None) -> Path:
    """Generate Ultralytics dataset YAML with absolute paths."""
    if data_dir is None:
        data_dir = PROCESSED_DATA_DIR
    if output_path is None:
        output_path = DATASET_YAML_PATH

    dataset_config = {
        "path": str(data_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": NUM_CLASSES,
        "names": CLASS_NAMES,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)

    return output_path
