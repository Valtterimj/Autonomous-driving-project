import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from kitti_object_detection.config import (
    PROCESSED_DATA_DIR,
    DATASET_YAML_PATH,
    DETECT_DIR,
    DEFAULT_MODEL,
    DEFAULT_TRAIN_NAME,
    generate_dataset_yaml,
)


def train(
    model_name: str = DEFAULT_MODEL,
    data_dir: Path | None = None,
    epochs: int = 150,
    imgsz: int = 640,
    batch: int = -1,
    device: str | int = "auto",
    run_name: str = DEFAULT_TRAIN_NAME,
    resume: bool = False,
) -> Path:
    """Fine-tune YOLO11 on the KITTI dataset.

    Returns the path to the best model weights.
    """
    if data_dir is None:
        data_dir = PROCESSED_DATA_DIR

    train_images = data_dir / "images" / "train"
    if not train_images.exists():
        raise FileNotFoundError(
            f"Training images not found at {train_images}. "
            "Run 'python -m kitti_object_detection.main' first to preprocess the dataset."
        )

    yaml_path = generate_dataset_yaml(data_dir=data_dir)
    print(f"Dataset YAML: {yaml_path}")

    if device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if resume:
        weights_path = DETECT_DIR / run_name / "weights" / "last.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Cannot resume: {weights_path} not found")
        model = YOLO(str(weights_path))
        print(f"Resuming training from: {weights_path}")
    else:
        model = YOLO(model_name)
        print(f"Loading COCO-pretrained model: {model_name}")

    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=50,
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,
        # Augmentation — driving-scene optimized
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.1,
        copy_paste=0.1,
        flipud=0.0,
        fliplr=0.5,
        scale=0.5,
        translate=0.1,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # Training control
        amp=True,
        project=str(DETECT_DIR),
        name=run_name,
        save=True,
        save_period=25,
        device=device,
        resume=resume,
        exist_ok=True,
    )

    best_weights = DETECT_DIR / run_name / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best_weights}")
    return best_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11 on KITTI")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="YOLO model name (default: yolo11m.pt)")
    parser.add_argument("--data-dir", type=Path, default=None, help="Path to processed KITTI data")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1 for auto)")
    parser.add_argument("--device", type=str, default="auto", help="Device: 0, 1, cpu, or auto")
    parser.add_argument("--name", type=str, default=DEFAULT_TRAIN_NAME, help="Run name")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device
    if device not in ("auto", "cpu"):
        try:
            device = int(device)
        except ValueError:
            pass

    train(
        model_name=args.model,
        data_dir=args.data_dir,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        run_name=args.name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
