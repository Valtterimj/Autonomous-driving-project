import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from kitti_object_detection.config import (
    PROCESSED_DATA_DIR,
    DETECT_DIR,
    PREDICTIONS_DIR,
    DEFAULT_TRAIN_NAME,
    ID_TO_CLASS,
)


def predict(
    weights_path: Path | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    conf: float = 0.001,
    iou: float = 0.45,
    max_det: int = 100,
    device: str | int = "auto",
) -> Path:
    """Run inference on validation images and save predictions in KITTI format.

    Returns the output directory path.
    """
    if weights_path is None:
        weights_path = DETECT_DIR / DEFAULT_TRAIN_NAME / "weights" / "best.pt"
    if data_dir is None:
        data_dir = PROCESSED_DATA_DIR
    if output_dir is None:
        output_dir = PREDICTIONS_DIR

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    val_images_dir = data_dir / "images" / "val"
    if not val_images_dir.exists():
        raise FileNotFoundError(f"Validation images not found at {val_images_dir}")

    if device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    print(f"Loaded model: {weights_path}")
    print(f"Running inference on: {val_images_dir}")

    image_paths = sorted(val_images_dir.glob("*.png"))
    print(f"Found {len(image_paths)} validation images")

    for image_path in image_paths:
        results = model.predict(
            source=str(image_path),
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            verbose=False,
        )

        result = results[0]
        sample_id = image_path.stem
        pred_file = output_dir / f"{sample_id}.txt"

        lines = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, score, cls_id in zip(boxes, scores, class_ids):
                class_name = ID_TO_CLASS.get(cls_id, f"Unknown_{cls_id}")
                x1, y1, x2, y2 = box
                line = (
                    f"{class_name} -1 -1 -10 "
                    f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                    f"-1 -1 -1 -1000 -1000 -1000 -10 {score:.6f}"
                )
                lines.append(line)

        with pred_file.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

    print(f"Predictions saved to: {output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on KITTI validation set")
    parser.add_argument("--weights", type=Path, default=None, help="Path to model weights")
    parser.add_argument("--data-dir", type=Path, default=None, help="Path to processed KITTI data")
    parser.add_argument("--output-dir", type=Path, default=None, help="Path to save predictions")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=100, help="Max detections per image")
    parser.add_argument("--device", type=str, default="auto", help="Device: 0, cpu, or auto")
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device
    if device not in ("auto", "cpu"):
        try:
            device = int(device)
        except ValueError:
            pass

    predict(
        weights_path=args.weights,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=device,
    )


if __name__ == "__main__":
    main()
