"""Visualization of KITTI object detection results.

Generates images with overlaid bounding boxes for both ground truth and predictions,
highlighting successful detections, false positives, and missed objects.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from kitti_object_detection.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PREDICTIONS_DIR,
    VISUALIZATIONS_DIR,
    KITTI_IOU_THRESHOLDS,
)
from kitti_object_detection.evaluate import (
    GroundTruthBox,
    PredictionBox,
    read_ground_truth,
    read_predictions,
    compute_iou,
    get_val_sample_ids,
)

# Colors: (R, G, B) for each class
CLASS_COLORS = {
    "Car": (0, 120, 255),
    "Pedestrian": (0, 200, 0),
    "Cyclist": (255, 50, 50),
}
GT_COLOR_ALPHA = 180
PRED_COLOR_ALPHA = 255
DEFAULT_COLOR = (200, 200, 200)


def draw_box(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[float, float, float, float],
    color: tuple[int, int, int],
    label: str,
    line_width: int = 2,
    dashed: bool = False,
) -> None:
    """Draw a bounding box with a label on an image."""
    x1, y1, x2, y2 = bbox

    if dashed:
        dash_len = 8
        gap_len = 5
        edges = [
            ((x1, y1), (x2, y1)),  # top
            ((x2, y1), (x2, y2)),  # right
            ((x2, y2), (x1, y2)),  # bottom
            ((x1, y2), (x1, y1)),  # left
        ]
        for (sx, sy), (ex, ey) in edges:
            length = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
            if length == 0:
                continue
            dx = (ex - sx) / length
            dy = (ey - sy) / length
            pos = 0.0
            while pos < length:
                seg_end = min(pos + dash_len, length)
                draw.line(
                    [(sx + dx * pos, sy + dy * pos), (sx + dx * seg_end, sy + dy * seg_end)],
                    fill=color,
                    width=line_width,
                )
                pos = seg_end + gap_len
    else:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    text_y = y1 - text_h - 4
    if text_y < 0:
        text_y = y1 + 2

    draw.rectangle(
        [x1, text_y, x1 + text_w + 4, text_y + text_h + 2],
        fill=color,
    )
    draw.text((x1 + 2, text_y), label, fill=(255, 255, 255), font=font)


def visualize_sample(
    image_path: Path,
    gt_boxes: list[GroundTruthBox],
    pred_boxes: list[PredictionBox],
    output_path: Path,
    conf_threshold: float = 0.25,
) -> None:
    """Create a visualization with GT and prediction bboxes overlaid."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    target_classes = set(CLASS_COLORS.keys())

    # Draw ground truth boxes (dashed lines)
    for gt in gt_boxes:
        if gt.class_name not in target_classes:
            continue
        color = CLASS_COLORS.get(gt.class_name, DEFAULT_COLOR)
        label = f"GT: {gt.class_name}"
        draw_box(draw, gt.bbox, color, label, line_width=2, dashed=True)

    # Draw prediction boxes (solid lines)
    for pred in pred_boxes:
        if pred.class_name not in target_classes:
            continue
        if pred.confidence < conf_threshold:
            continue
        color = CLASS_COLORS.get(pred.class_name, DEFAULT_COLOR)
        label = f"{pred.class_name} {pred.confidence:.2f}"
        draw_box(draw, pred.bbox, color, label, line_width=2, dashed=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")


def classify_detections(
    gt_boxes: list[GroundTruthBox],
    pred_boxes: list[PredictionBox],
    conf_threshold: float = 0.25,
) -> dict:
    """Classify detections into successes, false positives, and misses."""
    target_classes = set(CLASS_COLORS.keys())

    target_gt = [gt for gt in gt_boxes if gt.class_name in target_classes]
    target_preds = [p for p in pred_boxes if p.class_name in target_classes and p.confidence >= conf_threshold]

    gt_matched = [False] * len(target_gt)
    true_positives = 0
    false_positives = 0

    # Sort predictions by confidence
    sorted_preds = sorted(target_preds, key=lambda p: p.confidence, reverse=True)

    for pred in sorted_preds:
        iou_threshold = KITTI_IOU_THRESHOLDS.get(pred.class_name, 0.5)
        best_iou = 0.0
        best_idx = -1

        for i, gt in enumerate(target_gt):
            if gt_matched[i] or gt.class_name != pred.class_name:
                continue
            iou = compute_iou(pred.bbox, gt.bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= iou_threshold and best_idx >= 0:
            gt_matched[best_idx] = True
            true_positives += 1
        else:
            false_positives += 1

    missed = sum(1 for m in gt_matched if not m)

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "missed": missed,
        "total_gt": len(target_gt),
        "total_preds": len(target_preds),
    }


def visualize(
    data_dir: Path | None = None,
    gt_dir: Path | None = None,
    pred_dir: Path | None = None,
    output_dir: Path | None = None,
    num_samples: int = 20,
    conf_threshold: float = 0.25,
) -> None:
    """Generate visualization images for a selection of validation samples.

    Selects samples showing best detections, worst false positives, and most misses.
    """
    if data_dir is None:
        data_dir = PROCESSED_DATA_DIR
    if gt_dir is None:
        gt_dir = RAW_DATA_DIR / "label_2"
    if pred_dir is None:
        pred_dir = PREDICTIONS_DIR
    if output_dir is None:
        output_dir = VISUALIZATIONS_DIR

    val_ids = get_val_sample_ids(data_dir)
    val_images_dir = data_dir / "images" / "val"

    print(f"Analyzing {len(val_ids)} validation samples...")

    # Analyze all samples
    sample_stats = []
    for sample_id in val_ids:
        gt_path = gt_dir / f"{sample_id}.txt"
        pred_path = pred_dir / f"{sample_id}.txt"
        image_path = val_images_dir / f"{sample_id}.png"

        if not image_path.exists():
            continue

        gt_boxes = read_ground_truth(gt_path)
        pred_boxes = read_predictions(pred_path)
        stats = classify_detections(gt_boxes, pred_boxes, conf_threshold)
        stats["sample_id"] = sample_id
        sample_stats.append(stats)

    if not sample_stats:
        print("No valid samples found for visualization.")
        return

    # Select diverse samples
    selected_ids = set()

    # Best detections: highest TP rate with at least some GT objects
    with_gt = [s for s in sample_stats if s["total_gt"] > 0]
    by_tp_rate = sorted(
        with_gt,
        key=lambda s: s["true_positives"] / max(s["total_gt"], 1),
        reverse=True,
    )
    for s in by_tp_rate[:num_samples // 4]:
        selected_ids.add(s["sample_id"])

    # Worst false positives
    by_fp = sorted(sample_stats, key=lambda s: s["false_positives"], reverse=True)
    for s in by_fp[:num_samples // 4]:
        selected_ids.add(s["sample_id"])

    # Most missed objects
    by_missed = sorted(sample_stats, key=lambda s: s["missed"], reverse=True)
    for s in by_missed[:num_samples // 4]:
        selected_ids.add(s["sample_id"])

    # Fill remaining with random diverse samples
    remaining = [s for s in sample_stats if s["sample_id"] not in selected_ids]
    step = max(1, len(remaining) // (num_samples - len(selected_ids) + 1))
    for i in range(0, len(remaining), step):
        if len(selected_ids) >= num_samples:
            break
        selected_ids.add(remaining[i]["sample_id"])

    # Generate visualizations
    success_dir = output_dir / "successes"
    failure_dir = output_dir / "failures"
    all_dir = output_dir / "all"

    print(f"Generating {len(selected_ids)} visualizations...")

    for sample_id in sorted(selected_ids):
        gt_path = gt_dir / f"{sample_id}.txt"
        pred_path = pred_dir / f"{sample_id}.txt"
        image_path = val_images_dir / f"{sample_id}.png"

        gt_boxes = read_ground_truth(gt_path)
        pred_boxes = read_predictions(pred_path)
        stats = classify_detections(gt_boxes, pred_boxes, conf_threshold)

        # Save to all/
        visualize_sample(image_path, gt_boxes, pred_boxes, all_dir / f"{sample_id}.png", conf_threshold)

        # Categorize
        if stats["total_gt"] > 0 and stats["missed"] == 0 and stats["false_positives"] == 0:
            visualize_sample(image_path, gt_boxes, pred_boxes, success_dir / f"{sample_id}.png", conf_threshold)
        elif stats["false_positives"] > 0 or stats["missed"] > 0:
            visualize_sample(image_path, gt_boxes, pred_boxes, failure_dir / f"{sample_id}.png", conf_threshold)

    print(f"Visualizations saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize KITTI detection results")
    parser.add_argument("--data-dir", type=Path, default=None, help="Path to processed KITTI data")
    parser.add_argument("--gt-dir", type=Path, default=None, help="Path to original KITTI labels")
    parser.add_argument("--pred-dir", type=Path, default=None, help="Path to predictions")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of samples to visualize")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for display")
    return parser.parse_args()


def main():
    args = parse_args()
    visualize(
        data_dir=args.data_dir,
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        conf_threshold=args.conf,
    )


if __name__ == "__main__":
    main()
