"""KITTI 2D object detection evaluation using the AP_40 protocol.

Implements the official KITTI evaluation metrics:
- 40-point recall interpolation (AP_40)
- Per-class IoU thresholds: Car=0.7, Pedestrian=0.5, Cyclist=0.5
- Three difficulty levels: Easy, Moderate, Hard
- DontCare region handling
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kitti_object_detection.config import (
    RAW_DATA_DIR,
    PREDICTIONS_DIR,
    PROCESSED_DATA_DIR,
    KITTI_IOU_THRESHOLDS,
    DIFFICULTY_LEVELS,
    CLASS_TO_ID,
)
from kitti_object_detection.data.kitti_labels import parse_kitti_label_line


@dataclass
class GroundTruthBox:
    class_name: str
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    truncation: float
    occlusion: int
    height: float  # bbox height in pixels


@dataclass
class PredictionBox:
    class_name: str
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float


def compute_iou(box1: tuple[float, ...], box2: tuple[float, ...]) -> float:
    """Compute IoU between two boxes in (x1, y1, x2, y2) format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def read_ground_truth(label_path: Path) -> list[GroundTruthBox]:
    """Read ground truth boxes from a KITTI label file."""
    gt_boxes = []
    if not label_path.exists():
        return gt_boxes

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = parse_kitti_label_line(line)
            height = obj.ymax - obj.ymin
            gt_boxes.append(GroundTruthBox(
                class_name=obj.class_name,
                bbox=(obj.xmin, obj.ymin, obj.xmax, obj.ymax),
                truncation=obj.truncation,
                occlusion=int(obj.occlusion),
                height=height,
            ))
    return gt_boxes


def read_predictions(pred_path: Path) -> list[PredictionBox]:
    """Read prediction boxes from a KITTI-format prediction file."""
    predictions = []
    if not pred_path.exists():
        return predictions

    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 16:
                continue
            class_name = parts[0]
            x1 = float(parts[4])
            y1 = float(parts[5])
            x2 = float(parts[6])
            y2 = float(parts[7])
            confidence = float(parts[15])
            predictions.append(PredictionBox(
                class_name=class_name,
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
            ))
    return predictions


def filter_by_difficulty(
    gt_boxes: list[GroundTruthBox],
    difficulty: str,
) -> list[bool]:
    """Return a mask indicating which GT boxes qualify for the given difficulty level."""
    criteria = DIFFICULTY_LEVELS[difficulty]
    mask = []
    for gt in gt_boxes:
        qualifies = (
            gt.height >= criteria["min_height"]
            and gt.occlusion <= criteria["max_occlusion"]
            and gt.truncation <= criteria["max_truncation"]
        )
        mask.append(qualifies)
    return mask


def interpolate_ap_40(precision: np.ndarray, recall: np.ndarray) -> float:
    """Compute Average Precision using 40-point recall interpolation (KITTI AP_40).

    Samples precision at 41 equally-spaced recall points [0, 1/40, 2/40, ..., 1]
    using maximum precision at recall >= r for each sample point.
    """
    if len(precision) == 0 or len(recall) == 0:
        return 0.0

    recall_thresholds = np.linspace(0, 1, 41)
    ap = 0.0

    for t in recall_thresholds:
        precisions_above = precision[recall >= t]
        if len(precisions_above) == 0:
            p = 0.0
        else:
            p = float(np.max(precisions_above))
        ap += p

    ap /= 41.0
    return ap


def evaluate_class_difficulty(
    all_gt: dict[str, list[GroundTruthBox]],
    all_preds: dict[str, list[PredictionBox]],
    class_name: str,
    difficulty: str,
    iou_threshold: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate AP for a single class at a single difficulty level.

    Returns (AP, precision_array, recall_array).
    """
    all_pred_entries = []
    total_gt = 0
    gt_matched_flags: dict[str, list[bool]] = {}
    # Per-sample lists of GT indices, split into valid vs ignored
    sample_valid_gt: dict[str, list[int]] = {}
    sample_ignored_gt: dict[str, list[int]] = {}

    for sample_id in all_gt:
        gt_boxes = all_gt[sample_id]
        difficulty_mask = filter_by_difficulty(gt_boxes, difficulty)

        valid_indices = []
        ignored_indices = []
        for i, gt in enumerate(gt_boxes):
            if gt.class_name == class_name:
                if difficulty_mask[i]:
                    valid_indices.append(i)
                else:
                    ignored_indices.append(i)

        total_gt += len(valid_indices)
        gt_matched_flags[sample_id] = [False] * len(gt_boxes)
        sample_valid_gt[sample_id] = valid_indices
        sample_ignored_gt[sample_id] = ignored_indices

        # Collect DontCare boxes for this sample
        dontcare_boxes = [
            gt.bbox for gt in gt_boxes if gt.class_name == "DontCare"
        ]

        # Collect predictions for this class
        preds = all_preds.get(sample_id, [])
        for pred in preds:
            if pred.class_name != class_name:
                continue

            # Check if prediction overlaps a DontCare region
            is_dontcare = False
            for dc_box in dontcare_boxes:
                if compute_iou(pred.bbox, dc_box) > 0.5:
                    is_dontcare = True
                    break

            all_pred_entries.append((
                pred.confidence,
                sample_id,
                pred.bbox,
                is_dontcare,
            ))

    if total_gt == 0:
        return 0.0, np.array([]), np.array([])

    # Sort all predictions by confidence (descending)
    all_pred_entries.sort(key=lambda x: x[0], reverse=True)

    tp = np.zeros(len(all_pred_entries))
    fp = np.zeros(len(all_pred_entries))
    ignored = np.zeros(len(all_pred_entries))

    for i, (conf, sample_id, pred_box, is_dontcare) in enumerate(all_pred_entries):
        gt_boxes = all_gt[sample_id]
        matched = gt_matched_flags[sample_id]

        # Search ALL same-class GTs (both valid and ignored) for best IoU match
        best_iou = 0.0
        best_gt_idx = -1
        all_class_indices = sample_valid_gt[sample_id] + sample_ignored_gt[sample_id]

        for gt_idx in all_class_indices:
            if matched[gt_idx]:
                continue
            iou = compute_iou(pred_box, gt_boxes[gt_idx].bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            if best_gt_idx in sample_valid_gt[sample_id]:
                # Matched a valid GT -> true positive
                tp[i] = 1
                matched[best_gt_idx] = True
            else:
                # Matched an ignored GT (same class, wrong difficulty) -> skip
                ignored[i] = 1
                matched[best_gt_idx] = True
        elif is_dontcare:
            # Overlaps DontCare region -> skip
            ignored[i] = 1
        else:
            fp[i] = 1

    # Remove ignored predictions before computing precision/recall
    keep = ignored == 0
    tp = tp[keep]
    fp = fp[keep]

    if len(tp) == 0:
        return 0.0, np.array([]), np.array([])

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recall = cum_tp / total_gt
    precision = cum_tp / (cum_tp + cum_fp)

    ap = interpolate_ap_40(precision, recall)
    return ap, precision, recall


def evaluate_kitti(
    pred_dir: Path,
    gt_dir: Path,
    val_ids: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Run full KITTI evaluation.

    Args:
        pred_dir: Directory with prediction files (sample_id.txt)
        gt_dir: Directory with original KITTI label files (sample_id.txt)
        val_ids: List of sample IDs to evaluate. If None, uses all prediction files.

    Returns:
        Nested dict: results[class_name][difficulty] = AP value
    """
    if val_ids is None:
        pred_files = sorted(pred_dir.glob("*.txt"))
        val_ids = [f.stem for f in pred_files]

    print(f"Evaluating {len(val_ids)} samples")

    # Load all ground truth and predictions
    all_gt: dict[str, list[GroundTruthBox]] = {}
    all_preds: dict[str, list[PredictionBox]] = {}

    for sample_id in val_ids:
        gt_path = gt_dir / f"{sample_id}.txt"
        pred_path = pred_dir / f"{sample_id}.txt"

        all_gt[sample_id] = read_ground_truth(gt_path)
        all_preds[sample_id] = read_predictions(pred_path)

    results: dict[str, dict[str, float]] = {}
    classes = list(CLASS_TO_ID.keys())

    for class_name in classes:
        iou_threshold = KITTI_IOU_THRESHOLDS[class_name]
        results[class_name] = {}

        for difficulty in DIFFICULTY_LEVELS:
            ap, _, _ = evaluate_class_difficulty(
                all_gt=all_gt,
                all_preds=all_preds,
                class_name=class_name,
                difficulty=difficulty,
                iou_threshold=iou_threshold,
            )
            results[class_name][difficulty] = ap

    return results


def print_results_table(results: dict[str, dict[str, float]]) -> None:
    """Print evaluation results in a formatted table."""
    difficulties = list(DIFFICULTY_LEVELS.keys())

    print("\n" + "=" * 60)
    print("KITTI 2D Object Detection Results (AP_40)")
    print("=" * 60)
    header = f"{'Class':<14}" + "".join(f"{d:>12}" for d in difficulties)
    print(header)
    print("-" * 60)

    for class_name in results:
        iou = KITTI_IOU_THRESHOLDS[class_name]
        row = f"{class_name} (IoU={iou})"
        row = f"{row:<14}"
        for difficulty in difficulties:
            ap = results[class_name][difficulty] * 100
            row += f"{ap:>11.2f}%"
        print(row)

    print("-" * 60)

    # Compute mAP across classes for each difficulty
    row = f"{'mAP':<14}"
    for difficulty in difficulties:
        aps = [results[c][difficulty] for c in results]
        mean_ap = np.mean(aps) * 100
        row += f"{mean_ap:>11.2f}%"
    print(row)
    print("=" * 60)


def get_val_sample_ids(data_dir: Path) -> list[str]:
    """Get validation sample IDs from the processed dataset."""
    val_images_dir = data_dir / "images" / "val"
    if not val_images_dir.exists():
        raise FileNotFoundError(f"Validation images not found at {val_images_dir}")
    return sorted(p.stem for p in val_images_dir.glob("*.png"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KITTI 2D object detection")
    parser.add_argument("--pred-dir", type=Path, default=None, help="Path to prediction files")
    parser.add_argument("--gt-dir", type=Path, default=None, help="Path to original KITTI label files")
    parser.add_argument("--data-dir", type=Path, default=None, help="Path to processed KITTI data")
    return parser.parse_args()


def main():
    args = parse_args()

    pred_dir = args.pred_dir or PREDICTIONS_DIR
    gt_dir = args.gt_dir or (RAW_DATA_DIR / "label_2")
    data_dir = args.data_dir or PROCESSED_DATA_DIR

    if not pred_dir.exists():
        raise FileNotFoundError(
            f"Predictions not found at {pred_dir}. "
            "Run 'python -m kitti_object_detection.predict' first."
        )
    if not gt_dir.exists():
        raise FileNotFoundError(
            f"Ground truth labels not found at {gt_dir}. "
            "Ensure KITTI raw data is at data/kitti/raw/label_2/"
        )

    val_ids = get_val_sample_ids(data_dir)
    results = evaluate_kitti(pred_dir=pred_dir, gt_dir=gt_dir, val_ids=val_ids)
    print_results_table(results)


if __name__ == "__main__":
    main()
