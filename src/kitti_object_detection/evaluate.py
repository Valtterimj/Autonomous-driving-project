"""KITTI 2D object detection evaluation using the AP_40 protocol.

Implements KITTI-style object detection evaluation with:
- 40-point recall interpolation (AP_40)
- Per-class IoU thresholds: Car=0.7, Pedestrian=0.5, Cyclist=0.5
- Three difficulty levels: Easy, Moderate, Hard
- DontCare region handling
- Precision-Recall curve generation and plotting
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
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


@dataclass
class EvalResult:
    ap: float
    precision: np.ndarray
    recall: np.ndarray
    scores: np.ndarray
    tp: np.ndarray
    fp: np.ndarray
    num_gt: int
    num_predictions: int


def compute_iou(box1: tuple[float, ...], box2: tuple[float, ...]) -> float:
    """Compute IoU between two boxes in (x1, y1, x2, y2) format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0:
        return 0.0

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
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
            gt_boxes.append(
                GroundTruthBox(
                    class_name=obj.class_name,
                    bbox=(obj.xmin, obj.ymin, obj.xmax, obj.ymax),
                    truncation=obj.truncation,
                    occlusion=int(obj.occlusion),
                    height=height,
                )
            )
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

            predictions.append(
                PredictionBox(
                    class_name=class_name,
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                )
            )
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

    Samples precision at 41 equally spaced recall points:
    [0, 1/40, 2/40, ..., 1]
    using maximum precision at recall >= r for each sample point.
    """
    if len(precision) == 0 or len(recall) == 0:
        return 0.0

    recall_thresholds = np.linspace(0.0, 1.0, 41)
    ap = 0.0

    for t in recall_thresholds:
        precisions_above = precision[recall >= t]
        if len(precisions_above) == 0:
            p = 0.0
        else:
            p = float(np.max(precisions_above))
        ap += p

    return ap / 41.0


def compute_f1(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    """Compute F1 score pointwise from precision and recall."""
    denom = precision + recall
    f1 = np.zeros_like(denom)
    valid = denom > 0
    f1[valid] = 2.0 * precision[valid] * recall[valid] / denom[valid]
    return f1


def evaluate_class_difficulty(
    all_gt: dict[str, list[GroundTruthBox]],
    all_preds: dict[str, list[PredictionBox]],
    class_name: str,
    difficulty: str,
    iou_threshold: float,
) -> EvalResult:
    """Evaluate AP for a single class at a single difficulty level.

    Returns detailed precision-recall evaluation results.
    """
    all_pred_entries = []
    total_gt = 0

    gt_matched_flags: dict[str, list[bool]] = {}
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

        dontcare_boxes = [gt.bbox for gt in gt_boxes if gt.class_name == "DontCare"]

        preds = all_preds.get(sample_id, [])
        for pred in preds:
            if pred.class_name != class_name:
                continue

            is_dontcare = False
            for dc_box in dontcare_boxes:
                if compute_iou(pred.bbox, dc_box) > 0.5:
                    is_dontcare = True
                    break

            all_pred_entries.append(
                (
                    pred.confidence,
                    sample_id,
                    pred.bbox,
                    is_dontcare,
                )
            )

    if total_gt == 0:
        return EvalResult(
            ap=0.0,
            precision=np.array([], dtype=float),
            recall=np.array([], dtype=float),
            scores=np.array([], dtype=float),
            tp=np.array([], dtype=float),
            fp=np.array([], dtype=float),
            num_gt=0,
            num_predictions=0,
        )

    all_pred_entries.sort(key=lambda x: x[0], reverse=True)

    scores = np.array([entry[0] for entry in all_pred_entries], dtype=float)
    tp = np.zeros(len(all_pred_entries), dtype=float)
    fp = np.zeros(len(all_pred_entries), dtype=float)
    ignored = np.zeros(len(all_pred_entries), dtype=float)

    for i, (conf, sample_id, pred_box, is_dontcare) in enumerate(all_pred_entries):
        gt_boxes = all_gt[sample_id]
        matched = gt_matched_flags[sample_id]

        # Step 1: Find best overlapping VALID unmatched GT
        best_valid_iou = 0.0
        best_valid_idx = -1

        for gt_idx in sample_valid_gt[sample_id]:
            if matched[gt_idx]:
                continue
            iou = compute_iou(pred_box, gt_boxes[gt_idx].bbox)
            if iou > best_valid_iou:
                best_valid_iou = iou
                best_valid_idx = gt_idx

        if best_valid_iou >= iou_threshold and best_valid_idx >= 0:
            tp[i] = 1.0
            matched[best_valid_idx] = True
            continue

        # Step 2: If no valid match, check ignored GT of same class but wrong difficulty.
        # These ignored GTs are not consumed.
        best_ignored_iou = 0.0
        for gt_idx in sample_ignored_gt[sample_id]:
            iou = compute_iou(pred_box, gt_boxes[gt_idx].bbox)
            if iou > best_ignored_iou:
                best_ignored_iou = iou

        if best_ignored_iou >= iou_threshold:
            ignored[i] = 1.0
            continue

        # Step 3: Ignore predictions overlapping DontCare
        if is_dontcare:
            ignored[i] = 1.0
            continue

        # Step 4: Otherwise false positive
        fp[i] = 1.0

    keep = ignored == 0
    scores = scores[keep]
    tp = tp[keep]
    fp = fp[keep]

    if len(tp) == 0:
        return EvalResult(
            ap=0.0,
            precision=np.array([], dtype=float),
            recall=np.array([], dtype=float),
            scores=np.array([], dtype=float),
            tp=np.array([], dtype=float),
            fp=np.array([], dtype=float),
            num_gt=total_gt,
            num_predictions=0,
        )

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recall = cum_tp / total_gt
    precision = cum_tp / (cum_tp + cum_fp)
    ap = interpolate_ap_40(precision, recall)

    return EvalResult(
        ap=ap,
        precision=precision,
        recall=recall,
        scores=scores,
        tp=tp,
        fp=fp,
        num_gt=total_gt,
        num_predictions=len(scores),
    )


def load_dataset(
    pred_dir: Path,
    gt_dir: Path,
    val_ids: list[str],
) -> tuple[dict[str, list[GroundTruthBox]], dict[str, list[PredictionBox]]]:
    """Load all GT and predictions for the selected validation IDs."""
    all_gt: dict[str, list[GroundTruthBox]] = {}
    all_preds: dict[str, list[PredictionBox]] = {}

    for sample_id in val_ids:
        gt_path = gt_dir / f"{sample_id}.txt"
        pred_path = pred_dir / f"{sample_id}.txt"

        all_gt[sample_id] = read_ground_truth(gt_path)
        all_preds[sample_id] = read_predictions(pred_path)

    return all_gt, all_preds


def evaluate_kitti_detailed(
    pred_dir: Path,
    gt_dir: Path,
    val_ids: list[str] | None = None,
) -> dict[str, dict[str, EvalResult]]:
    """Run full KITTI evaluation and return detailed results."""
    if val_ids is None:
        pred_files = sorted(pred_dir.glob("*.txt"))
        val_ids = [f.stem for f in pred_files]

    print(f"Evaluating {len(val_ids)} samples")

    all_gt, all_preds = load_dataset(pred_dir=pred_dir, gt_dir=gt_dir, val_ids=val_ids)

    results: dict[str, dict[str, EvalResult]] = {}
    classes = list(CLASS_TO_ID.keys())

    for class_name in classes:
        iou_threshold = KITTI_IOU_THRESHOLDS[class_name]
        results[class_name] = {}

        for difficulty in DIFFICULTY_LEVELS:
            result = evaluate_class_difficulty(
                all_gt=all_gt,
                all_preds=all_preds,
                class_name=class_name,
                difficulty=difficulty,
                iou_threshold=iou_threshold,
            )
            results[class_name][difficulty] = result

    return results


def print_results_table(results: dict[str, dict[str, EvalResult]]) -> None:
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
            ap = results[class_name][difficulty].ap * 100.0
            row += f"{ap:>11.2f}%"

        print(row)

    print("-" * 60)

    row = f"{'mAP':<14}"
    for difficulty in difficulties:
        aps = [results[c][difficulty].ap for c in results]
        mean_ap = np.mean(aps) * 100.0
        row += f"{mean_ap:>11.2f}%"
    print(row)
    print("=" * 60)


def plot_pr_curve(
    result: EvalResult,
    class_name: str,
    difficulty: str,
    save_path: Path,
) -> None:
    """Plot a precision-recall curve for one class and difficulty."""
    plt.figure(figsize=(6, 5))

    if len(result.recall) > 0 and len(result.precision) > 0:
        plt.plot(
            result.recall,
            result.precision,
            linewidth=2,
            label=f"AP_40 = {result.ap:.4f}",
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve - {class_name} ({difficulty})")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_all_pr_curves(
    results: dict[str, dict[str, EvalResult]],
    output_dir: Path,
) -> None:
    """Plot PR curves for all classes and difficulty levels."""
    for class_name, class_results in results.items():
        for difficulty, result in class_results.items():
            save_path = output_dir / f"pr_{class_name.lower()}_{difficulty.lower()}.png"
            plot_pr_curve(
                result=result,
                class_name=class_name,
                difficulty=difficulty,
                save_path=save_path,
            )

def plot_pr_curves_moderate_combined(
    results: dict[str, dict[str, EvalResult]],
    save_path: Path,
) -> None:
    """Plot PR curves for all classes on Moderate difficulty in one figure."""
    plt.figure(figsize=(6, 5))

    difficulty = "Moderate"

    for class_name, class_results in results.items():
        if difficulty not in class_results:
            continue

        result = class_results[difficulty]
        if len(result.recall) == 0 or len(result.precision) == 0:
            continue

        plt.plot(
            result.recall,
            result.precision,
            linewidth=2,
            label=f"{class_name} (AP_40 = {result.ap:.4f})",
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (Moderate)")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def save_ap_table_csv(
    results: dict[str, dict[str, EvalResult]],
    output_path: Path,
) -> None:
    """Save AP table as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    difficulties = list(DIFFICULTY_LEVELS.keys())

    lines = []
    header = ["class"] + difficulties
    lines.append(",".join(header))

    for class_name in results:
        row = [class_name]
        for difficulty in difficulties:
            row.append(f"{results[class_name][difficulty].ap:.6f}")
        lines.append(",".join(row))

    map_row = ["mAP"]
    for difficulty in difficulties:
        aps = [results[c][difficulty].ap for c in results]
        map_row.append(f"{np.mean(aps):.6f}")
    lines.append(",".join(map_row))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_operating_points_csv(
    results: dict[str, dict[str, EvalResult]],
    output_path: Path,
) -> None:
    """Save best-F1 operating point summary as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    header = [
        "class",
        "difficulty",
        "ap_40",
        "num_gt",
        "num_predictions",
        "best_f1",
        "best_precision",
        "best_recall",
        "best_score_threshold",
    ]
    lines.append(",".join(header))

    for class_name, class_results in results.items():
        for difficulty, result in class_results.items():
            if len(result.precision) == 0:
                row = [
                    class_name,
                    difficulty,
                    f"{result.ap:.6f}",
                    str(result.num_gt),
                    str(result.num_predictions),
                    "0.000000",
                    "0.000000",
                    "0.000000",
                    "",
                ]
            else:
                f1 = compute_f1(result.precision, result.recall)
                best_idx = int(np.argmax(f1))
                row = [
                    class_name,
                    difficulty,
                    f"{result.ap:.6f}",
                    str(result.num_gt),
                    str(result.num_predictions),
                    f"{f1[best_idx]:.6f}",
                    f"{result.precision[best_idx]:.6f}",
                    f"{result.recall[best_idx]:.6f}",
                    f"{result.scores[best_idx]:.6f}",
                ]
            lines.append(",".join(row))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_val_sample_ids(data_dir: Path) -> list[str]:
    """Get validation sample IDs from the processed dataset."""
    val_images_dir = data_dir / "images" / "val"
    if not val_images_dir.exists():
        raise FileNotFoundError(f"Validation images not found at {val_images_dir}")
    return sorted(p.stem for p in val_images_dir.glob("*.png"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KITTI 2D object detection")
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=None,
        help="Path to prediction files",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=None,
        help="Path to original KITTI label files",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to processed KITTI data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save PR curves and CSV outputs",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving PR curve plots",
    )
    return parser.parse_args()

def plot_metric_vs_threshold(
    scores: np.ndarray,
    values: np.ndarray,
    metric_name: str,
    class_name: str,
    difficulty: str,
    save_path: Path,
) -> None:
    """Plot one metric as a function of confidence threshold."""
    plt.figure(figsize=(6, 5))

    if len(scores) > 0 and len(values) > 0:
        x = scores[::-1]
        y = values[::-1]
        plt.plot(x, y, linewidth=2)

    plt.xlabel("Confidence Threshold")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs Threshold - {class_name} ({difficulty})")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_threshold_curves(
    result: EvalResult,
    class_name: str,
    difficulty: str,
    save_path: Path,
) -> None:
    """Plot precision, recall, and F1 as functions of confidence threshold."""
    plt.figure(figsize=(7, 5))

    if len(result.scores) > 0:
        scores = result.scores[::-1]
        precision = result.precision[::-1]
        recall = result.recall[::-1]
        f1 = compute_f1(result.precision, result.recall)[::-1]

        plt.plot(scores, precision, linewidth=2, label="Precision")
        plt.plot(scores, recall, linewidth=2, label="Recall")
        plt.plot(scores, f1, linewidth=2, label="F1")

        best_idx = int(np.argmax(compute_f1(result.precision, result.recall)))
        best_score = result.scores[best_idx]
        best_f1 = compute_f1(result.precision, result.recall)[best_idx]

        plt.axvline(best_score, linestyle="--", alpha=0.7,
                    label=f"Best F1 threshold = {best_score:.3f}")

    plt.xlabel("Confidence Threshold")
    plt.ylabel("Score")
    plt.title(f"Threshold Analysis - {class_name} ({difficulty})")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_all_threshold_curves(
    results: dict[str, dict[str, EvalResult]],
    output_dir: Path,
) -> None:
    """Generate threshold analysis plots for all classes and difficulty levels."""
    for class_name, class_results in results.items():
        for difficulty, result in class_results.items():
            save_path = output_dir / f"threshold_{class_name.lower()}_{difficulty.lower()}.png"
            plot_threshold_curves(
                result=result,
                class_name=class_name,
                difficulty=difficulty,
                save_path=save_path,
            )

def main() -> None:
    args = parse_args()

    pred_dir = args.pred_dir or PREDICTIONS_DIR
    gt_dir = args.gt_dir or (RAW_DATA_DIR / "label_2")
    data_dir = args.data_dir or PROCESSED_DATA_DIR
    output_dir = args.output_dir or (pred_dir / "evaluation")

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
    results = evaluate_kitti_detailed(pred_dir=pred_dir, gt_dir=gt_dir, val_ids=val_ids)

    print_results_table(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_ap_table_csv(results, output_dir / "ap_results.csv")
    save_operating_points_csv(results, output_dir / "operating_points.csv")

    if not args.no_plots:
        # plot_all_pr_curves(results, output_dir / "pr_curves")
        # plot_all_threshold_curves(results, output_dir / "threshold_curves")
        plot_pr_curves_moderate_combined(
            results=results,
            save_path=output_dir / "pr_moderate_combined.png",
        )
    

    print(f"\nSaved evaluation outputs to: {output_dir}")


if __name__ == "__main__":
    main()