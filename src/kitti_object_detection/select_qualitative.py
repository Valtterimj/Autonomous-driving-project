"""Select qualitative KITTI detection examples for report figures.

This script analyzes validation-set predictions and selects images that are
useful for qualitative discussion in the report, including:
- strong success cases
- mixed-class success cases
- missed detections
- false positives
- localization errors
- low-confidence correct detections

It reuses the existing visualization utilities to save annotated images.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

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
    compute_iou,
    get_val_sample_ids,
    read_ground_truth,
    read_predictions,
)
from kitti_object_detection.visualize import visualize_sample


TARGET_CLASSES = {"Car", "Pedestrian", "Cyclist"}


@dataclass
class MatchedPrediction:
    pred: PredictionBox
    matched_gt: GroundTruthBox | None
    iou: float
    outcome: str  # "tp", "fp", "ignored_class_mismatch", "localization_error"


@dataclass
class SampleAnalysis:
    sample_id: str
    num_gt: int
    num_preds: int
    tp: int
    fp: int
    fn: int
    localization_errors: int
    mixed_classes_present: int
    low_conf_tp_count: int
    mean_tp_conf: float
    score_success: float
    score_missed: float
    score_fp: float
    score_localization: float
    categories: list[str]


def filter_target_gt(gt_boxes: list[GroundTruthBox]) -> list[GroundTruthBox]:
    return [gt for gt in gt_boxes if gt.class_name in TARGET_CLASSES]


def filter_target_preds(
    pred_boxes: list[PredictionBox],
    conf_threshold: float,
) -> list[PredictionBox]:
    return [
        pred
        for pred in pred_boxes
        if pred.class_name in TARGET_CLASSES and pred.confidence >= conf_threshold
    ]


def match_predictions(
    gt_boxes: list[GroundTruthBox],
    pred_boxes: list[PredictionBox],
) -> tuple[list[MatchedPrediction], list[bool]]:
    """Match predictions to GT with class-specific IoU thresholds.

    Matching is confidence-ordered and one GT can be matched at most once.
    """
    gt_used = [False] * len(gt_boxes)
    matched_predictions: list[MatchedPrediction] = []

    sorted_preds = sorted(pred_boxes, key=lambda p: p.confidence, reverse=True)

    for pred in sorted_preds:
        best_same_class_iou = 0.0
        best_same_class_idx = -1

        best_any_class_iou = 0.0
        best_any_class_idx = -1

        for i, gt in enumerate(gt_boxes):
            if gt_used[i]:
                continue

            iou = compute_iou(pred.bbox, gt.bbox)

            if iou > best_any_class_iou:
                best_any_class_iou = iou
                best_any_class_idx = i

            if gt.class_name == pred.class_name and iou > best_same_class_iou:
                best_same_class_iou = iou
                best_same_class_idx = i

        iou_threshold = KITTI_IOU_THRESHOLDS.get(pred.class_name, 0.5)

        if best_same_class_idx >= 0 and best_same_class_iou >= iou_threshold:
            gt_used[best_same_class_idx] = True
            matched_predictions.append(
                MatchedPrediction(
                    pred=pred,
                    matched_gt=gt_boxes[best_same_class_idx],
                    iou=best_same_class_iou,
                    outcome="tp",
                )
            )
            continue

        # Distinguish localization error from generic FP:
        # if prediction overlaps same-class GT reasonably but not enough to pass IoU threshold
        unmatched_same_class_ious = [
            compute_iou(pred.bbox, gt.bbox)
            for i, gt in enumerate(gt_boxes)
            if not gt_used[i] and gt.class_name == pred.class_name
        ]
        best_unmatched_same_class_iou = max(unmatched_same_class_ious, default=0.0)

        if best_unmatched_same_class_iou >= 0.1:
            matched_predictions.append(
                MatchedPrediction(
                    pred=pred,
                    matched_gt=None,
                    iou=best_unmatched_same_class_iou,
                    outcome="localization_error",
                )
            )
        else:
            matched_predictions.append(
                MatchedPrediction(
                    pred=pred,
                    matched_gt=None,
                    iou=best_any_class_iou,
                    outcome="fp",
                )
            )

    return matched_predictions, gt_used


def analyze_sample(
    sample_id: str,
    gt_boxes: list[GroundTruthBox],
    pred_boxes: list[PredictionBox],
    low_conf_tp_max: float,
) -> SampleAnalysis:
    target_gt = filter_target_gt(gt_boxes)
    target_preds = pred_boxes

    matched_predictions, gt_used = match_predictions(target_gt, target_preds)

    tp_preds = [m for m in matched_predictions if m.outcome == "tp"]
    fp_preds = [m for m in matched_predictions if m.outcome == "fp"]
    loc_preds = [m for m in matched_predictions if m.outcome == "localization_error"]

    tp = len(tp_preds)
    fp = len(fp_preds)
    localization_errors = len(loc_preds)
    fn = sum(1 for used in gt_used if not used)

    gt_classes_present = {gt.class_name for gt in target_gt}
    mixed_classes_present = len(gt_classes_present)

    low_conf_tp_count = sum(1 for m in tp_preds if m.pred.confidence <= low_conf_tp_max)
    mean_tp_conf = (
        sum(m.pred.confidence for m in tp_preds) / len(tp_preds) if tp_preds else 0.0
    )

    categories: list[str] = []

    if tp > 0 and fp == 0 and fn == 0:
        categories.append("success_perfect")

    if tp > 0 and fp == 0 and fn == 0 and mixed_classes_present >= 2:
        categories.append("success_mixed_scene")

    if fn > 0:
        categories.append("failure_missed")

    if fp > 0:
        categories.append("failure_false_positive")

    if localization_errors > 0:
        categories.append("failure_localization")

    if low_conf_tp_count > 0:
        categories.append("interesting_low_conf_tp")

    # Ranking scores for candidate selection
    score_success = tp - 0.5 * fp - 0.5 * fn + 0.5 * mixed_classes_present + mean_tp_conf
    score_missed = fn + 0.25 * mixed_classes_present
    score_fp = fp + 0.25 * localization_errors
    score_localization = localization_errors + 0.1 * fn

    return SampleAnalysis(
        sample_id=sample_id,
        num_gt=len(target_gt),
        num_preds=len(target_preds),
        tp=tp,
        fp=fp,
        fn=fn,
        localization_errors=localization_errors,
        mixed_classes_present=mixed_classes_present,
        low_conf_tp_count=low_conf_tp_count,
        mean_tp_conf=mean_tp_conf,
        score_success=score_success,
        score_missed=score_missed,
        score_fp=score_fp,
        score_localization=score_localization,
        categories=categories,
    )


def select_top(
    analyses: list[SampleAnalysis],
    category: str,
    n: int,
) -> list[SampleAnalysis]:
    filtered = [a for a in analyses if category in a.categories]

    if category == "success_perfect":
        filtered.sort(
            key=lambda a: (
                a.score_success,
                a.mixed_classes_present,
                a.tp,
                -a.fp,
                -a.fn,
            ),
            reverse=True,
        )
    elif category == "success_mixed_scene":
        filtered.sort(
            key=lambda a: (
                a.mixed_classes_present,
                a.score_success,
                a.tp,
            ),
            reverse=True,
        )
    elif category == "failure_missed":
        filtered.sort(
            key=lambda a: (
                a.score_missed,
                a.fn,
                a.mixed_classes_present,
            ),
            reverse=True,
        )
    elif category == "failure_false_positive":
        filtered.sort(
            key=lambda a: (
                a.score_fp,
                a.fp,
                a.localization_errors,
            ),
            reverse=True,
        )
    elif category == "failure_localization":
        filtered.sort(
            key=lambda a: (
                a.score_localization,
                a.localization_errors,
                a.fn,
            ),
            reverse=True,
        )
    elif category == "interesting_low_conf_tp":
        filtered.sort(
            key=lambda a: (
                a.low_conf_tp_count,
                a.mixed_classes_present,
                a.tp,
            ),
            reverse=True,
        )

    return filtered[:n]


def save_summary_csv(
    analyses: list[SampleAnalysis],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id",
            "num_gt",
            "num_preds",
            "tp",
            "fp",
            "fn",
            "localization_errors",
            "mixed_classes_present",
            "low_conf_tp_count",
            "mean_tp_conf",
            "categories",
        ])
        for a in analyses:
            writer.writerow([
                a.sample_id,
                a.num_gt,
                a.num_preds,
                a.tp,
                a.fp,
                a.fn,
                a.localization_errors,
                a.mixed_classes_present,
                a.low_conf_tp_count,
                f"{a.mean_tp_conf:.6f}",
                ";".join(a.categories),
            ])


def save_selected_csv(
    selected: dict[str, list[SampleAnalysis]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "bucket",
            "sample_id",
            "num_gt",
            "num_preds",
            "tp",
            "fp",
            "fn",
            "localization_errors",
            "mixed_classes_present",
            "low_conf_tp_count",
            "mean_tp_conf",
        ])
        for bucket, items in selected.items():
            for a in items:
                writer.writerow([
                    bucket,
                    a.sample_id,
                    a.num_gt,
                    a.num_preds,
                    a.tp,
                    a.fp,
                    a.fn,
                    a.localization_errors,
                    a.mixed_classes_present,
                    a.low_conf_tp_count,
                    f"{a.mean_tp_conf:.6f}",
                ])


def generate_visualizations(
    selected: dict[str, list[SampleAnalysis]],
    data_dir: Path,
    gt_dir: Path,
    pred_dir: Path,
    output_dir: Path,
    conf_threshold: float,
) -> None:
    val_images_dir = data_dir / "images" / "val"

    for bucket, items in selected.items():
        bucket_dir = output_dir / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)

        for a in items:
            sample_id = a.sample_id
            image_path = val_images_dir / f"{sample_id}.png"
            gt_path = gt_dir / f"{sample_id}.txt"
            pred_path = pred_dir / f"{sample_id}.txt"

            gt_boxes = read_ground_truth(gt_path)
            pred_boxes = read_predictions(pred_path)

            visualize_sample(
                image_path=image_path,
                gt_boxes=gt_boxes,
                pred_boxes=pred_boxes,
                output_path=bucket_dir / f"{sample_id}.png",
                conf_threshold=conf_threshold,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select qualitative KITTI examples for report figures"
    )
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--gt-dir", type=Path, default=RAW_DATA_DIR / "label_2")
    parser.add_argument("--pred-dir", type=Path, default=PREDICTIONS_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VISUALIZATIONS_DIR / "qualitative_selection",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold used for displayed predictions",
    )
    parser.add_argument(
        "--analysis-conf",
        type=float,
        default=0.001,
        help="Confidence threshold used when analyzing predictions",
    )
    parser.add_argument(
        "--low-conf-tp-max",
        type=float,
        default=0.5,
        help="Maximum confidence considered 'low-confidence TP'",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="How many examples to save per bucket",
    )
    args = parser.parse_args()

    val_ids = get_val_sample_ids(args.data_dir)

    analyses: list[SampleAnalysis] = []

    for sample_id in val_ids:
        gt_path = args.gt_dir / f"{sample_id}.txt"
        pred_path = args.pred_dir / f"{sample_id}.txt"

        gt_boxes = read_ground_truth(gt_path)
        pred_boxes = read_predictions(pred_path)
        pred_boxes = filter_target_preds(pred_boxes, conf_threshold=args.analysis_conf)

        analysis = analyze_sample(
            sample_id=sample_id,
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            low_conf_tp_max=args.low_conf_tp_max,
        )
        analyses.append(analysis)

    selected = {
        "success_perfect": select_top(analyses, "success_perfect", args.top_k),
        "success_mixed_scene": select_top(analyses, "success_mixed_scene", args.top_k),
        "failure_missed": select_top(analyses, "failure_missed", args.top_k),
        "failure_false_positive": select_top(analyses, "failure_false_positive", args.top_k),
        "failure_localization": select_top(analyses, "failure_localization", args.top_k),
        "interesting_low_conf_tp": select_top(analyses, "interesting_low_conf_tp", args.top_k),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_summary_csv(analyses, args.output_dir / "all_samples_summary.csv")
    save_selected_csv(selected, args.output_dir / "selected_examples.csv")

    generate_visualizations(
        selected=selected,
        data_dir=args.data_dir,
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        output_dir=args.output_dir,
        conf_threshold=args.conf,
    )

    print(f"Saved qualitative selections to: {args.output_dir}")


if __name__ == "__main__":
    main()