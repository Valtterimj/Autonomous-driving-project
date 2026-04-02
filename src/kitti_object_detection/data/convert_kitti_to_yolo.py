import shutil
from pathlib import Path

from kitti_object_detection.data.kitti_reader import KITTISample
from kitti_object_detection.data.image_utils import get_image_size
from kitti_object_detection.data.kitti_labels import read_kitti_label_file, filter_target_classes, convert_kitti_object_to_yolo, format_yolo_label

def convert_kitti_label_file_to_yolo(
        iamge_path: Path,
        kitti_label_path: Path,
        yolo_label_path: Path,
) -> int:
    """
    Convert one KITTI label file to one YOLO label file.
    Returns the number of YOLO objects written
    """

    image_width, image_height = get_image_size(iamge_path)

    objects = read_kitti_label_file(kitti_label_path)
    filter_objects = filter_target_classes(objects)

    yolo_lines: list[str] = []

    for obj in filter_objects:
        yolo_label = convert_kitti_object_to_yolo(
            obj=obj,
            image_width=image_width,
            image_height=image_height,
        )
        if yolo_label is None:
            continue

        yolo_lines.append(format_yolo_label(yolo_label))
    
    yolo_label_path.parent.mkdir(parents=True, exist_ok=True)

    with yolo_label_path.open("w", encoding="utf-8") as f:
        for line in yolo_lines:
            f.write(line + "\n")

    return len(yolo_lines)

def process_split(
        samples: list[KITTISample],
        output_root: Path,
        split_name: str,
) -> tuple[int, int]:
    """
    Process one split into YOLO directory structure:
    output_root/
        images/{split_name}/
        labels/{split_name}/
    Returns:
        (num_images_processed, num_objects_writtten)
    """

    image_output_dir = output_root / "images" / split_name
    label_output_dir = output_root / "labels" / split_name

    image_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    num_images_processed = 0
    num_objects_processed = 0

    for sample in samples:
        output_image_path = image_output_dir / sample.image_path.name
        output_label_path = label_output_dir / sample.label_path.name

        shutil.copy2(sample.image_path, output_image_path)

        num_written = convert_kitti_label_file_to_yolo(
            iamge_path=sample.image_path,
            kitti_label_path=sample.label_path,
            yolo_label_path=output_label_path,
        )

        num_images_processed += 1
        num_objects_processed += num_written

    return num_images_processed, num_objects_processed
