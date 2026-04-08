
from pathlib import Path

from kitti_object_detection.data.kitti_reader import read_kitti_samples
from kitti_object_detection.data.convert_kitti_to_yolo import process_split
from kitti_object_detection.data.splits import train_val_split
from kitti_object_detection.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    generate_dataset_yaml,
)


def preprocess_kitti_data(raw_data: Path, output: Path, val_fraction: float = 0.2, seed: int = 42) -> None:

    print(f"reading KITTI samples from: {raw_data}")
    samples = read_kitti_samples(raw_data)
    print(f"Found {len(samples)} total samples")

    train_samples, val_samples = train_val_split(
        samples,
        val_fraction=val_fraction,
        seed=seed,
    )

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")

    print("\nProcessing train split...")
    train_images, train_objects = process_split(
        samples=train_samples,
        output_root=output,
        split_name="train",
    )

    print("Processing val split...")
    val_images, val_objects = process_split(
        samples=val_samples,
        output_root=output,
        split_name="val"
    )

    print(f"\nProcessed dataset written to: {output}")
    print(f"Train: {train_images} images, {train_objects} objects")
    print(f"Val: {val_images} images, {val_objects} objects")

    yaml_path = generate_dataset_yaml(data_dir=output)
    print(f"Dataset YAML written to: {yaml_path}")


def main():

    raw_data_root = RAW_DATA_DIR
    processed_data_root = PROCESSED_DATA_DIR
    val_fraction = 0.2
    seed = 42

    preprocess_kitti_data(
        raw_data=raw_data_root,
        output=processed_data_root,
        val_fraction=val_fraction,
        seed=seed
    )

if __name__ == "__main__":
    main()
