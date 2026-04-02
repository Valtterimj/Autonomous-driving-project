from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

class KITTISample(NamedTuple):
    sample_id: str
    image_path: Path
    label_path: Path
    calib_path: Path | None

def read_kitti_samples(data_root: Path) -> list[KITTISample]:
    """
    Read KITTI raw training data from a folder containgin:
     - image_2/
     - label_2/
    
    Returns a list of matched KITTI samples sorted by sample id
    """

    image_dir = data_root / "image_2"
    label_dir = data_root / 'label_2'
    calib_dir = data_root / 'calib'

    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Missing required folder: {image_dir}")
    if not label_dir.exists() or not label_dir.is_dir():
        raise FileNotFoundError(f"Missing required folder: {label_dir}")

    image_paths = sorted(image_dir.glob("*.png"))
    label_paths = sorted(label_dir.glob("*.txt"))

    if not image_paths:
        raise ValueError(f"No image files in {image_dir}")
    if not label_paths:
        raise ValueError(f"No label files in {label_dir}")

    image_ids = {p.stem for p in image_paths}
    label_ids = {p.stem for p in label_paths}

    missing_images = sorted(label_ids - image_ids)
    missing_labels = sorted(image_ids - label_ids)

    if missing_labels or missing_images:
        message_parts = []
        if missing_labels:
            message_parts.append(
                f"Images without matching labels: {missing_labels[:10]}"
            )
        if missing_images:
            message_parts.append(
                f"Labels without matching images: {missing_labels[:10]}"
            )
        raise ValueError("KITTI image/label mismatch. " + " | ".join(message_parts))

    common_ids = sorted(image_ids & label_ids)
    has_calib = calib_dir.exists() and calib_dir.is_dir()

    samples: list[KITTISample] = []
    for sample_id in common_ids:
        calib_path = calib_dir / f"{sample_id}.txt" if has_calib else None
        if calib_path is not None and calib_path.exists():
            calib_path = None
        
        samples.append(
            KITTISample(
                sample_id=sample_id,
                image_path=image_dir / f"{sample_id}.png",
                label_path=label_dir / f"{sample_id}.txt",
                calib_path=calib_path,
            )
        )
    return samples