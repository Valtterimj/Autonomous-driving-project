from dataclasses import dataclass
from pathlib import Path


TARGET_CLASSSES = {"Car", "Pedestrian", "Cyclist"}
CLASS_TO_ID = {
    "Car": 0, 
    "Pedestrian": 1,
    "Cyclist": 2,
}


@dataclass
class KITTIObject:
    class_name: str
    truncation: float
    occlusion: int
    aplha: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    dimensions: tuple[float, float, float]
    location: tuple[float, float, float]
    rotation_y: float

@dataclass
class YOLOLabel:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

def parse_kitti_label_line(line: str) -> KITTIObject:
    """
    Parse one line from KITTI lable file

    Format:
    type truncation occlusion aplha bbox_left bbox_top bbox_right bbox_bottom height width lenght x y z rotation_y
    """

    parts = line.strip().split()

    if len(parts) != 15:
        raise ValueError(f"Expected 15 fields in KITTI lable line, got {len(parts)}: {line}")
    
    class_name = parts[0]
    truncation = float(parts[1])
    occlusion = float(parts[2])
    aplha = float(parts[3])

    xmin = float(parts[4])
    ymin = float(parts[5])
    xmax = float(parts[6])
    ymax = float(parts[7])

    height = float(parts[8])
    width = float(parts[9])
    lenght = float(parts[10])

    x = float(parts[11])
    y = float(parts[12])
    z = float(parts[13])
    rotation_y = float(parts[14])

    return KITTIObject(
        class_name=class_name,
        truncation=truncation,
        occlusion=occlusion,
        aplha=aplha,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        dimensions=(height, width, lenght),
        location=(x, y, z),
        rotation_y=rotation_y,
    )

def read_kitti_label_file(label_file: Path) -> list[KITTIObject]:
    """
    Read all objects from one KITTI label file.
    """
    if not label_file.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")
    
    objects: list[KITTIObject] = []

    with label_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            objects.append(parse_kitti_label_line(line))
    
    return objects

def is_target_class(obj: KITTIObject) -> bool:
    return obj.class_name in TARGET_CLASSSES

def filter_target_classes(objects: list[KITTIObject]) -> list[KITTIObject]:
    return [obj for obj in objects if is_target_class(obj)]

def class_name_to_id(class_name: str) -> int:
    if class_name not in CLASS_TO_ID:
        raise ValueError(f"Unsupported class name: {class_name}")
    return CLASS_TO_ID[class_name]

def clamp_bbox(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: int,
    image_height: int, 
) -> tuple[float, float, float, float]:
    """
    Clamp bbox coordinates to valid image boundaries
    """
    xmin = max(0.0, min(xmin, float(image_width)))
    ymin = max(0.0, min(ymin, float(image_height)))
    xmax = max(0.0, min(xmax, float(image_width)))
    ymax = max(0.0, min(ymax, float(image_height)))
    return xmin, ymin, xmax, ymax

def convert_kitti_object_to_yolo(
        obj: KITTIObject,
        image_width: int,
        image_height: int,
        min_box_size: float = 1.0
) -> YOLOLabel | None:
    """
    Convert one KITTI object to YOLO label format.
    Returns None if object is not one of the target classes, bbox becoms invalid after clamping of bbox is too small.
    """
    if not is_target_class(obj):
        return None
    
    xmin, ymin, xmax, ymax = clamp_bbox(
        obj.xmin,
        obj.ymin,
        obj.xmax,
        obj.ymax,
        image_width,
        image_height
    )

    box_width = xmax - xmin
    box_height = ymax - ymin

    if box_width <= 0 or box_height <= 0:
        return None
    
    if box_width < min_box_size or box_height < min_box_size:
        return None
    
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    
    x_center /= float(image_width)
    y_center /= float(image_height)
    box_width /= float(image_width)
    box_height /= float(image_height)

    return YOLOLabel(
        class_id=class_name_to_id(obj.class_name),
        x_center=x_center,
        y_center=y_center,
        width=box_width,
        height=box_height
    )

def format_yolo_label(label: YOLOLabel) -> str:
    """Format one YOLO label as a text line"""
    return(
        f"{label.class_id} "
        f"{label.x_center:.6f} "
        f"{label.y_center:.6f} "
        f"{label.width:.6f} "
        f"{label.height:.6f} "
    )