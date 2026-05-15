from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLOWorld


MODEL_NAME = "yolov8s-worldv2.pt"
MODEL_IMAGE_SIZE = 960
MODEL_CONFIDENCE_THRESHOLD = 0.18
MODEL_IOU_THRESHOLD = 0.45
PERSON_LABEL = "person"
TARGET_CLASSES = [
    "person",
    "pen",
    "pencil",
    "marker",
    "eraser",
    "notebook",
    "book",
]
CLASS_CONFIDENCE_THRESHOLDS = {
    "person": 0.28,
    "pen": 0.38,
    "pencil": 0.38,
    "marker": 0.34,
    "eraser": 0.36,
    "notebook": 0.30,
}
CLASS_MIN_AREA_RATIO = {
    "person": 0.03,
    "pen": 0.0010,
    "pencil": 0.0010,
    "marker": 0.0012,
    "eraser": 0.0015,
    "notebook": 0.0100,
}


@dataclass
class DetectionResult:
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    source: str
    description: str
    blocked_by_human: bool

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["confidence_percent"] = round(self.confidence * 100, 1)
        return payload


def resource_path(relative_path: str) -> Path:
    return Path(__file__).resolve().parent / relative_path


def box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    start_x, start_y, end_x, end_y = box
    return ((start_x + end_x) / 2, (start_y + end_y) / 2)


def point_inside_box(point: tuple[float, float], box: tuple[int, int, int, int]) -> bool:
    px, py = point
    start_x, start_y, end_x, end_y = box
    return start_x <= px <= end_x and start_y <= py <= end_y


def boxes_intersect(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def expand_box(
    box: tuple[int, int, int, int],
    frame_shape: tuple[int, int, int],
    width_ratio: float = 0.08,
    height_ratio: float = 0.08,
) -> tuple[int, int, int, int]:
    frame_height, frame_width = frame_shape[:2]
    start_x, start_y, end_x, end_y = box
    width = end_x - start_x
    height = end_y - start_y
    pad_x = int(width * width_ratio)
    pad_y = int(height * height_ratio)
    return (
        max(0, start_x - pad_x),
        max(0, start_y - pad_y),
        min(frame_width - 1, end_x + pad_x),
        min(frame_height - 1, end_y + pad_y),
    )


def normalize_label(label: str) -> str:
    normalized = label.strip().lower()
    if normalized == "book":
        return "notebook"
    return normalized


def box_area_ratio(box: tuple[int, int, int, int], frame_shape: tuple[int, int, int]) -> float:
    frame_height, frame_width = frame_shape[:2]
    start_x, start_y, end_x, end_y = box
    width = max(0, end_x - start_x)
    height = max(0, end_y - start_y)
    return (width * height) / float(frame_width * frame_height)


def passes_detection_filters(
    label: str,
    confidence: float,
    box: tuple[int, int, int, int],
    frame_shape: tuple[int, int, int],
) -> bool:
    minimum_confidence = CLASS_CONFIDENCE_THRESHOLDS.get(label, MODEL_CONFIDENCE_THRESHOLD)
    if confidence < minimum_confidence:
        return False

    minimum_area_ratio = CLASS_MIN_AREA_RATIO.get(label, 0.001)
    if box_area_ratio(box, frame_shape) < minimum_area_ratio:
        return False

    return True


def is_safe_from_people(
    candidate: DetectionResult,
    detections: list[DetectionResult],
    frame_shape: tuple[int, int, int],
) -> bool:
    for detection in detections:
        if detection.label != PERSON_LABEL:
            continue
        protected_zone = expand_box(detection.box, frame_shape)
        if boxes_intersect(candidate.box, protected_zone):
            return False
        if point_inside_box(box_center(candidate.box), protected_zone):
            return False
    return True


class ProCVDetector:
    def __init__(self, model_path: Optional[Path] = None) -> None:
        resolved_path = Path(model_path) if model_path else resource_path(MODEL_NAME)
        self.model = YOLOWorld(str(resolved_path))
        self.model.set_classes(TARGET_CLASSES)

    def detect_frame(self, frame: np.ndarray) -> tuple[Optional[DetectionResult], list[DetectionResult]]:
        prediction = self.model.predict(
            source=frame,
            imgsz=MODEL_IMAGE_SIZE,
            conf=MODEL_CONFIDENCE_THRESHOLD,
            iou=MODEL_IOU_THRESHOLD,
            verbose=False,
            device="cpu",
        )[0]

        candidates: list[DetectionResult] = []
        if prediction.boxes is None:
            return None, candidates

        names = prediction.names
        for box_values, confidence, class_index in zip(
            prediction.boxes.xyxy.tolist(),
            prediction.boxes.conf.tolist(),
            prediction.boxes.cls.tolist(),
        ):
            raw_label = names[int(class_index)]
            label = normalize_label(raw_label)
            start_x, start_y, end_x, end_y = [int(value) for value in box_values]
            box = (start_x, start_y, end_x, end_y)
            confidence = float(confidence)
            if not label or not passes_detection_filters(label, confidence, box, frame.shape):
                continue

            candidates.append(
                DetectionResult(
                    label=label,
                    confidence=confidence,
                    box=box,
                    source="Pro CV",
                    description=f"Pro CV recognized a {label}.",
                    blocked_by_human=False,
                )
            )

        if not candidates:
            return None, candidates

        candidates.sort(key=lambda item: item.confidence, reverse=True)
        non_person_candidates = [item for item in candidates if item.label != PERSON_LABEL]
        safe_non_person_candidates = [
            item for item in non_person_candidates if is_safe_from_people(item, candidates, frame.shape)
        ]

        if safe_non_person_candidates:
            best_object = safe_non_person_candidates[0]
            best_object.description = (
                f"Pro CV recognized a {best_object.label} and kept a safe distance from people."
            )
            return best_object, candidates

        if non_person_candidates:
            blocked_object = non_person_candidates[0]
            blocked_object.blocked_by_human = any(item.label == PERSON_LABEL for item in candidates)
            blocked_object.description = (
                f"Pro CV recognized a {blocked_object.label}, but it is too close to a person. "
                "Action remains blocked for safety."
            )
            return blocked_object, candidates

        person_candidate = candidates[0]
        person_candidate.blocked_by_human = True
        person_candidate.description = "Human detected in the frame. No action is allowed."
        return person_candidate, candidates

    @staticmethod
    def annotate_frame(frame: np.ndarray, detection: Optional[DetectionResult]) -> np.ndarray:
        annotated = frame.copy()
        if detection is None:
            return annotated

        color = (0, 255, 0)
        if detection.blocked_by_human:
            color = (0, 140, 255)
        if detection.label == PERSON_LABEL:
            color = (0, 0, 255)

        start_x, start_y, end_x, end_y = detection.box
        status = "Blocked" if detection.blocked_by_human else "Clear"
        cv2.rectangle(annotated, (start_x, start_y), (end_x, end_y), color, 2)
        cv2.putText(
            annotated,
            f"{detection.label} {detection.confidence * 100:.1f}% | {status}",
            (start_x, max(25, start_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        return annotated

    @staticmethod
    def encode_jpeg(frame: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise ValueError("Could not encode image.")
        return encoded.tobytes()
