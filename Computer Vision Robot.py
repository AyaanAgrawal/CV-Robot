from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

try:
    from ultralytics import YOLOWorld
except ImportError as exc:
    python_path = sys.executable
    raise SystemExit(
        "Ultralytics is not installed for this Python interpreter.\n"
        f"Current Python: {python_path}\n"
        f"Install it with:\n\"{python_path}\" -m pip install ultralytics"
    ) from exc


WINDOW_TITLE = "Computer Vision Robot"
MODEL_NAME = "yolov8s-worldv2.pt"
MODEL_IMAGE_SIZE = 960
MODEL_CONFIDENCE_THRESHOLD = 0.18
MODEL_IOU_THRESHOLD = 0.45
CONFIRMATION_FRAMES = 4
MIN_CONFIRMATIONS = 3
HUMAN_BLOCK_FRAMES = 2
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


def resource_path(relative_path: str) -> Path:
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base_path / relative_path


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


def is_holding_detected_object(
    person_box: tuple[int, int, int, int],
    detections: list[DetectionResult],
) -> bool:
    start_x, start_y, end_x, end_y = person_box
    person_height = end_y - start_y
    hand_zone_top = start_y + int(person_height * 0.25)

    for detection in detections:
        if detection.label == PERSON_LABEL:
            continue

        center_x, center_y = box_center(detection.box)
        if center_y < hand_zone_top:
            continue

        if point_inside_box((center_x, center_y), person_box):
            return True

    return False


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


def load_detector() -> YOLOWorld:
    model = YOLOWorld(str(resource_path(MODEL_NAME)))
    model.set_classes(TARGET_CLASSES)
    return model


def detect_with_pro_cv(
    frame,
    model: YOLOWorld,
) -> tuple[Optional[DetectionResult], list[DetectionResult]]:
    prediction = model.predict(
        source=frame,
        imgsz=MODEL_IMAGE_SIZE,
        conf=MODEL_CONFIDENCE_THRESHOLD,
        iou=MODEL_IOU_THRESHOLD,
        verbose=False,
        device="cpu",
    )[0]

    names = prediction.names
    candidates: list[DetectionResult] = []

    if prediction.boxes is None:
        return None, candidates

    xyxy_values = prediction.boxes.xyxy.tolist()
    confidence_values = prediction.boxes.conf.tolist()
    class_values = prediction.boxes.cls.tolist()

    for box_values, confidence, class_index in zip(xyxy_values, confidence_values, class_values):
        raw_label = names[int(class_index)]
        label = normalize_label(raw_label)
        if not label:
            continue

        start_x, start_y, end_x, end_y = [int(value) for value in box_values]
        box = (start_x, start_y, end_x, end_y)
        confidence = float(confidence)
        if not passes_detection_filters(label, confidence, box, frame.shape):
            continue

        result = DetectionResult(
            label=label,
            confidence=confidence,
            box=box,
            source="Pro CV",
            description=f"Pro CV recognized a {label}.",
        )
        candidates.append(result)

    if not candidates:
        return None, candidates

    candidates.sort(key=lambda item: item.confidence, reverse=True)

    non_person_candidates = [
        item
        for item in candidates
        if item.label != PERSON_LABEL
    ]
    safe_non_person_candidates = [
        item
        for item in non_person_candidates
        if is_safe_from_people(item, candidates, frame.shape)
    ]
    if safe_non_person_candidates:
        best_object = safe_non_person_candidates[0]
        best_object.description = f"Pro CV recognized a {best_object.label} and kept a safe distance from people."
        return best_object, candidates

    if non_person_candidates:
        blocked_object = non_person_candidates[0]
        blocked_object.description = (
            f"Pro CV recognized a {blocked_object.label}, but it is too close to a person. "
            "Action remains blocked for safety."
        )
        return blocked_object, candidates

    return None, candidates


def choose_consensus_detection(
    detection_groups: list[tuple[Optional[DetectionResult], list[DetectionResult]]],
) -> tuple[Optional[DetectionResult], list[DetectionResult], bool]:
    confirmed_by_label: dict[str, list[DetectionResult]] = {}
    all_detections: list[DetectionResult] = []
    person_frames = 0

    for detection, detections_in_frame in detection_groups:
        all_detections.extend(detections_in_frame)
        if any(item.label == PERSON_LABEL for item in detections_in_frame):
            person_frames += 1
        if detection is None:
            continue
        confirmed_by_label.setdefault(detection.label, []).append(detection)

    human_blocked = person_frames >= HUMAN_BLOCK_FRAMES
    if not confirmed_by_label:
        return None, all_detections, human_blocked

    best_label = None
    best_group: list[DetectionResult] = []

    for label, group in confirmed_by_label.items():
        if len(group) < MIN_CONFIRMATIONS:
            continue
        if not best_group:
            best_label = label
            best_group = group
            continue

        current_score = (len(group), sum(item.confidence for item in group) / len(group))
        best_score = (len(best_group), sum(item.confidence for item in best_group) / len(best_group))
        if current_score > best_score:
            best_label = label
            best_group = group

    if not best_group or best_label is None:
        return None, all_detections, human_blocked

    best_detection = max(best_group, key=lambda item: item.confidence)
    consensus_detections = [
        item for item in all_detections
        if item.label == best_label
    ]
    return best_detection, consensus_detections, human_blocked


def detect_with_confirmation(cap, model: YOLOWorld) -> tuple[Optional[DetectionResult], list[DetectionResult], bool]:
    detection_groups: list[tuple[Optional[DetectionResult], list[DetectionResult]]] = []

    for _ in range(CONFIRMATION_FRAMES):
        ok, frame = cap.read()
        if not ok:
            continue
        detection_groups.append(detect_with_pro_cv(frame, model))

    if not detection_groups:
        return None, [], False

    return choose_consensus_detection(detection_groups)


def get_primary_person_detection(all_detections: list[DetectionResult]) -> Optional[DetectionResult]:
    person_detections = [item for item in all_detections if item.label == PERSON_LABEL]
    if not person_detections:
        return None
    return max(person_detections, key=lambda item: item.confidence)


def draw_overlay(
    frame,
    lines: list[str],
    display_detection: Optional[DetectionResult],
) -> None:
    if display_detection is not None:
        color = (0, 255, 0)
        thickness = 2
        if display_detection.label == PERSON_LABEL:
            color = (0, 0, 255)

        start_x, start_y, end_x, end_y = display_detection.box
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, thickness)
        cv2.putText(
            frame,
            f"{display_detection.label} {display_detection.confidence * 100:.1f}%",
            (start_x, max(25, start_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    y = 25
    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        y += 28


def print_result(result: DetectionResult, all_detections: list[DetectionResult], blocked_by_human: bool) -> None:
    width = result.box[2] - result.box[0]
    height = result.box[3] - result.box[1]
    safety_status = "Blocked by human safety" if blocked_by_human else "Clear"
    other_items = [
        f"{item.label} {item.confidence * 100:.1f}%"
        for item in all_detections
        if item.box != result.box or item.label != result.label
    ]

    print("\n" + "=" * 48)
    print(" PRO CV DETECTION RESULT")
    print("=" * 48)
    print(f" Target     : {result.label.title()}")
    print(f" Confidence : {result.confidence * 100:.1f}%")
    print(f" Safety     : {safety_status}")
    print(f" Source     : {result.source}")
    print(f" Summary    : {result.description}")
    print(f" Box        : x={result.box[0]}, y={result.box[1]}, w={width}, h={height}")
    if other_items:
        print(f" Also Seen   : {', '.join(other_items[:5])}")
    print("=" * 48 + "\n")


def main() -> None:
    try:
        model = load_detector()
    except Exception as exc:
        print(f"Could not load the Pro CV detector: {exc}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("==============================================")
    print("SPACE : run Pro CV detection")
    print("Q     : quit")
    print("==============================================")
    print("Small-item vocabulary enabled: pen, pencil, marker, eraser, notebook.")
    print("Detection now requires repeated agreement across multiple frames.")
    print("People remain protected, but object names stay visible when blocked.")

    overlay_lines = [
        "SPACE: Pro CV detect",
        "Targets: pen, eraser, notebook, person-with-object",
        "Q: quit",
    ]
    last_detection: Optional[DetectionResult] = None
    last_display_detection: Optional[DetectionResult] = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to capture frame.")
            break

        display_frame = frame.copy()
        draw_overlay(display_frame, overlay_lines, last_display_detection)
        cv2.imshow(WINDOW_TITLE, display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key != 32:
            continue

        detection, all_detections, human_blocked = detect_with_confirmation(cap, model)
        if detection is None:
            if human_blocked:
                blocker_detection = get_primary_person_detection(all_detections)
                overlay_lines = [
                    "Safety block active",
                    "Object may be visible but still protected",
                    "SPACE: Pro CV detect | Q: quit",
                ]
                last_display_detection = detection or blocker_detection
                print("\nHuman detected nearby. Object name is shown, but safety block remains active.\n")
            else:
                overlay_lines = [
                    "No valid target found",
                    "Show the item clearly for a moment",
                    "SPACE: Pro CV detect | Q: quit",
                ]
                last_display_detection = None
                print("\nNo confirmed target detected. False positives are being filtered out.\n")
            last_detection = None
            continue

        if human_blocked:
            overlay_lines = [
                f"Detected: {detection.label.title()}",
                "Blocked: too close to person",
                "SPACE: Pro CV detect | Q: quit",
            ]
        else:
            overlay_lines = [
                f"Target: {detection.label.title()}",
                f"Confidence: {detection.confidence * 100:.1f}%",
                "SPACE: Pro CV detect | Q: quit",
            ]
        last_detection = detection
        last_display_detection = detection
        print_result(detection, all_detections, human_blocked)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
