"""
Equipment Detector

Wraps YOLOv8 (ultralytics) for detecting construction equipment.

COCO-pretrained YOLOv8 knows 80 classes.  Heavy equipment maps roughly to:
  • class 7  → truck
  • class 5  → bus   (sometimes misidentified but similar shape)

For a production system you would fine-tune on a labelled dataset of
excavators / dump trucks (e.g., from ImageNet Construction or a custom
dataset). The fine-tuning note is included in README.md.

We assign equipment_ids based on detection track-ids (ByteTrack via
ultralytics built-in tracker), so "EX-001" persists across frames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO class-ids that represent construction equipment
# Extend this list if you fine-tune the model.
EQUIPMENT_CLASS_MAP = {
    7: "dump_truck",
    5: "dump_truck",   # bus  (fallback)
    # Add custom fine-tuned class IDs here, e.g.:
    # 80: "excavator",
    # 81: "dump_truck",
}

# Prefix map for readable equipment IDs
PREFIX_MAP = {
    "excavator":  "EX",
    "dump_truck": "DT",
    "unknown":    "EQ",
}


@dataclass
class Detection:
    equipment_id: str                    # e.g. "EX-001"
    equipment_class: str                 # "excavator" | "dump_truck"
    bbox: Tuple[int, int, int, int]      # x1, y1, x2, y2
    confidence: float
    track_id: int


class EquipmentDetector:
    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.35):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLOv8 model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

        self.conf_threshold = confidence
        self._id_registry: dict[int, str] = {}   # track_id → equipment_id
        self._class_registry: dict[int, str] = {}
        self._counters: dict[str, int] = {}

    def _get_equipment_id(self, track_id: int, eq_class: str) -> str:
        """Assign a stable, human-readable ID to a tracked object."""
        if track_id not in self._id_registry:
            prefix = PREFIX_MAP.get(eq_class, "EQ")
            self._counters[prefix] = self._counters.get(prefix, 0) + 1
            self._id_registry[track_id] = f"{prefix}-{self._counters[prefix]:03d}"
            self._class_registry[track_id] = eq_class
        return self._id_registry[track_id]

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLOv8 + ByteTrack on a single frame.

        Returns a list of Detection objects for equipment-class objects only.
        If the model hasn't been fine-tuned, we include ALL tracked objects
        and label them "unknown" so the motion pipeline still works.
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf_threshold,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        detections: List[Detection] = []

        if results is None or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for box in boxes:
            if box.id is None:
                continue  # not yet assigned a track ID

            track_id = int(box.id.item())
            cls_id   = int(box.cls.item())
            conf     = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Map to equipment class (or use "unknown" for unlabelled prototype)
            eq_class = EQUIPMENT_CLASS_MAP.get(cls_id, "unknown")

            # In a real deployment, skip unknowns.
            # For prototype / demo, include everything so you can test with any video.
            equipment_id = self._get_equipment_id(track_id, eq_class)

            detections.append(Detection(
                equipment_id=equipment_id,
                equipment_class=eq_class,
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                track_id=track_id,
            ))

        return detections

    def draw(self, frame: np.ndarray, detections: List[Detection],
             states: Optional[dict] = None) -> np.ndarray:
        """Overlay bounding boxes + labels on a frame."""
        overlay = frame.copy()
        states = states or {}

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            info = states.get(det.equipment_id, {})
            is_active = info.get("is_active", False)
            activity  = info.get("activity", "WAITING")

            color = (0, 200, 80) if is_active else (0, 80, 220)   # green / red (BGR)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # Label background
            label = f"{det.equipment_id} | {activity}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw region split line (upper / lower)
            mid_y = (y1 + y2) // 2
            cv2.line(overlay, (x1, mid_y), (x2, mid_y), color, 1, cv2.LINE_AA)

        return overlay
