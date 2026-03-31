"""
CV Service — Main Entry Point

Pipeline per frame:
  1. Decode frame from video source (file or RTSP stream)
  2. Detect + track equipment with YOLOv8 / ByteTrack
  3. Compute region-based optical flow per detection
  4. Classify activity from flow vectors
  5. Update per-machine time accumulators
  6. Publish EquipmentEvent to Kafka
  7. (Optional) write annotated frame to output video

Usage:
  python main.py --source /data/video.mp4
  python main.py --source rtsp://camera_ip/stream
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections import defaultdict
from datetime import timedelta
from typing import Dict, Tuple

import cv2
import numpy as np

from detector import EquipmentDetector, Detection
from motion_analyzer import RegionMotionAnalyzer, MotionResult
from activity_classifier import classify, Activity
from kafka_producer import EquipmentKafkaProducer, EquipmentEvent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("cv_service")


# ── State container for each tracked machine ────────────────────────────────

class MachineState:
    def __init__(self, equipment_id: str, equipment_class: str, fps: float):
        self.equipment_id    = equipment_id
        self.equipment_class = equipment_class
        self.fps             = fps

        # time accumulators (in frames, converted to seconds on publish)
        self.total_frames  = 0
        self.active_frames = 0
        self.idle_frames   = 0

        self.current_state    = "INACTIVE"
        self.current_activity = Activity.WAITING
        self.motion_source    = "none"

    def update(self, motion: MotionResult, activity: Activity):
        self.total_frames  += 1
        self.current_state = "ACTIVE" if motion.is_active else "INACTIVE"
        self.motion_source = motion.motion_source
        self.current_activity = activity

        if motion.is_active:
            self.active_frames += 1
        else:
            self.idle_frames += 1

    @property
    def total_seconds(self) -> float:
        return self.total_frames / self.fps

    @property
    def active_seconds(self) -> float:
        return self.active_frames / self.fps

    @property
    def idle_seconds(self) -> float:
        return self.idle_frames / self.fps

    @property
    def utilization_percent(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return round(100.0 * self.active_frames / self.total_frames, 1)

    def to_event(self, frame_id: int, timestamp_sec: float, bbox) -> EquipmentEvent:
        ts = str(timedelta(seconds=timestamp_sec))[:-3]   # "HH:MM:SS.mmm"
        return EquipmentEvent(
            frame_id=frame_id,
            equipment_id=self.equipment_id,
            equipment_class=self.equipment_class,
            timestamp=ts,
            utilization={
                "current_state":    self.current_state,
                "current_activity": self.current_activity.value,
                "motion_source":    self.motion_source,
            },
            time_analytics={
                "total_tracked_seconds": round(self.total_seconds, 2),
                "total_active_seconds":  round(self.active_seconds, 2),
                "total_idle_seconds":    round(self.idle_seconds, 2),
                "utilization_percent":   self.utilization_percent,
            },
            bbox=list(bbox),
        )


# ── Main pipeline ────────────────────────────────────────────────────────────

def run(
    source: str,
    kafka_servers: str,
    model_path: str,
    output_path: str | None,
    display: bool,
    frame_skip: int,
):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Source: {source} | {w}×{h} @ {fps:.1f} fps")

    # Optional output video
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps / max(1, frame_skip), (w, h))

    detector  = EquipmentDetector(model_path=model_path)
    analyzer  = RegionMotionAnalyzer()
    producer  = EquipmentKafkaProducer(bootstrap_servers=kafka_servers)

    machine_states: Dict[str, MachineState] = {}
    prev_gray = None
    frame_id  = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream.")
                break

            frame_id += 1

            # Skip frames to reduce processing load (still increments time)
            if frame_id % max(1, frame_skip) != 0:
                continue

            timestamp_sec = frame_id / fps
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect + track
            detections = detector.detect(frame)

            # Current states dict for overlay drawing
            overlay_states: Dict[str, dict] = {}

            for det in detections:
                motion = MotionResult(
                    is_active=False, motion_source="none",
                    upper_magnitude=0.0, lower_magnitude=0.0,
                )

                # Compute flow only when we have a previous frame
                if prev_gray is not None:
                    motion = analyzer.analyse(prev_gray, curr_gray, det.bbox)

                activity = classify(motion, det.equipment_class)

                # Lazy-init machine state
                if det.equipment_id not in machine_states:
                    machine_states[det.equipment_id] = MachineState(
                        det.equipment_id, det.equipment_class, fps
                    )

                state = machine_states[det.equipment_id]
                state.update(motion, activity)

                # Publish to Kafka
                event = state.to_event(frame_id, timestamp_sec, det.bbox)
                producer.send(event)

                overlay_states[det.equipment_id] = {
                    "is_active": motion.is_active,
                    "activity":  activity.value,
                }

            prev_gray = curr_gray

            # Annotate frame
            annotated = detector.draw(frame, detections, overlay_states)

            # Add HUD: utilization summary at top-left
            _draw_hud(annotated, machine_states, timestamp_sec)

            if writer:
                writer.write(annotated)

            if display:
                cv2.imshow("Equipment Utilization", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Display quit.")
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        producer.flush()
        logger.info("CV service shut down cleanly.")


def _draw_hud(frame: np.ndarray, states: Dict[str, MachineState], ts: float):
    """Draw a small utilization summary panel in the top-left corner."""
    x, y = 8, 18
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thick = 1
    line_h = 16

    # Semi-transparent background
    panel_h = 14 + len(states) * line_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (4, 4), (260, panel_h + 8), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"t={ts:.1f}s | machines={len(states)}",
                (x, y), font, scale, (200, 200, 200), thick)

    for i, (eid, st) in enumerate(states.items()):
        color  = (80, 220, 80) if st.current_state == "ACTIVE" else (80, 80, 220)
        label  = (f"{eid} | {st.current_state} | {st.current_activity.value} "
                  f"| util={st.utilization_percent}%")
        cv2.putText(frame, label, (x, y + (i + 1) * line_h),
                    font, scale, color, thick)


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CV Service — Equipment Utilization")
    parser.add_argument("--source",  default=os.getenv("VIDEO_SOURCE", "0"),
                        help="Video file path or RTSP URL (default: webcam)")
    parser.add_argument("--kafka",   default=os.getenv("KAFKA_BOOTSTRAP", "kafka:9092"))
    parser.add_argument("--model",   default=os.getenv("YOLO_MODEL", "yolov8n.pt"))
    parser.add_argument("--output",  default=os.getenv("OUTPUT_VIDEO", None),
                        help="Optional path to save annotated output video")
    parser.add_argument("--display", action="store_true",
                        help="Show live OpenCV window (requires display)")
    parser.add_argument("--skip",    type=int, default=2,
                        help="Process every Nth frame (default: 2, ~halves CPU load)")
    args = parser.parse_args()

    run(
        source=args.source,
        kafka_servers=args.kafka,
        model_path=args.model,
        output_path=args.output,
        display=args.display,
        frame_skip=args.skip,
    )
