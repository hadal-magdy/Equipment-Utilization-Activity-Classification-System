"""
Motion Analyzer — Region-based optical flow for articulated equipment.

The core challenge: an excavator arm can be actively digging while its
tracks are completely still. A naive "did the bounding box move?" check
would wrongly mark it as INACTIVE. We solve this by splitting the bbox
into functional regions and analysing each independently.

Regions (relative to bbox height):
  ─────────────────
  │  UPPER  (0-50%)│  ← cab + arm/boom + bucket
  │─────────────────│
  │  LOWER  (50-100%)│ ← undercarriage / tracks / wheels
  ─────────────────
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


# Motion is considered real when pixel magnitude exceeds this threshold.
# Tune this for your lighting conditions / camera resolution.
MOTION_THRESHOLD_LOW  = 1.5   # pixels per frame — very slight movement
MOTION_THRESHOLD_HIGH = 3.5   # pixels per frame — definitive movement


@dataclass
class MotionResult:
    is_active: bool
    motion_source: str            # "full_body" | "arm_only" | "tracks_only" | "none"
    upper_magnitude: float        # mean optical-flow magnitude in upper region
    lower_magnitude: float        # mean optical-flow magnitude in lower region
    upper_flow_vec: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))  # mean (dx, dy)
    lower_flow_vec: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    overall_magnitude: float = 0.0


class RegionMotionAnalyzer:
    """
    Uses dense optical flow (Farneback) on two consecutive frames within
    a detected bounding box.  The bbox is sliced into upper / lower halves
    so we can detect arm-only or tracks-only activity.
    """

    def __init__(self):
        # Farneback parameters (good default balance of speed vs accuracy)
        self._fb_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

    def analyse(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        bbox: Tuple[int, int, int, int],   # x1, y1, x2, y2  (absolute pixels)
    ) -> MotionResult:
        x1, y1, x2, y2 = bbox
        # Guard: bbox must be at least 20×20 px for flow to be meaningful
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return MotionResult(is_active=False, motion_source="none",
                                upper_magnitude=0.0, lower_magnitude=0.0)

        # Crop to bbox
        prev_crop = prev_gray[y1:y2, x1:x2]
        curr_crop = curr_gray[y1:y2, x1:x2]

        # Dense optical flow over the entire bbox
        flow = cv2.calcOpticalFlowFarneback(prev_crop, curr_crop, None, **self._fb_params)

        # Split into upper / lower halves
        h = flow.shape[0]
        mid = h // 2

        upper_flow = flow[:mid, :, :]
        lower_flow = flow[mid:, :, :]

        upper_mag = float(np.mean(np.sqrt(upper_flow[..., 0]**2 + upper_flow[..., 1]**2)))
        lower_mag = float(np.mean(np.sqrt(lower_flow[..., 0]**2 + lower_flow[..., 1]**2)))
        overall   = float(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))

        upper_vec = (float(np.mean(upper_flow[..., 0])), float(np.mean(upper_flow[..., 1])))
        lower_vec = (float(np.mean(lower_flow[..., 0])), float(np.mean(lower_flow[..., 1])))

        upper_active = upper_mag > MOTION_THRESHOLD_LOW
        lower_active = lower_mag > MOTION_THRESHOLD_LOW

        # Determine motion source and overall active state
        if upper_active and lower_active:
            source = "full_body"
            is_active = True
        elif upper_active:
            source = "arm_only"      # classic articulated-motion scenario
            is_active = True
        elif lower_active:
            source = "tracks_only"   # driving / repositioning
            is_active = True
        else:
            source = "none"
            is_active = False

        return MotionResult(
            is_active=is_active,
            motion_source=source,
            upper_magnitude=upper_mag,
            lower_magnitude=lower_mag,
            upper_flow_vec=upper_vec,
            lower_flow_vec=lower_vec,
            overall_magnitude=overall,
        )
