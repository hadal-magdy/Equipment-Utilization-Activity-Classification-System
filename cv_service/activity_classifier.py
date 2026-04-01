"""
Activity Classifier

Maps optical-flow motion signals → activity labels without requiring a
trained classifier.  This is intentional for a prototype: it keeps the
system self-contained (no model weights to download) while still being
surprisingly accurate for construction equipment.

The logic mirrors what a human observer notices:
  DIGGING        → arm moving **downward** (positive dy in upper region)
  SWINGING/LOADING → arm moving **laterally** (large |dx| in upper region)
  DUMPING        → arm moving **upward-then-lateral** OR high upper magnitude
                   while machine heading changes (large dx + negative dy)
  WAITING        → negligible motion everywhere

In production you would replace / augment this with a lightweight
sequence model (e.g., a GRU over 10-frame motion vectors) fine-tuned
on labelled excavator footage.
"""

from enum import Enum
from motion_analyzer import MotionResult


class Activity(str, Enum):
    WAITING          = "WAITING"
    DIGGING          = "DIGGING"
    SWINGING_LOADING = "SWINGING_LOADING"
    DUMPING          = "DUMPING"


# ── Thresholds (tune per camera / resolution) ───────────────────────────────
_MIN_ACTIVE_MAG   = 1.5   # below this → WAITING regardless
_LATERAL_BIAS     = 1.4   # |dx| must be N× |dy| to call lateral motion
_DUMP_UP_THRESH   = -0.8  # negative dy (upward) in upper region → lifting


def classify(motion: MotionResult, equipment_class: str = "excavator") -> Activity:
    """
    Derive the current activity from a MotionResult.

    Parameters
    ----------
    motion : MotionResult
        Output of RegionMotionAnalyzer.analyse()
    equipment_class : str
        "excavator" | "dump_truck" (affects heuristic weights)
    """
    if not motion.is_active:
        return Activity.WAITING

    upper_mag = motion.upper_magnitude
    u_dx, u_dy = motion.upper_flow_vec

    # ── Dump trucks have simpler activity vocabulary ──────────────────────
    if equipment_class == "dump_truck":
        # A dump truck either drives (tracks active) or dumps (bed tilts)
        if motion.motion_source == "arm_only":
            return Activity.DUMPING        # bed raising counts as arm_only
        return Activity.SWINGING_LOADING   # moving-to-drop-zone

    # ── Excavator heuristics ─────────────────────────────────────────────
    if upper_mag < _MIN_ACTIVE_MAG:
        return Activity.WAITING

    abs_dx, abs_dy = abs(u_dx), abs(u_dy)

    # Lateral swing dominates → SWINGING / LOADING
    if abs_dx > _LATERAL_BIAS * abs_dy:
        return Activity.SWINGING_LOADING

    # Arm moving upward (negative dy) with lateral component → DUMPING
    if u_dy < _DUMP_UP_THRESH and abs_dx > 0.3:
        return Activity.DUMPING

    # Arm moving downward (positive dy) → DIGGING
    if u_dy > 0.5:
        return Activity.DIGGING

    # Fallback: active but motion direction is ambiguous
    return Activity.DIGGING
