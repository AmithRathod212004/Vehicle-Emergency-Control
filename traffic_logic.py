"""Traffic signal control logic for emergency-priority intersection management."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

LANES = ("North", "South", "East", "West")


def choose_emergency_lane(detections: Iterable[dict]) -> Optional[str]:
    """Return the lane of the highest-confidence ambulance detection."""
    best_lane: Optional[str] = None
    best_confidence = -1.0

    for detection in detections:
        lane = detection.get("lane")
        confidence = float(detection.get("confidence", 0.0))
        if not detection.get("is_ambulance") or lane not in LANES:
            continue
        if confidence > best_confidence:
            best_confidence = confidence
            best_lane = lane

    return best_lane


def get_signal_state(detections: Iterable[dict]) -> Dict[str, str]:
    """Build a 4-way signal map. Emergency lane gets green; all others stay red."""
    emergency_lane = choose_emergency_lane(detections)
    return {
        lane: "green" if lane == emergency_lane else "red"
        for lane in LANES
    }
