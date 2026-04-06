from typing import List

from detector import Detection


def print_alerts(detections: List[Detection], prefix: str = "") -> bool:
    tag = f"[{prefix}] " if prefix else ""
    if not detections:
        print(f"  {tag}[OK] No fire or smoke detected.")
        return False

    for det in detections:
        print(f"  {tag}[ALERT] {det.label.capitalize()} detected! Confidence: {det.score:.2f}")
    return True
