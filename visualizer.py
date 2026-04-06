from pathlib import Path
from typing import List

import cv2
import numpy as np
import PIL.Image

import config
from detector import Detection


def overlay_masks(image: PIL.Image.Image, detections: List[Detection], output_path: str) -> str:
    img = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    overlay = img_bgr.copy()

    for det in detections:
        color = config.PROMPTS[det.label]  # BGR
        mask = det.mask  # (H, W) bool

        overlay[mask] = (
            np.array(overlay[mask], dtype=np.float32) * (1 - config.MASK_ALPHA)
            + np.array(color, dtype=np.float32) * config.MASK_ALPHA
        ).astype(np.uint8)

        # Draw bounding box
        x0, y0, x1, y1 = (int(v) for v in det.box)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)

        # Draw label
        label_text = f"{det.label} {det.score:.2f}"
        cv2.putText(overlay, label_text, (x0, max(y0 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    _draw_status(overlay, detected=len(detections) > 0)
    cv2.imwrite(output_path, overlay)
    return output_path


def _draw_status(img: np.ndarray, detected: bool) -> None:
    h, w = img.shape[:2]
    if detected:
        text, bg_color, text_color = "FIRE DETECTED", (0, 0, 180), (255, 255, 255)
    else:
        text, bg_color, text_color = "CLEAR", (0, 140, 0), (255, 255, 255)

    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 8
    x1, y1 = 10, 10
    x2, y2 = x1 + tw + pad * 2, y1 + th + baseline + pad * 2
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    cv2.putText(img, text, (x1 + pad, y2 - pad - baseline // 2),
                font, scale, text_color, thickness, cv2.LINE_AA)
