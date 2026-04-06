import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import PIL.Image
import torch

import config

# Make sam3 importable from local source
sys.path.insert(0, str(Path(config.SAM3_SRC_PATH).resolve()))

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class Detection:
    def __init__(self, label: str, score: float, mask: np.ndarray, box: List[float]):
        self.label = label
        self.score = score
        self.mask = mask   # (H, W) bool array
        self.box = box     # [x0, y0, x1, y1] in pixel coords


class FireDetector:
    def __init__(self, threshold: float = config.CONFIDENCE_THRESHOLD):
        print("Loading SAM3 model...")
        model = build_sam3_image_model(
            device=config.DEVICE,
            checkpoint_path=config.CHECKPOINT_PATH,
            load_from_HF=False,
        )
        self.processor = Sam3Processor(model, device=config.DEVICE, confidence_threshold=threshold)
        print("Model loaded.")

    def detect(self, image: PIL.Image.Image) -> List[Detection]:
        state = self.processor.set_image(image)
        detections = []

        for prompt in config.PROMPTS:
            # Reset only text/result state; image backbone features are preserved
            self.processor.reset_all_prompts(state)
            state = self.processor.set_text_prompt(prompt, state)

            masks_logits = state.get("masks_logits")
            scores = state.get("scores")
            boxes = state.get("boxes")

            if masks_logits is None or len(masks_logits) == 0:
                continue

            # Apply configurable threshold to tighten mask area
            masks_np = (masks_logits.squeeze(1).cpu().numpy() > config.MASK_THRESHOLD)
            scores_np = scores.cpu().numpy()
            boxes_np = boxes.cpu().numpy()

            for i in range(len(masks_np)):
                detections.append(Detection(
                    label=prompt,
                    score=float(scores_np[i]),
                    mask=masks_np[i].astype(bool),
                    box=boxes_np[i].tolist(),
                ))

        return detections
