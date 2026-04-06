import os

VERSION = "1.0.5"

# SAM3 source path (relative to this file)
SAM3_SRC_PATH = os.path.join(os.path.dirname(__file__), "..", "sam3-src")

# Model checkpoint
CHECKPOINT_PATH = "/home/david/.cache/huggingface/hub/models--facebook--sam3/sam3.pt"

# Model settings
DEVICE = "cuda"
CONFIDENCE_THRESHOLD = 0.5

# Mask binarization threshold — raise to tighten mask area (0.0~1.0)
MASK_THRESHOLD = 0.6

# Text prompts — all red (BGR)
PROMPTS = {
    "fire": (0, 0, 255),
    "flame": (0, 0, 255),
    "burning": (0, 0, 255),
    "wildfire": (0, 0, 255),
    "smoke": (0, 0, 255),
    "dense smoke": (0, 0, 255),
}

# Mask overlay alpha (0.0 = transparent, 1.0 = opaque)
MASK_ALPHA = 0.45
