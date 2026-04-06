import argparse
import random
from pathlib import Path

import PIL.Image

import config
from alerter import print_alerts
from detector import FireDetector
from visualizer import overlay_masks

FIRE_DIR = Path(__file__).parent / "fire_dataset" / "fire_images"
NON_FIRE_DIR = Path(__file__).parent / "fire_dataset" / "non_fire_images"
BATCH_SAMPLE_COUNT = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Fire & Smoke Detection using SAM3")
    parser.add_argument("--image", default=None, help="Path to input image (omit for batch mode)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {config.CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save result images (default: results/)",
    )
    return parser.parse_args()


def process_image(image_path: Path, detector: FireDetector, output_dir: Path) -> None:
    image = PIL.Image.open(image_path).convert("RGB")
    detections = detector.detect(image)
    print_alerts(detections, prefix=image_path.name)
    output_path = str(output_dir / f"result_{image_path.name}")
    overlay_masks(image, detections, output_path)
    print(f"  → saved: {output_path}")


def main():
    import torch
    gpu_info = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "GPU: not available (CPU mode)"
    print(f"Fire Detection v{config.VERSION} | {gpu_info}")
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = FireDetector(threshold=args.threshold)

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: image not found: {image_path}")
            return
        process_image(image_path, detector, output_dir)
    else:
        # Batch mode: sample randomly from fire and non_fire dirs
        fire_images = list(FIRE_DIR.glob("*.png")) + list(FIRE_DIR.glob("*.jpg"))
        non_fire_images = list(NON_FIRE_DIR.glob("*.png")) + list(NON_FIRE_DIR.glob("*.jpg"))

        sampled = (
            random.sample(fire_images, min(BATCH_SAMPLE_COUNT, len(fire_images)))
            + random.sample(non_fire_images, min(BATCH_SAMPLE_COUNT, len(non_fire_images)))
        )
        random.shuffle(sampled)

        print(f"Batch mode: processing {len(sampled)} images ({BATCH_SAMPLE_COUNT} fire + {BATCH_SAMPLE_COUNT} non-fire)\n")
        for i, image_path in enumerate(sampled, 1):
            print(f"[{i}/{len(sampled)}] {image_path.name}")
            process_image(image_path, detector, output_dir)
            print()


if __name__ == "__main__":
    main()
