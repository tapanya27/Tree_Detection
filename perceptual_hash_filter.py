from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class HashDecision:
    image_path: Path
    hash_value: np.ndarray
    copied: bool
    distance: int | None


def compute_dhash(image_path: Path, hash_size: int = 8) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten().astype(np.uint8)


def hamming_distance(hash_a: np.ndarray, hash_b: np.ndarray) -> int:
    return int(np.count_nonzero(hash_a != hash_b))


def iter_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def filter_similar_images(
    input_dir: Path,
    output_dir: Path,
    threshold: int,
    hash_size: int = 8,
) -> list[HashDecision]:
    if threshold < 0:
        raise ValueError("--threshold must be 0 or greater")

    output_dir.mkdir(parents=True, exist_ok=True)

    previous_hash: np.ndarray | None = None
    decisions: list[HashDecision] = []

    for image_path in iter_images(input_dir):
        current_hash = compute_dhash(image_path, hash_size=hash_size)

        if previous_hash is None:
            distance = None
            should_copy = True
        else:
            distance = hamming_distance(previous_hash, current_hash)
            should_copy = distance > threshold

        if should_copy:
            shutil.copy2(image_path, output_dir / image_path.name)
            previous_hash = current_hash

        decisions.append(
            HashDecision(
                image_path=image_path,
                hash_value=current_hash,
                copied=should_copy,
                distance=distance,
            )
        )

    return decisions


def print_summary(decisions: list[HashDecision], threshold: int) -> None:
    total = len(decisions)
    kept = sum(1 for decision in decisions if decision.copied)
    skipped = total - kept

    print(f"Processed images : {total}")
    print(f"Kept images      : {kept}")
    print(f"Skipped images   : {skipped}")
    print(f"Hamming threshold: {threshold}")

    print("\nDetailed decisions:")
    for decision in decisions:
        if decision.distance is None:
            print(f"{decision.image_path.name}: kept (first image)")
            continue

        action = "kept" if decision.copied else "skipped"
        print(f"{decision.image_path.name}: {action} (distance={decision.distance})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter near-duplicate frames using perceptual hashing (dHash)."
    )
    parser.add_argument("--input", type=Path, required=True, help="Folder containing images")
    parser.add_argument("--output", type=Path, required=True, help="Folder to copy distinct images")
    parser.add_argument(
        "--threshold",
        type=int,
        default=6,
        help="Maximum Hamming distance to still consider two images similar",
    )
    parser.add_argument(
        "--hash-size",
        type=int,
        default=8,
        help="Hash size used to build the perceptual hash",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    decisions = filter_similar_images(
        input_dir=args.input,
        output_dir=args.output,
        threshold=args.threshold,
        hash_size=args.hash_size,
    )
    print_summary(decisions, threshold=args.threshold)


if __name__ == "__main__":
    main()

#threshold 6 means “skip the image if its perceptual hash differs by 6 bits or less from the last kept image.”