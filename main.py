from __future__ import annotations

import argparse
from pathlib import Path

from utils.frame_extractor import extract_frames
from utils.inference import detect_image, detect_video
from utils.train import train_model

#Video -> extracted frames -> manual annotation in COCO format -> dataset split preparation -> model training -> saved weights -> image/video detection -> optional tracking output.

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tree detection project CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract-frames", help="Extract frames from a video")
    extract_parser.add_argument("--video", type=Path, required=True)
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.add_argument("--every-n", type=int, default=15)
    extract_parser.add_argument("--max-frames", type=int, default=None)

    train_parser = subparsers.add_parser("train", help="Train Faster R-CNN on COCO annotations")
    train_parser.add_argument("--train-images", type=Path, required=True)
    train_parser.add_argument("--train-annotations", type=Path, required=True)
    train_parser.add_argument("--val-images", type=Path, required=True)
    train_parser.add_argument("--val-annotations", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--learning-rate", type=float, default=0.005)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--weights-out", type=Path, default=Path("models/tree_detector.pth"))

    image_parser = subparsers.add_parser("detect-image", help="Run inference on one image")
    image_parser.add_argument("--image", type=Path, required=True)
    image_parser.add_argument("--weights", type=Path, required=True)
    image_parser.add_argument("--score-threshold", type=float, default=0.5)
    image_parser.add_argument("--output", type=Path, default=None)

    video_parser = subparsers.add_parser("detect-video", help="Run inference and tracking on one video")
    video_parser.add_argument("--video", type=Path, required=True)
    video_parser.add_argument("--weights", type=Path, required=True)
    video_parser.add_argument("--score-threshold", type=float, default=0.5)
    video_parser.add_argument("--output-video", type=Path, default=None)
    video_parser.add_argument("--output-csv", type=Path, default=None)

    gps_parser = subparsers.add_parser("sync-gps", help="Join detections with GPS data")
    gps_parser.add_argument("--detections", type=Path, required=True)
    gps_parser.add_argument("--gps", type=Path, required=True)
    gps_parser.add_argument("--fps", type=float, default=None)
    gps_parser.add_argument("--output", type=Path, default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "extract-frames":

        if args.video.is_dir():
            for video in args.video.glob("*"):
                if video.suffix.lower() not in [".mp4", ".mov"]:
                    continue

                extract_frames(
                video_path=video,
                output_dir=args.output / video.stem,
                every_n=args.every_n,
                max_frames=args.max_frames,
                )
        else:
            extract_frames(
            video_path=args.video,
            output_dir=args.output,
            every_n=args.every_n,
            max_frames=args.max_frames,
            )
    elif args.command == "train":
        train_model(
            train_images=args.train_images,
            train_annotations=args.train_annotations,
            val_images=args.val_images,
            val_annotations=args.val_annotations,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_workers=args.num_workers,
            weights_out=args.weights_out,
        )
    elif args.command == "detect-image":
        detect_image(
            image_path=args.image,
            weights_path=args.weights,
            score_threshold=args.score_threshold,
            output_path=args.output,
        )
    elif args.command == "detect-video":
        detect_video(
            video_path=args.video,
            weights_path=args.weights,
            score_threshold=args.score_threshold,
            output_video=args.output_video,
            output_csv=args.output_csv,
        )
    elif args.command == "sync-gps":
        sync_detections_with_gps(
            detections_csv=args.detections,
            gps_csv=args.gps,
            fps=args.fps,
            output_csv=args.output,
        )


if __name__ == "__main__":
    main()

   # ResNet50 = extracts image features
#FPN = helps with different object sizes   it creates multi-scale feature maps(small trees and big trees)
#RPN = suggests where trees might be    this part scans the feature maps
#RCNN head = confirms if it is a tree and adjusts the box

#Optimizer: SGD
#Loss: sum of the internal Faster R-CNN losses returned by the model


