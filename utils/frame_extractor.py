from __future__ import annotations

from pathlib import Path

import cv2


def extract_frames(
    video_path: Path,
    output_dir: Path,
    every_n: int = 10,#Save one frame after every 10 frames while reading the video.(Fixed Interval sampling)
    max_frames: int | None = None,
) -> int:
    if every_n <= 0:
        raise ValueError("--every-n must be greater than 0")

    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))#means open the video file located at video_path for reading frames
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    saved_count = 0 #How many images were saved
    frame_index = 0 #means the index of the current frame being processed in the video

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_index % every_n == 0:
            frame_name = output_dir / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(frame_name), frame)
            saved_count += 1

            if max_frames is not None and saved_count >= max_frames:
                break

        frame_index += 1

    capture.release()
    print(f"Saved {saved_count} frames to {output_dir}")
    return saved_count
