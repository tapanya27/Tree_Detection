from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2 as T

from utils.modeling import TREE_CLASS_NAME, load_model
from utils.tracker import SimpleIoUTracker


def _prepare_image(image: Image.Image, device: torch.device):
    transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    return transform(image).to(device)


def _draw_prediction(frame, bbox, score: float, track_id: int | None = None):
    x1, y1, x2, y2 = [int(value) for value in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)
    label = f"{TREE_CLASS_NAME} {score:.2f}"
    if track_id is not None:
        label = f"ID {track_id} | {label}"
    cv2.putText(frame, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)


def _predict(model, image_tensor, score_threshold: float):
    with torch.no_grad():
        prediction = model([image_tensor])[0]

    detections = []
    for box, score, label in zip(prediction["boxes"], prediction["scores"], prediction["labels"]):
        score_value = float(score.item())
        if score_value < score_threshold:
            continue
        detections.append(
            {
                "bbox": [float(value) for value in box.tolist()],
                "score": score_value,
                "label": int(label.item()),
            }
        )
    return detections


def detect_image(
    image_path: Path,
    weights_path: Path,
    score_threshold: float = 0.5,
    output_path: Path | None = None,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights_path, device)

    image = Image.open(image_path).convert("RGB")
    detections = _predict(model, _prepare_image(image, device), score_threshold)

    frame = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    for detection in detections:
        _draw_prediction(frame, detection["bbox"], detection["score"])

    if output_path is None:
        output_path = Path("outputs") / f"{image_path.stem}_detected.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"Saved annotated image to {output_path}")
    return output_path


def detect_video(
    video_path: Path,
    weights_path: Path,
    score_threshold: float = 0.5,
    output_video: Path | None = None,
    output_csv: Path | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights_path, device)
    tracker = SimpleIoUTracker()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_video is None:
        output_video = Path("outputs") / f"{video_path.stem}_annotated.mp4"
    if output_csv is None:
        output_csv = Path("outputs") / f"{video_path.stem}_detections.csv"

    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    detection_rows = []
    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections = _predict(model, _prepare_image(image, device), score_threshold)
        tracked = tracker.update(detections)

        for track in tracked:
            if track.missed_frames > 0:
                continue
            _draw_prediction(frame, track.bbox, track.score, track_id=track.track_id)
            detection_rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp_sec": frame_index / fps,
                    "track_id": track.track_id,
                    "label": TREE_CLASS_NAME,
                    "score": track.score,
                    "x1": track.bbox[0],
                    "y1": track.bbox[1],
                    "x2": track.bbox[2],
                    "y2": track.bbox[3],
                }
            )

        writer.write(frame)
        frame_index += 1

    capture.release()
    writer.release()

    pd.DataFrame(detection_rows).to_csv(output_csv, index=False)
    print(f"Saved annotated video to {output_video}")
    print(f"Saved detections CSV to {output_csv}")
    return output_video, output_csv
