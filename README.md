# Tree Detection

A Python computer-vision project for detecting trees in images and videos. The repository includes two detection workflows:

- **Faster R-CNN** using PyTorch and COCO-format annotations, with optional video tracking.
- **YOLO** using Ultralytics and YOLO-format datasets for training and evaluation.

The project can extract frames from videos, prepare annotated datasets, train a detector, run image/video inference, track detections across video frames, and export detection results to CSV.

## Features

- Extract every *N*th frame from one video or a directory of videos.
- Prepare and merge COCO annotation exports.
- Split data into training, validation, and test sets.
- Train a Faster R-CNN ResNet-50 FPN model.
- Train and evaluate a YOLO tree detector.
- Run tree detection on individual images or videos.
- Track detected trees across video frames using IoU-based tracking.
- Save annotated images, annotated videos, and detection CSV files.

## Project Structure

```text
.
├── main.py                       # Main CLI for the Faster R-CNN workflow
├── train_yolo.py                 # Train the YOLO model
├── test_yolo.py                  # Evaluate YOLO and predict on test images
├── prepare_dataset.py            # Merge COCO exports and create train/val/test data
├── split_dataset.py              # Split a YOLO dataset into train/val/test sets
├── merge_dataset.py              # Merge two YOLO datasets
├── copy_images.py                # Copy images that have matching labels
├── requirements.txt              # Core Python dependencies
├── dataset/                      # Dataset exports, frames, and prepared COCO data
├── yolo_dataset/                 # YOLO images, labels, and data.yaml
├── models/                       # Faster R-CNN model weights
├── runs/                         # YOLO training and prediction outputs
├── utils/
│   ├── coco_dataset.py           # COCO dataset loading
│   ├── frame_extractor.py        # Video frame extraction
│   ├── inference.py              # Image/video inference
│   ├── modeling.py               # Faster R-CNN model creation/loading
│   ├── tracker.py                # IoU-based tracking
│   └── train.py                  # Faster R-CNN training loop
└── tree_detector_final.pt        # Saved YOLO model, if present
```

## Requirements

- Python 3.10 or newer is recommended.
- A CUDA-capable GPU is recommended for training, but CPU inference and training are supported.
- Input videos should be readable by OpenCV.

Install the dependencies listed in `requirements.txt`:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

The YOLO scripts additionally require Ultralytics:

```bash
pip install ultralytics
```

## Faster R-CNN Workflow

### 1. Extract frames from video

Extract one frame every 15 frames:

```bash
python main.py extract-frames \
  --video path/to/video.mp4 \
  --output dataset/extracted_frames/video_name \
  --every-n 15
```

To process every supported video in a directory:

```bash
python main.py extract-frames \
  --video dataset/videos \
  --output dataset/extracted_frames \
  --every-n 15
```

Use `--max-frames` to limit the number of extracted frames.

### 2. Annotate frames

Annotate the extracted frames with tree bounding boxes and export the annotations in COCO format. Place each export in a directory under:

```text
dataset/exports/<export-name>/instances_default.json
```

The corresponding extracted frames should be available under:

```text
dataset/extracted_frames/<export-name>/
```

### 3. Prepare the COCO dataset

Merge the COCO exports, remap image and annotation IDs, shuffle the data, create an 80/10/10 train/validation/test split, copy the images, and write `annotations.json` files:

```bash
python prepare_dataset.py
```

This creates:

```text
dataset/train/images/
dataset/train/annotations.json
dataset/val/images/
dataset/val/annotations.json
dataset/test/images/
dataset/test/annotations.json
```

### 4. Train Faster R-CNN

```bash
python main.py train \
  --train-images dataset/train/images \
  --train-annotations dataset/train/annotations.json \
  --val-images dataset/val/images \
  --val-annotations dataset/val/annotations.json \
  --epochs 10 \
  --batch-size 2 \
  --learning-rate 0.005 \
  --weights-out models/tree_detector.pth
```

The best validation-loss checkpoint is saved to the path supplied with `--weights-out`.

### 5. Run inference on an image

```bash
python main.py detect-image \
  --image path/to/image.jpg \
  --weights models/tree_detector.pth \
  --score-threshold 0.5 \
  --output outputs/detected_image.jpg
```

If `--output` is omitted, the result is saved under `outputs/`.

### 6. Run inference and tracking on a video

```bash
python main.py detect-video \
  --video path/to/video.mp4 \
  --weights models/tree_detector.pth \
  --score-threshold 0.5 \
  --output-video outputs/annotated_video.mp4 \
  --output-csv outputs/detections.csv
```

The command writes an annotated video and a CSV containing frame index, timestamp, track ID, confidence score, class label, and bounding-box coordinates.

## YOLO Workflow

The YOLO dataset configuration is stored in `yolo_dataset/data.yaml`. It defines one class:

```yaml
names:
  0: Tree
```

Before training, check the `path` value in `yolo_dataset/data.yaml` and update it for your local machine if necessary.

Train the YOLO model:

```bash
python train_yolo.py
```

The current training script uses the pretrained `yolov8n.pt` model, trains for 10 epochs at image size 640, and saves the final model as:

```text
tree_detector_final.pt
```

Evaluate the model and run predictions on the test images:

```bash
python test_yolo.py
```

Prediction images are saved under a directory in `runs/detect/`.

## Dataset Utilities

The repository also contains helper scripts for YOLO datasets:

- `copy_images.py` copies images that correspond to existing label files.
- `split_dataset.py` creates a 70/15/15 train/validation/test split.
- `merge_dataset.py` merges two YOLO datasets while prefixing filenames to avoid collisions.

These scripts currently contain local filesystem paths. Update those paths before running them on another machine.

## Notes

- The repository contains generated datasets, model weights, and run outputs. Consider using Git LFS or external storage for large binary files.
- `requirements.txt` contains the core PyTorch/OpenCV dependencies; install `ultralytics` separately for the YOLO scripts.
- `main.py` exposes a `sync-gps` command, but the corresponding `sync_detections_with_gps` implementation is not currently available in the checked-in code. The command should be implemented before use.
- The dataset and model paths in `yolo_dataset/data.yaml` and the helper scripts may need to be changed to match your local directory structure.

## License

No license file is currently included. Add a license before distributing or reusing this project publicly.
