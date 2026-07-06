from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T


class CocoTreeDataset(Dataset):
    def __init__(self, images_dir: Path, annotations_path: Path, train: bool = True):
        self.images_dir = images_dir
        self.annotations_path = annotations_path
        self.train = train
        self.transforms = self._build_transforms(train=train)

        with annotations_path.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)

        self.images = {image["id"]: image for image in coco["images"]}
        self.image_ids = sorted(self.images.keys())
        self.annotations_by_image = {image_id: [] for image_id in self.image_ids}

        for annotation in coco["annotations"]:
            self.annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        image_path = self.images_dir / image_info["file_name"]

        image = Image.open(image_path).convert("RGB")
        annotations = self.annotations_by_image.get(image_id, [])

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for annotation in annotations:
            x, y, width, height = annotation["bbox"]
            boxes.append([x, y, x + width, y + height])
            labels.append(annotation["category_id"])
            areas.append(annotation.get("area", width * height))
            iscrowd.append(annotation.get("iscrowd", 0))

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            area_tensor = torch.tensor(areas, dtype=torch.float32)
            iscrowd_tensor = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": area_tensor,
            "iscrowd": iscrowd_tensor,
        }

        image_tensor = self.transforms(image)
        return image_tensor, target

    @staticmethod
    def _build_transforms(train: bool):
        transforms = [T.ToImage(), T.ToDtype(torch.float32, scale=True)]
        if train:
            transforms.append(T.RandomHorizontalFlip(p=0.5))
        return T.Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))
