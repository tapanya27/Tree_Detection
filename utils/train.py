from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.coco_dataset import CocoTreeDataset, collate_fn
from utils.modeling import build_model


def _evaluate_loss(model, dataloader, device: torch.device) -> float:
    model.train()
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            losses = model(images, targets)
            total_loss += sum(loss for loss in losses.values()).item()

    return total_loss / max(len(dataloader), 1)


def train_model(
    train_images: Path,
    train_annotations: Path,
    val_images: Path,
    val_annotations: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
    weights_out: Path,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_dataset = CocoTreeDataset(train_images, train_annotations, train=True)
    val_dataset = CocoTreeDataset(val_images, val_annotations, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    model = build_model()
    model.to(device)

    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_val_loss = float("inf")
    weights_out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for images, targets in progress:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

            loss_dict = model(images, targets)
            loss = sum(component for component in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss / max(len(train_loader), 1)
        val_loss = _evaluate_loss(model, val_loader, device)
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), weights_out)
            print(f"Saved best model to {weights_out}")

    return weights_out
