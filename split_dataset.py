from pathlib import Path
import random
import shutil

# Dataset paths
images_dir = Path(r"C:\Users\tanay\Downloads\yolo_2468\data\images\train")
labels_dir = Path(r"C:\Users\tanay\Downloads\yolo_2468\labels\train")

# Create destination folders
for split in ["train", "val", "test"]:
    (Path(r"C:\Users\tanay\Downloads\yolo_2468\data\images") / split).mkdir(parents=True, exist_ok=True)
    (Path(r"C:\Users\tanay\Downloads\yolo_2468\labels") / split).mkdir(parents=True, exist_ok=True)

# Read image files
images = list(images_dir.glob("*.jpg"))
random.seed(42)
random.shuffle(images)

n = len(images)
train_end = int(0.70 * n)
val_end = int(0.85 * n)

train_imgs = images[:train_end]
val_imgs = images[train_end:val_end]
test_imgs = images[val_end:]

splits = {
    "train": train_imgs,
    "val": val_imgs,
    "test": test_imgs,
}

for split, img_list in splits.items():
    for img in img_list:
        label = labels_dir / (img.stem + ".txt")

        shutil.move(
            str(img),
            str(Path(r"C:\Users\tanay\Downloads\yolo_2468\data\images") / split / img.name),
        )

        if label.exists():
            shutil.move(
                str(label),
                str(Path(r"C:\Users\tanay\Downloads\yolo_2468\labels") / split / label.name),
            )

print("✅ Dataset split completed!")

print(f"Train: {len(train_imgs)}")
print(f"Validation: {len(val_imgs)}")
print(f"Test: {len(test_imgs)}")