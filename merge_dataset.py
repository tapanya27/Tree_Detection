from pathlib import Path
import shutil

# ==============================
# CHANGE THESE PATHS
# ==============================

dataset1 = Path(r"C:\Users\tanay\Downloads\yolo_2471")
dataset2 = Path(r"C:\Users\tanay\Downloads\yolo_2468")

merged = Path(r"C:\Users\tanay\Downloads\yolo_dataset")

# ==============================
# Create destination folders
# ==============================

for split in ["train", "val", "test"]:
    (merged / "data" / "images" / split).mkdir(parents=True, exist_ok=True)
    (merged / "labels" / split).mkdir(parents=True, exist_ok=True)

# ==============================
# Function to copy one dataset
# ==============================

def copy_dataset(source_dataset, prefix):
    for split in ["train", "val", "test"]:

        image_folder = source_dataset / "data" / "images" / split
        label_folder = source_dataset / "labels" / split

        if not image_folder.exists():
            continue

        for image in image_folder.glob("*.jpg"):

            new_image_name = f"{prefix}_{image.name}"
            new_label_name = f"{prefix}_{image.stem}.txt"

            shutil.copy2(
                image,
                merged / "data" / "images" / split / new_image_name
            )

            label = label_folder / (image.stem + ".txt")

            if label.exists():
                shutil.copy2(
                    label,
                    merged / "labels" / split / new_label_name
                )

# ==============================
# Copy both datasets
# ==============================

copy_dataset(dataset1, "2471")
copy_dataset(dataset2, "2468")

print("✅ Datasets merged successfully!")