from pathlib import Path
import shutil

# Folder containing ALL original extracted frames
source_images = Path(r"C:\Users\tanay\TreeDetection\dataset\extracted_frames\IMG_2468")

# Folder containing YOLO label files exported from CVAT
labels_folder = Path(r"C:\Users\tanay\Downloads\yolo_2468\labels\train")

# Destination folder for YOLO images
destination_images = Path(r"C:\Users\tanay\Downloads\yolo_2468\data\images\train")

destination_images.mkdir(parents=True, exist_ok=True)

copied = 0
missing = []

for label_file in labels_folder.glob("*.txt"):
    image_name = label_file.stem + ".jpg"
    source_image = source_images / image_name

    if source_image.exists():
        shutil.copy2(source_image, destination_images / image_name)
        copied += 1
    else:
        missing.append(image_name)

print(f"✅ Copied {copied} images.")

if missing:
    print("\n⚠️ Missing images:")
    for image in missing:
        print(image)
else:
    print("🎉 All annotated images were found and copied successfully!")