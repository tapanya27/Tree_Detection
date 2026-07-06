from pathlib import Path
import json
import random
import shutil

random.seed(42)

# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).parent

EXPORTS = ROOT / "dataset" / "exports"
FRAMES = ROOT / "dataset" / "extracted_frames"

TRAIN_DIR = ROOT / "dataset" / "train"
VAL_DIR = ROOT / "dataset" / "val"
TEST_DIR = ROOT / "dataset" / "test"

TRAIN_IMG = TRAIN_DIR / "images"
VAL_IMG = VAL_DIR / "images"
TEST_IMG = TEST_DIR / "images"

for folder in [TRAIN_IMG, VAL_IMG, TEST_IMG]:
    folder.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Read all COCO files
# --------------------------------------------------

all_images = []
all_annotations = []
categories = None

next_image_id = 1
next_annotation_id = 1

for export_folder in sorted(EXPORTS.iterdir()):

    json_file = export_folder / "instances_default.json"

    if not json_file.exists():
        continue

    with open(json_file, "r") as f:
        coco = json.load(f)

    if categories is None:
        categories = coco["categories"]

    id_mapping = {}

    for img in coco["images"]:

        old_id = img["id"]

        new_img = img.copy()

        new_img["id"] = next_image_id
        new_img["video_folder"] = export_folder.name

        id_mapping[old_id] = next_image_id

        all_images.append(new_img)

        next_image_id += 1

    for ann in coco["annotations"]:

        new_ann = ann.copy()

        new_ann["id"] = next_annotation_id
        new_ann["image_id"] = id_mapping[ann["image_id"]]

        all_annotations.append(new_ann)

        next_annotation_id += 1

print(f"Total Images       : {len(all_images)}")
print(f"Total Annotations  : {len(all_annotations)}")

# --------------------------------------------------
# Shuffle
# --------------------------------------------------

random.shuffle(all_images)

n = len(all_images)

train_end = int(0.8 * n)
val_end = int(0.9 * n)

train_images = all_images[:train_end]
val_images = all_images[train_end:val_end]
test_images = all_images[val_end:]


# --------------------------------------------------
# Create COCO Split
# --------------------------------------------------

def create_split(images, destination):

    image_ids = {img["id"] for img in images}

    anns = []

    for ann in all_annotations:
        if ann["image_id"] in image_ids:
            anns.append(ann)

    output_images = []

    for img in images:

        src = FRAMES / img["video_folder"] / img["file_name"]

        new_name = f"{img['video_folder']}_{img['file_name']}"

        dst = destination / new_name

        shutil.copy(src, dst)

        new_img = img.copy()

        new_img["file_name"] = new_name

        del new_img["video_folder"]

        output_images.append(new_img)

    return {
        "images": output_images,
        "annotations": anns,
        "categories": categories
    }


train_json = create_split(train_images, TRAIN_IMG)
val_json = create_split(val_images, VAL_IMG)
test_json = create_split(test_images, TEST_IMG)

with open(TRAIN_DIR / "annotations.json", "w") as f:
    json.dump(train_json, f)

with open(VAL_DIR / "annotations.json", "w") as f:
    json.dump(val_json, f)

with open(TEST_DIR / "annotations.json", "w") as f:
    json.dump(test_json, f)

print("\nDataset Prepared Successfully!")
print(f"Train Images : {len(train_json['images'])}")
print(f"Val Images   : {len(val_json['images'])}")
print(f"Test Images  : {len(test_json['images'])}")