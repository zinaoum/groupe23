
import json
import csv
import re
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent

ANNOTATIONS_DIR = BASE_DIR / "dataset" / "annotations"
MANIFEST_PATH = BASE_DIR / "results" / "statistics" / "split_manifest.csv"

OUTPUT_DIR = BASE_DIR / "dataset" / "processed" / "captions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load COCO caption annotations
def load_captions(filename):
    with open(ANNOTATIONS_DIR / filename, encoding="utf-8") as f:
        data = json.load(f)

    captions = defaultdict(list)

    for ann in data["annotations"]:
        captions[int(ann["image_id"])].append({
            "id": ann["id"],
            "caption": ann["caption"]
        })

    for image_id in captions:
        captions[image_id].sort(key=lambda x: x["id"])

    return captions


train_captions = load_captions("captions_train2017.json")
test_captions = load_captions("captions_val2017.json")

# Clean captions for retrieval
def clean_caption(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Read frozen split manifest
with open(MANIFEST_PATH, encoding="utf-8") as f:
    manifest = list(csv.DictReader(f))

all_raw_rows = []
processed_rows = []

for item in manifest:
    image_id = int(item["image_id"])
    split = item["split"]

    source = test_captions if split == "test" else train_captions
    original_captions = source.get(image_id, [])

    if len(original_captions) < 5:
        raise ValueError(
            f"Image {image_id} has only {len(original_captions)} captions"
        )

    # Preserve every original caption
    for index, caption_data in enumerate(original_captions, 1):
        all_raw_rows.append({
            "image_id": image_id,
            "file_name": item["file_name"],
            "split": split,
            "caption_index": index,
            "caption": caption_data["caption"]
        })

    # Select exactly five references deterministically
    for index, caption_data in enumerate(original_captions[:5], 1):
        raw = caption_data["caption"]

        processed_rows.append({
            "image_id": image_id,
            "file_name": item["file_name"],
            "split": split,
            "caption_index": index,
            "raw_caption": raw,
            "clean_caption": clean_caption(raw)
        })


def save_csv(filename, rows):
    if not rows:
        return

    with open(OUTPUT_DIR / filename, "w",
              newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# Save original captions without modification
save_csv("all_original_captions.csv", all_raw_rows)

# Save exactly five processed references per image
save_csv("processed_captions.csv", processed_rows)

# Separate each split to prevent accidental leakage
for split in ["train", "validation", "test"]:
    split_rows = [
        row for row in processed_rows
        if row["split"] == split
    ]

    filename = f"{split}_captions.csv"
    save_csv(filename, split_rows)

print("Caption preprocessing completed!")
print("Images:", len(manifest))
print("Original captions:", len(all_raw_rows))
print("Selected captions:", len(processed_rows))

for split in ["train", "validation", "test"]:
    count = sum(row["split"] == split for row in processed_rows)
    print(f"{split}: {count} captions")

print("\nFiles saved in:", OUTPUT_DIR)
print("Important: Only train_captions.csv is intended for the KB.")
print("Test captions must only be used as evaluation references.")
