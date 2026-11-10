
import csv
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean

PROJECT_DIR = Path(__file__).resolve().parent.parent

MANIFEST = PROJECT_DIR / "results" / "statistics" / "split_manifest.csv"
ANNOTATIONS = PROJECT_DIR / "dataset" / "annotations"
OUTPUT_DIR = PROJECT_DIR / "results" / "statistics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read selected images
with open(MANIFEST, "r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

selected = {
    int(row["image_id"]): row["split"].strip()
    for row in rows
}

# Each selected image will have a list of captions
captions_by_image = defaultdict(list)

# Train and validation use train2017 captions
# Test uses val2017 captions
annotation_files = {
    "train": ANNOTATIONS / "captions_train2017.json",
    "validation": ANNOTATIONS / "captions_train2017.json",
    "test": ANNOTATIONS / "captions_val2017.json"
}

for split, json_path in annotation_files.items():

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for annotation in data["annotations"]:
        image_id = annotation["image_id"]

        if selected.get(image_id) == split:
            captions_by_image[image_id].append(
                annotation["caption"].strip()
            )

# Calculate statistics
caption_counts = []
word_counts = []
split_stats = defaultdict(list)
detailed_rows = []

for image_id, split in selected.items():

    image_captions = captions_by_image.get(image_id, [])

    caption_counts.append(len(image_captions))
    split_stats[split].append(len(image_captions))

    for index, caption in enumerate(image_captions, start=1):
        words = caption.split()
        word_counts.append(len(words))

        detailed_rows.append({
            "image_id": image_id,
            "split": split,
            "caption_number": index,
            "caption": caption,
            "word_count": len(words)
        })

# Save detailed captions
output_csv = OUTPUT_DIR / "caption_metadata.csv"

with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image_id", "split",
            "caption_number", "caption", "word_count"
        ]
    )
    writer.writeheader()
    writer.writerows(detailed_rows)

print("\n========== CAPTION ANALYSIS ==========")
print("Selected images:", len(selected))
print("Total captions:", len(detailed_rows))
print("Average captions per image:", round(mean(caption_counts), 2))
print("Average words per caption:", round(mean(word_counts), 2))
print("Minimum words:", min(word_counts))
print("Maximum words:", max(word_counts))

print("\n========== CAPTIONS PER SPLIT ==========")

for split in ("train", "validation", "test"):
    counts = split_stats[split]
    print(
        f"{split}: {sum(counts)} captions "
        f"for {len(counts)} images "
        f"(average: {mean(counts):.2f} per image)"
    )

print("\nCaption metadata saved to:", output_csv)
