
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
ANNOTATIONS = PROJECT_DIR / "dataset" / "annotations"
STATS_DIR = PROJECT_DIR / "results" / "statistics"
FIGURES_DIR = PROJECT_DIR / "results" / "figures"

STATS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load selected image IDs and their splits
with open(
    STATS_DIR / "split_manifest.csv",
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:
    rows = list(csv.DictReader(f))

selected = {
    int(row["image_id"]): row["split"].strip()
    for row in rows
}

# Load category names
with open(
    ANNOTATIONS / "instances_train2017.json",
    "r",
    encoding="utf-8"
) as f:
    train_data = json.load(f)

categories = {
    category["id"]: category["name"]
    for category in train_data["categories"]
}

category_counts = Counter()
category_images = defaultdict(set)

# Train and validation images come from train2017
# Test images come from val2017
annotation_files = [
    ("train", "instances_train2017.json"),
    ("test", "instances_val2017.json")
]

for source_split, filename in annotation_files:

    path = ANNOTATIONS / filename

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for annotation in data["annotations"]:

        image_id = annotation["image_id"]
        actual_split = selected.get(image_id)

        if actual_split is None:
            continue

        if source_split == "train":
            if actual_split not in ("train", "validation"):
                continue
        elif actual_split != "test":
            continue

        category_id = annotation["category_id"]
        category_name = categories[category_id]

        category_counts[category_name] += 1
        category_images[category_name].add(image_id)

# Save category statistics
output_csv = STATS_DIR / "selected_category_distribution.csv"

with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Category",
        "Object_Count",
        "Images_Containing_Category"
    ])

    for category, count in category_counts.most_common():
        writer.writerow([
            category,
            count,
            len(category_images[category])
        ])

# Plot top 20 categories
top_categories = category_counts.most_common(20)

names = [item[0] for item in top_categories]
counts = [item[1] for item in top_categories]

plt.figure(figsize=(12, 8))
plt.barh(names[::-1], counts[::-1])
plt.xlabel("Number of Object Instances")
plt.ylabel("Object Category")
plt.title("Top 20 Object Categories in Selected COCO Dataset")
plt.tight_layout()

figure_path = FIGURES_DIR / "selected_category_distribution.png"
plt.savefig(figure_path, dpi=300)
plt.close()

print("\n========== SELECTED CATEGORY ANALYSIS ==========")
print("Selected images:", len(selected))
print("Total object instances:", sum(category_counts.values()))
print("Number of categories:", len(category_counts))

print("\nTop 20 categories:")
for category, count in top_categories:
    print(f"{category}: {count}")

print("\nCSV saved to:", output_csv)
print("Chart saved to:", figure_path)
