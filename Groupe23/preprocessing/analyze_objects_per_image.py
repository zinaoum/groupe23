
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

ANNOTATIONS_DIR = BASE_DIR / "dataset" / "annotations"
RESULTS_DIR = BASE_DIR / "results"
STATS_DIR = RESULTS_DIR / "statistics"
FIGURES_DIR = RESULTS_DIR / "figures"

STATS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = STATS_DIR / "split_manifest.csv"

# Load selected images
manifest = pd.read_csv(MANIFEST_PATH)

# Count objects for each image
object_counts = Counter()

for split, annotation_file in [
    ("train", "instances_train2017.json"),
    ("validation", "instances_train2017.json"),
    ("test", "instances_val2017.json")
]:
    print(f"Processing {split} annotations...")

    with open(ANNOTATIONS_DIR / annotation_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    selected_ids = set(
        manifest.loc[manifest["split"] == split, "image_id"].astype(int)
    )

    for annotation in coco["annotations"]:
        image_id = annotation["image_id"]

        if image_id in selected_ids:
            object_counts[image_id] += 1

# Create results for all selected images
rows = []

for _, image in manifest.iterrows():
    image_id = int(image["image_id"])

    rows.append({
        "image_id": image_id,
        "file_name": image["file_name"],
        "split": image["split"],
        "object_count": object_counts.get(image_id, 0)
    })

df = pd.DataFrame(rows)

# Save CSV
output_csv = STATS_DIR / "objects_per_image.csv"
df.to_csv(output_csv, index=False)

# Statistics
print("\nObject count statistics:")
print(df.groupby("split")["object_count"].describe())

print("\nOverall average:",
      round(df["object_count"].mean(), 2))

# Histogram
plt.figure(figsize=(10, 6))
plt.hist(df["object_count"], bins=30, edgecolor="black")
plt.title("Distribution of Objects per Image")
plt.xlabel("Number of Objects")
plt.ylabel("Number of Images")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

output_plot = FIGURES_DIR / "objects_per_image_histogram.png"
plt.savefig(output_plot, dpi=300)
plt.close()

print("\nResults saved successfully:")
print(output_csv)
print(output_plot)
