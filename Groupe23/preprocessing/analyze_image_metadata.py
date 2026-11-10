
import csv
from pathlib import Path
from PIL import Image
from collections import Counter
from statistics import mean

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = PROJECT_DIR / "results" / "statistics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

metadata = []

for split in ("train", "validation", "test"):
    folder = DATASET_DIR / split

    for path in folder.glob("*.jpg"):
        try:
            with Image.open(path) as img:
                width, height = img.size

            size_kb = path.stat().st_size / 1024
            ratio = width / height

            metadata.append({
                "file_name": path.name,
                "split": split,
                "width": width,
                "height": height,
                "aspect_ratio": round(ratio, 3),
                "size_kb": round(size_kb, 2)
            })

        except Exception as error:
            print("Error:", path.name, error)

# Save detailed metadata
csv_path = OUTPUT_DIR / "image_metadata.csv"

with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
    writer.writeheader()
    writer.writerows(metadata)

# Calculate statistics
widths = [item["width"] for item in metadata]
heights = [item["height"] for item in metadata]
sizes = [item["size_kb"] for item in metadata]

dimensions = Counter(
    (item["width"], item["height"]) for item in metadata
)

print("\n========== IMAGE METADATA ==========")
print("Total images:", len(metadata))
print("Average width:", round(mean(widths), 2))
print("Average height:", round(mean(heights), 2))
print("Average size (KB):", round(mean(sizes), 2))
print("Minimum size (KB):", round(min(sizes), 2))
print("Maximum size (KB):", round(max(sizes), 2))

print("\nMost common dimensions:")
for dimension, count in dimensions.most_common(10):
    print(f"{dimension[0]} x {dimension[1]} : {count} images")

print("\nMetadata saved to:", csv_path)
