
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
STATS_DIR = BASE_DIR / "results" / "statistics"
FIGURES_DIR = BASE_DIR / "results" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load YOLO results
detections = pd.read_csv(STATS_DIR / "yolo_detections.csv")
images = pd.read_csv(STATS_DIR / "yolo_image_summary.csv")

# 1. Object category distribution
category_counts = (
    detections["class_name"]
    .value_counts()
    .reset_index()
)

category_counts.columns = ["category", "count"]

category_counts.to_csv(
    STATS_DIR / "yolo_category_distribution.csv",
    index=False
)

# Top 50 categories chart
top50 = category_counts.head(50)

plt.figure(figsize=(14, 10))
plt.barh(top50["category"][::-1], top50["count"][::-1])
plt.xlabel("Number of detections")
plt.ylabel("Object category")
plt.title("Top 50 YOLO Object Categories")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "yolo_top50_categories.png")
plt.close()

# 2. Objects per image statistics
split_stats = images.groupby("split")["objects_count"].agg(
    ["count", "mean", "std", "min", "median", "max"]
).round(2)

print("\nObjects per image by split:")
print(split_stats)

split_stats.to_csv(STATS_DIR / "yolo_objects_per_split.csv")

# 3. Scene complexity
def classify_scene(count):
    if count <= 3:
        return "Simple"
    elif count <= 10:
        return "Medium"
    else:
        return "Complex"

images["scene_complexity"] = images["objects_count"].apply(
    classify_scene
)

complexity_counts = (
    images["scene_complexity"]
    .value_counts()
    .reindex(["Simple", "Medium", "Complex"], fill_value=0)
)

complexity_table = pd.DataFrame({
    "scene_complexity": complexity_counts.index,
    "count": complexity_counts.values,
    "percentage": (
        complexity_counts.values / len(images) * 100
    ).round(2)
})

complexity_table.to_csv(
    STATS_DIR / "yolo_scene_complexity.csv",
    index=False
)

# Scene complexity chart
plt.figure(figsize=(8, 5))
plt.bar(complexity_table["scene_complexity"],
        complexity_table["count"])
plt.xlabel("Scene complexity")
plt.ylabel("Number of images")
plt.title("YOLO Scene Complexity Distribution")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "yolo_scene_complexity.png")
plt.close()

# 4. Object count histogram
plt.figure(figsize=(9, 5))
plt.hist(images["objects_count"], bins=30, edgecolor="black")
plt.xlabel("Number of detected objects")
plt.ylabel("Number of images")
plt.title("YOLO Objects per Image")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "yolo_objects_per_image.png")
plt.close()

print("\nTop 15 categories:")
print(category_counts.head(15).to_string(index=False))

print("\nScene complexity:")
print(complexity_table.to_string(index=False))

print("\nAnalysis completed successfully!")
