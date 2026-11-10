
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

STATS_DIR = BASE_DIR / "results" / "statistics"
FIGURES_DIR = BASE_DIR / "results" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = STATS_DIR / "objects_per_image.csv"

# Load object count data
df = pd.read_csv(INPUT_FILE)

# Classify scene complexity
def classify_scene(count):
    if count <= 3:
        return "Simple"
    elif count <= 10:
        return "Medium"
    else:
        return "Complex"

df["scene_complexity"] = df["object_count"].apply(classify_scene)

# Save detailed results
output_csv = STATS_DIR / "scene_complexity.csv"
df.to_csv(output_csv, index=False)

# Overall distribution
complexity_counts = (
    df["scene_complexity"]
    .value_counts()
    .reindex(["Simple", "Medium", "Complex"], fill_value=0)
)

print("\n========== SCENE COMPLEXITY ANALYSIS ==========")
print("\nNumber of images per complexity level:")
print(complexity_counts)

print("\nPercentage distribution:")
print((complexity_counts / len(df) * 100).round(2))

print("\nAverage objects per split:")
print(df.groupby("split")["object_count"].mean().round(2))

# Bar chart
plt.figure(figsize=(8, 5))
complexity_counts.plot(kind="bar", edgecolor="black")

plt.title("Scene Complexity Distribution")
plt.xlabel("Scene Complexity")
plt.ylabel("Number of Images")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

chart_path = FIGURES_DIR / "scene_complexity_distribution.png"
plt.savefig(chart_path, dpi=300)
plt.close()

# Histogram
plt.figure(figsize=(9, 5))
plt.hist(df["object_count"], bins=30, edgecolor="black")

plt.title("Number of Objects per Image")
plt.xlabel("Number of Objects")
plt.ylabel("Number of Images")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

histogram_path = FIGURES_DIR / "scene_complexity_histogram.png"
plt.savefig(histogram_path, dpi=300)
plt.close()

print("\nFiles saved successfully:")
print(output_csv)
print(chart_path)
print(histogram_path)
