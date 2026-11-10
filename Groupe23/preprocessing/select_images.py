
import json
import random
import os
import pandas as pd

# Project configuration
random_seed = 42

train_captions_path = "dataset/annotations/captions_train2017.json"
val_captions_path = "dataset/annotations/captions_val2017.json"

# Load COCO caption annotations
with open(train_captions_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(val_captions_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)

# Sort images to ensure reproducibility
train_images = sorted(train_data["images"], key=lambda x: x["id"])
test_images = sorted(test_data["images"], key=lambda x: x["id"])

# Select 10,100 distinct images from Train2017
rng = random.Random(random_seed)
selected_train = rng.sample(train_images, 10100)

# Split into Train and Validation
train_set = selected_train[:10000]
validation_set = selected_train[10000:]

# Select 1,000 images from Val2017 for Test
test_set = rng.sample(test_images, 1000)

# Build the final manifest
manifest = []

for split_name, images in [
    ("train", train_set),
    ("validation", validation_set),
    ("test", test_set)
]:
    for img in images:
        manifest.append({
            "image_id": img["id"],
            "file_name": img["file_name"],
            "coco_url": img["coco_url"],
            "split": split_name
        })

df = pd.DataFrame(manifest)

# Verify split sizes
print("Dataset Split Results:")
print(df["split"].value_counts())

# Verify no image ID overlap
assert df["image_id"].is_unique, "Duplicate image IDs found!"

# Save the manifest
os.makedirs("results/statistics", exist_ok=True)

df.to_csv(
    "results/statistics/split_manifest.csv",
    index=False
)

print("\nTotal selected images:", len(df))
print("Unique image IDs:", df["image_id"].nunique())
print("\nManifest saved successfully!")
