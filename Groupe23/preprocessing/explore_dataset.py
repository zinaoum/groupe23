import json
import os

annotation_path = "dataset/annotations"

files = [
    "captions_train2017.json",
    "captions_val2017.json",
    "instances_train2017.json",
    "instances_val2017.json"
]

for file in files:
    path = os.path.join(annotation_path, file)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\nFile:", file)
    print("Images:", len(data.get("images", [])))
    print("Annotations:", len(data.get("annotations", [])))
    print("Categories:", len(data.get("categories", [])))