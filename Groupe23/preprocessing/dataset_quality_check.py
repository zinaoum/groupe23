
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "dataset"
ANNOTATIONS = DATASET / "annotations"

splits = {
    "train": "captions_train2017.json",
    "validation": "captions_train2017.json",
    "test": "captions_val2017.json"
}

caption_counts = defaultdict(Counter)
all_ids = {}
hashes = defaultdict(list)

print("Starting Dataset Quality Check...\n")

# Load caption annotations
for split, filename in splits.items():
    with open(ANNOTATIONS / filename, encoding="utf-8") as f:
        data = json.load(f)

    image_ids = {
        int(Path(img["file_name"]).stem)
        for img in data["images"]
    }

    selected_ids = set()

    for image in (DATASET / split).iterdir():
        if image.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        image_id = int(image.stem)
        selected_ids.add(image_id)

    all_ids[split] = selected_ids

    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        if image_id in selected_ids:
            caption_counts[split][image_id] += 1

    missing = selected_ids - set(caption_counts[split])
    wrong_counts = {
        image_id: count
        for image_id, count in caption_counts[split].items()
        if count != 5
    }

    print(f"--- {split.upper()} ---")
    print("Images:", len(selected_ids))
    print("Images without captions:", len(missing))
    print("Images with caption count different from 5:", len(wrong_counts))

    distribution = Counter(caption_counts[split].values())
    print("Caption count distribution:", dict(sorted(distribution.items())))
    print()

# Check split overlap
print("--- SPLIT OVERLAP ---")
for a, b in [("train", "validation"), ("train", "test"),
             ("validation", "test")]:
    overlap = all_ids[a] & all_ids[b]
    print(f"{a} / {b} overlap:", len(overlap))

# Check exact duplicate images using SHA256
print("\n--- DUPLICATE IMAGE CHECK ---")

for split in splits:
    folder = DATASET / split

    for image in folder.iterdir():
        if image.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        hashes[digest].append(f"{split}/{image.name}")

duplicates = {
    digest: files
    for digest, files in hashes.items()
    if len(files) > 1
}

print("Unique image files:", len(hashes))
print("Duplicate groups:", len(duplicates))

for files in list(duplicates.values())[:10]:
    print("Duplicate:", files)

print("\nDataset Quality Check completed!")
