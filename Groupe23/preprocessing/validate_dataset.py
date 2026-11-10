
import csv
from pathlib import Path
from collections import Counter
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_DIR / "results" / "statistics" / "split_manifest.csv"
DATASET_DIR = PROJECT_DIR / "dataset"
REPORT = PROJECT_DIR / "results" / "statistics" / "validation_report.csv"

expected = Counter()
missing = []
corrupted = []
extra = []
valid = 0

# Read expected images from manifest
with open(MANIFEST, "r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

print("Checking dataset images...")
print("Expected images:", len(rows))

for row in rows:
    split = row["split"].strip()
    filename = row["file_name"].strip()
    expected[(split, filename)] += 1

    path = DATASET_DIR / split / filename

    if not path.exists():
        missing.append((split, filename))
        continue

    try:
        with Image.open(path) as img:
            img.verify()
        valid += 1
    except Exception:
        corrupted.append((split, filename))

# Find files that are not expected in each split
for split in ("train", "validation", "test"):
    folder = DATASET_DIR / split

    for path in folder.glob("*.jpg"):
        if (split, path.name) not in expected:
            extra.append((split, path.name))

# Save detailed report
with open(REPORT, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Split", "File", "Status"])

    for split, name in missing:
        writer.writerow([split, name, "Missing"])

    for split, name in corrupted:
        writer.writerow([split, name, "Corrupted"])

    for split, name in extra:
        writer.writerow([split, name, "Unexpected"])

print("\n========== VALIDATION RESULTS ==========")
print("Expected:", len(rows))
print("Valid:", valid)
print("Missing:", len(missing))
print("Corrupted:", len(corrupted))
print("Unexpected:", len(extra))

print("\n========== SPLIT COUNTS ==========")

for split in ("train", "validation", "test"):
    expected_count = sum(
        1 for row in rows if row["split"].strip() == split
    )
    actual_count = sum(
        1 for path in (DATASET_DIR / split).glob("*.jpg")
        if path.is_file()
    )

    print(f"{split}: {actual_count} / {expected_count}")

print("\nReport saved to:", REPORT)
