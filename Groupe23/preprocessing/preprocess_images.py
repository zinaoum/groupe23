
from pathlib import Path
from PIL import Image, ImageOps
import csv

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = DATASET_DIR / "processed"
STATS_DIR = BASE_DIR / "results" / "statistics"

STATS_DIR.mkdir(parents=True, exist_ok=True)

splits = ["train", "validation", "test"]

extensions = {".jpg", ".jpeg", ".png"}

report = []

for split in splits:
    source_folder = DATASET_DIR / split
    target_folder = OUTPUT_DIR / split

    target_folder.mkdir(parents=True, exist_ok=True)

    images = [
        p for p in source_folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]

    print(f"\nProcessing {split}: {len(images)} images")

    for index, image_path in enumerate(images, 1):
        try:
            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")

                width, height = img.size

                output_path = target_folder / (
                    image_path.stem + ".jpg"
                )

                img.save(
                    output_path,
                    format="JPEG",
                    quality=95,
                    optimize=True
                )

                report.append({
                    "file_name": image_path.name,
                    "split": split,
                    "width": width,
                    "height": height,
                    "mode": img.mode,
                    "status": "success"
                })

        except Exception as error:
            report.append({
                "file_name": image_path.name,
                "split": split,
                "width": "",
                "height": "",
                "mode": "",
                "status": f"error: {error}"
            })

        if index % 500 == 0:
            print(f"Processed: {index}/{len(images)}")

report_path = STATS_DIR / "image_preprocessing_report.csv"

with open(report_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=[
        "file_name", "split", "width",
        "height", "mode", "status"
    ])
    writer.writeheader()
    writer.writerows(report)

success = sum(r["status"] == "success" for r in report)
errors = len(report) - success

print("\nImage preprocessing completed!")
print("Successful:", success)
print("Errors:", errors)
print("Report:", report_path)
