
from ultralytics import YOLO
from pathlib import Path
import csv

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "results" / "statistics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DETECTIONS_FILE = OUTPUT_DIR / "yolo_detections.csv"
SUMMARY_FILE = OUTPUT_DIR / "yolo_image_summary.csv"

# Load YOLO model
model = YOLO("yolo11n.pt")

# Dataset splits
splits = ["train", "validation", "test"]

image_paths = []

for split in splits:
    folder = DATASET_DIR / split

    for image in folder.iterdir():
        if image.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            image_paths.append((image, split))

print("Total images:", len(image_paths))

# CSV headers
detection_fields = [
    "image_id", "file_name", "split",
    "class_id", "class_name", "confidence",
    "x1", "y1", "x2", "y2"
]

summary_fields = [
    "image_id", "file_name", "split", "objects_count"
]

# Start processing
with open(DETECTIONS_FILE, "w", newline="", encoding="utf-8") as det_file, \
     open(SUMMARY_FILE, "w", newline="", encoding="utf-8") as sum_file:

    det_writer = csv.DictWriter(det_file, fieldnames=detection_fields)
    sum_writer = csv.DictWriter(sum_file, fieldnames=summary_fields)

    det_writer.writeheader()
    sum_writer.writeheader()

    batch_size = 8

    for start in range(0, len(image_paths), batch_size):

        batch = image_paths[start:start + batch_size]

        results = model.predict(
            source=[str(item[0]) for item in batch],
            device="cpu",
            conf=0.25,
            imgsz=640,
            verbose=False
        )

        for (image_path, split), result in zip(batch, results):

            image_id = image_path.stem
            boxes = result.boxes

            sum_writer.writerow({
                "image_id": image_id,
                "file_name": image_path.name,
                "split": split,
                "objects_count": len(boxes)
            })

            for box in boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                det_writer.writerow({
                    "image_id": image_id,
                    "file_name": image_path.name,
                    "split": split,
                    "class_id": class_id,
                    "class_name": model.names[class_id],
                    "confidence": round(confidence, 4),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                })

        det_file.flush()
        sum_file.flush()

        processed = min(start + batch_size, len(image_paths))
        print(f"Processed: {processed}/{len(image_paths)}")

print("\nYOLO extraction completed!")
print("Detections:", DETECTIONS_FILE)
print("Image summary:", SUMMARY_FILE)
