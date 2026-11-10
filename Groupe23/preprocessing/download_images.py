
import csv
import time
import requests
from io import BytesIO
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_DIR / "results" / "statistics" / "split_manifest.csv"
DATASET_DIR = PROJECT_DIR / "dataset"
STATS_DIR = PROJECT_DIR / "results" / "statistics"

# Download settings
MAX_WORKERS = 6
MAX_RETRIES = 3
TIMEOUT = (15, 45)

STATS_DIR.mkdir(parents=True, exist_ok=True)


def get_download_url(coco_url):
    """Convert the original COCO URL to the working Amazon S3 URL."""
    from urllib.parse import urlparse

    parsed = urlparse(coco_url)
    image_path = parsed.path.lstrip("/")

    return f"https://s3.amazonaws.com/images.cocodataset.org/{image_path}"


def is_valid_image(path):
    """Check whether an existing image is a valid image."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def download_one(row):
    split = row["split"].strip()
    filename = row["file_name"].strip()
    original_url = row["coco_url"].strip()

    if split not in ("train", "validation", "test"):
        return "failed", filename, "Unknown split: " + split

    output_dir = DATASET_DIR / split
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    # Skip an already downloaded and valid image
    if output_path.exists() and is_valid_image(output_path):
        return "skipped", filename, ""

    # Remove an invalid existing file
    if output_path.exists():
        output_path.unlink()

    url = get_download_url(original_url)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=TIMEOUT)
            response.raise_for_status()

            # Verify the downloaded content before saving
            with Image.open(BytesIO(response.content)) as img:
                img.verify()

            # Save only after successful validation
            output_path.write_bytes(response.content)

            return "downloaded", filename, ""

        except Exception as error:
            if attempt == MAX_RETRIES:
                return "failed", filename, str(error)

            time.sleep(attempt * 2)

    return "failed", filename, "Unknown error"


def main():
    if not MANIFEST_PATH.exists():
        print("ERROR: split_manifest.csv was not found!")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    total = len(rows)

    print("=" * 50)
    print("COCO IMAGE DOWNLOAD")
    print("=" * 50)
    print("Total images:", total)
    print("Workers:", MAX_WORKERS)
    print("Source: Amazon S3")
    print("Starting download...\n")

    downloaded = 0
    skipped = 0
    failed = 0
    failed_rows = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_one, row) for row in rows]

        for index, future in enumerate(as_completed(futures), start=1):
            status, filename, error = future.result()

            if status == "downloaded":
                downloaded += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                failed_rows.append({
                    "file_name": filename,
                    "error": error
                })

            if index % 100 == 0 or index == total:
                print(
                    f"Progress: {index}/{total} | "
                    f"Downloaded: {downloaded} | "
                    f"Skipped: {skipped} | "
                    f"Failed: {failed}"
                )

    # Save failed downloads for later retry
    failed_path = STATS_DIR / "failed_downloads.csv"

    with open(failed_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["file_name", "error"]
        )
        writer.writeheader()
        writer.writerows(failed_rows)

    # Save download summary
    summary_path = STATS_DIR / "download_summary.csv"

    with open(summary_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Count"])
        writer.writerow(["Total", total])
        writer.writerow(["Downloaded", downloaded])
        writer.writerow(["Skipped", skipped])
        writer.writerow(["Failed", failed])

    print("\n" + "=" * 50)
    print("DOWNLOAD FINISHED")
    print("=" * 50)
    print("Total:", total)
    print("Downloaded:", downloaded)
    print("Skipped:", skipped)
    print("Failed:", failed)
    print("\nSummary saved to:", summary_path)
    print("Failed list saved to:", failed_path)


if __name__ == "__main__":
    main()
