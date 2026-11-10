
import csv
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = BASE_DIR / "dataset" / "processed"
CAPTION_DIR = BASE_DIR / "dataset" / "processed" / "captions"

OUTPUT_DIR = BASE_DIR / "results" / "embeddings"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"
BATCH_SIZE = 16

MODEL_NAME = "openai/clip-vit-base-patch32"

print("Loading CLIP...")

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

print("CLIP loaded successfully!")


# Generate image embeddings
def generate_image_embeddings(split):

    folder = IMAGE_DIR / split

    images = sorted([
        p for p in folder.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    embeddings = []
    metadata = []

    print(f"\nProcessing {split}: {len(images)} images")

    for start in range(0, len(images), BATCH_SIZE):

        batch_paths = images[start:start + BATCH_SIZE]

        batch_images = []

        for path in batch_paths:
            with Image.open(path) as img:
                batch_images.append(img.convert("RGB"))

        inputs = processor(
            images=batch_images,
            return_tensors="pt"
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

            if not isinstance(outputs, torch.Tensor):
                outputs = outputs.pooler_output

            outputs = F.normalize(outputs, p=2, dim=-1)

        embeddings.append(outputs.cpu().numpy())

        for path in batch_paths:
            metadata.append({
                "image_id": path.stem,
                "file_name": path.name,
                "split": split
            })

        processed = min(start + BATCH_SIZE, len(images))
        print(f"{split}: {processed}/{len(images)}")

    embeddings = np.concatenate(embeddings, axis=0)

    np.save(
        OUTPUT_DIR / f"{split}_image_embeddings.npy",
        embeddings
    )

    with open(
        OUTPUT_DIR / f"{split}_image_metadata.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_id", "file_name", "split"]
        )
        writer.writeheader()
        writer.writerows(metadata)

    print(f"{split} image embeddings shape:", embeddings.shape)


# Generate train text embeddings only
def generate_train_text_embeddings():

    caption_file = CAPTION_DIR / "train_captions.csv"

    with open(caption_file, encoding="utf-8-sig") as f:
        captions = list(csv.DictReader(f))

    embeddings = []
    metadata = []

    print(f"\nProcessing train captions: {len(captions)}")

    for start in range(0, len(captions), 32):

        batch = captions[start:start + 32]

        texts = [row["clean_caption"] for row in batch]

        inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.get_text_features(**inputs)

            if not isinstance(outputs, torch.Tensor):
                outputs = outputs.pooler_output

            outputs = F.normalize(outputs, p=2, dim=-1)

        embeddings.append(outputs.cpu().numpy())

        for row in batch:
            metadata.append({
                "image_id": row["image_id"],
                "file_name": row["file_name"],
                "caption_index": row["caption_index"],
                "caption": row["raw_caption"],
                "clean_caption": row["clean_caption"]
            })

        processed = min(start + 32, len(captions))
        print(f"Train captions: {processed}/{len(captions)}")

    embeddings = np.concatenate(embeddings, axis=0)

    np.save(
        OUTPUT_DIR / "train_text_embeddings.npy",
        embeddings
    )

    with open(
        OUTPUT_DIR / "train_text_metadata.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "file_name",
                "caption_index",
                "caption",
                "clean_caption"
            ]
        )
        writer.writeheader()
        writer.writerows(metadata)

    print("Train text embeddings shape:", embeddings.shape)


# Run all image splits
for split in ["train", "validation", "test"]:
    generate_image_embeddings(split)

# Only train captions enter the retrieval knowledge base
generate_train_text_embeddings()

print("\nAll embeddings generated successfully!")
print("Output folder:", OUTPUT_DIR)
