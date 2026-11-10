
import numpy as np
import pandas as pd
import faiss
import torch
from pathlib import Path
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor

# Paths
BASE = Path(__file__).resolve().parent.parent
EMBEDDINGS = BASE / "results" / "embeddings"
KB = BASE / "results" / "knowledge_base"
DATASET = BASE / "dataset" / "processed" / "validation"

# Load CLIP
print("Loading CLIP model...")

model_name = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)

model.eval()

device = torch.device("cpu")
model.to(device)

# Load FAISS indexes
print("Loading FAISS indexes...")

image_index = faiss.read_index(str(KB / "image_index.index"))
text_index = faiss.read_index(str(KB / "text_index.index"))

# Load metadata
image_metadata = pd.read_csv(KB / "image_index_metadata.csv")
text_metadata = pd.read_csv(KB / "text_index_metadata.csv")

# Select one validation image
validation_metadata = pd.read_csv(
    EMBEDDINGS / "validation_image_metadata.csv"
)

sample = validation_metadata.iloc[0]

image_id = int(sample["image_id"])
file_name = sample["file_name"]

image_path = DATASET / file_name

print("\nQuery image:", file_name)
print("Image ID:", image_id)

# Load and preprocess image
image = Image.open(image_path)
image = ImageOps.exif_transpose(image).convert("RGB")

inputs = processor(images=image, return_tensors="pt")

# Generate image embedding
with torch.no_grad():
    outputs = model.vision_model(
        pixel_values=inputs["pixel_values"]
    )

    features = model.visual_projection(
        outputs.pooler_output
    )

query_vector = features.cpu().numpy().astype("float32")

# Normalize vector
faiss.normalize_L2(query_vector)

# Search similar training images
image_scores, image_indices = image_index.search(query_vector, 5)

print("\n===== TOP 5 SIMILAR IMAGES =====")

for rank, (score, idx) in enumerate(
    zip(image_scores[0], image_indices[0]), start=1
):
    result = image_metadata.iloc[idx]

    print(f"\n{rank}. Image ID: {result['image_id']}")
    print("File:", result["file_name"])
    print("Similarity:", round(float(score), 4))

# Search similar training captions
text_scores, text_indices = text_index.search(query_vector, 5)

print("\n===== TOP 5 SIMILAR CAPTIONS =====")

for rank, (score, idx) in enumerate(
    zip(text_scores[0], text_indices[0]), start=1
):
    result = text_metadata.iloc[idx]

    print(f"\n{rank}. Image ID: {result['image_id']}")
    print("Caption:", result["caption"])
    print("Similarity:", round(float(score), 4))

print("\nRETRIEVAL TEST COMPLETED SUCCESSFULLY!")
