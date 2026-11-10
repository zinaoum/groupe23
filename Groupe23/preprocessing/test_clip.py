
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_NAME = "openai/clip-vit-base-patch32"

print("Loading CLIP model...")

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model.eval()

# Test one processed image
image_path = next(
    (BASE_DIR / "dataset" / "processed" / "train").glob("*.jpg")
)

image = Image.open(image_path).convert("RGB")

inputs = processor(
    text=["A photo of an object"],
    images=image,
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    outputs = model(**inputs)

image_embedding = outputs.image_embeds
text_embedding = outputs.text_embeds

print("\nCLIP test successful!")
print("Image:", image_path.name)
print("Image embedding shape:", image_embedding.shape)
print("Text embedding shape:", text_embedding.shape)
print("Device: CPU")
