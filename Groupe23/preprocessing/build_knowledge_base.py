
import numpy as np
import pandas as pd
import faiss
import json
from pathlib import Path

# Project paths
BASE = Path(__file__).resolve().parent.parent
EMBEDDINGS = BASE / "results" / "embeddings"
OUTPUT = BASE / "results" / "knowledge_base"

OUTPUT.mkdir(parents=True, exist_ok=True)

print("Building FAISS Knowledge Base...\n")


def build_index(embedding_file, metadata_file, index_name):

    # Load embeddings and metadata
    vectors = np.load(EMBEDDINGS / embedding_file)
    metadata = pd.read_csv(EMBEDDINGS / metadata_file)

    print(f"Processing {index_name}...")

    # Validate dimensions and metadata
    if vectors.ndim != 2 or vectors.shape[1] != 512:
        raise ValueError(f"Invalid embedding shape: {vectors.shape}")

    if len(vectors) != len(metadata):
        raise ValueError("Embeddings and metadata counts do not match.")

    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain NaN or Inf.")

    # FAISS requires contiguous float32 vectors
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    # Create cosine similarity index
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)

    # Add vectors
    index.add(vectors)

    # Save FAISS index
    index_path = OUTPUT / f"{index_name}.index"
    faiss.write_index(index, str(index_path))

    # Save corresponding metadata
    metadata_path = OUTPUT / f"{index_name}_metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    print("Vectors:", index.ntotal)
    print("Dimension:", dimension)
    print("Index saved:", index_path.name)
    print("Metadata saved:", metadata_path.name)
    print("Status: PASS\n")

    return index.ntotal


# Build image index using TRAIN images only
image_count = build_index(
    "train_image_embeddings.npy",
    "train_image_metadata.csv",
    "image_index"
)

# Build text index using TRAIN captions only
text_count = build_index(
    "train_text_embeddings.npy",
    "train_text_metadata.csv",
    "text_index"
)

# Save knowledge base information
info = {
    "method": "FAISS IndexFlatIP",
    "similarity_metric": "Cosine Similarity",
    "embedding_dimension": 512,
    "training_images": image_count,
    "training_captions": text_count,
    "data_used": "Train only",
    "validation_indexed": False,
    "test_indexed": False
}

with open(OUTPUT / "knowledge_base_info.json", "w") as f:
    json.dump(info, f, indent=4)

print("================================")
print("KNOWLEDGE BASE CREATED SUCCESSFULLY")
print("Image vectors:", image_count)
print("Text vectors:", text_count)
print("Output folder:", OUTPUT)
print("================================")
