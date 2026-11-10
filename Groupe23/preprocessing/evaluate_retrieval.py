
import json
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent
KB = BASE / "results" / "knowledge_base"
EMB = BASE / "results" / "embeddings"
ANN = BASE / "dataset" / "annotations"

TOP_K = 5

print("Starting fast Retrieval Evaluation...")

# Load FAISS index and data
index = faiss.read_index(str(KB / "image_index.index"))

train_meta = pd.read_csv(KB / "image_index_metadata.csv")
val_meta = pd.read_csv(EMB / "validation_image_metadata.csv")
val_vectors = np.load(EMB / "validation_image_embeddings.npy")

# Load COCO category annotations
def load_categories(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = defaultdict(set)

    for ann in data["annotations"]:
        result[ann["image_id"]].add(ann["category_id"])

    return result


train_categories = load_categories(
    ANN / "instances_train2017.json"
)

val_categories = train_categories

# Build inverted index:
# category_id -> set of training image IDs
category_to_images = defaultdict(set)

for image_id, categories in train_categories.items():
    for category in categories:
        category_to_images[category].add(image_id)

# Search all validation images
scores, indices = index.search(
    np.ascontiguousarray(val_vectors, dtype=np.float32),
    TOP_K
)

train_ids = set(train_meta["image_id"].astype(int))

recalls = []
reciprocal_ranks = []
similarities = []
evaluated = 0

for i, row in val_meta.iterrows():

    query_id = int(row["image_id"])
    query_categories = val_categories.get(query_id, set())

    if not query_categories:
        continue

    # Find relevant images using category index
    relevant_ids = set()

    for category in query_categories:
        relevant_ids.update(category_to_images[category])

    # Exclude the query itself and keep train images
    relevant_ids &= train_ids
    relevant_ids.discard(query_id)

    if not relevant_ids:
        continue

    retrieved_ids = [
        int(train_meta.iloc[idx]["image_id"])
        for idx in indices[i]
    ]

    relevant_retrieved = set(retrieved_ids) & relevant_ids

    recalls.append(len(relevant_retrieved) / len(relevant_ids))

    rank = next(
        (r for r, image_id in enumerate(retrieved_ids, 1)
         if image_id in relevant_ids),
        None
    )

    reciprocal_ranks.append(1 / rank if rank else 0)

    similarities.extend(scores[i].tolist())
    evaluated += 1

    if evaluated % 20 == 0:
        print(f"Evaluated: {evaluated}/100")

if evaluated == 0:
    raise RuntimeError("No validation images could be evaluated.")

results = {
    "evaluated_images": evaluated,
    "recall_at_5": float(np.mean(recalls)),
    "mrr": float(np.mean(reciprocal_ranks)),
    "average_similarity": float(np.mean(similarities)),
    "relevance_rule": "At least one shared COCO object category"
}

output = BASE / "results" / "statistics"
output.mkdir(parents=True, exist_ok=True)

with open(output / "retrieval_evaluation.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n===== RETRIEVAL EVALUATION =====")

for key, value in results.items():
    print(f"{key}: {value}")

print("\nEVALUATION COMPLETED!")
