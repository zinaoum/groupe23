
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
EMBEDDINGS = BASE / "results" / "embeddings"

files = {
    "train_image": (10000, "train_image_embeddings.npy",
                    "train_image_metadata.csv"),
    "validation_image": (100, "validation_image_embeddings.npy",
                         "validation_image_metadata.csv"),
    "test_image": (1000, "test_image_embeddings.npy",
                   "test_image_metadata.csv"),
    "train_text": (50000, "train_text_embeddings.npy",
                   "train_text_metadata.csv")
}

print("Checking embeddings...\n")

all_valid = True

for name, (expected, array_file, metadata_file) in files.items():

    vectors = np.load(EMBEDDINGS / array_file)
    metadata = pd.read_csv(EMBEDDINGS / metadata_file)

    valid = (
        vectors.shape == (expected, 512)
        and len(metadata) == expected
        and np.isfinite(vectors).all()
        and np.allclose(
            np.linalg.norm(vectors, axis=1),
            1.0,
            atol=1e-4
        )
    )

    print(name)
    print("Shape:", vectors.shape)
    print("Metadata rows:", len(metadata))
    print("No NaN/Inf:", np.isfinite(vectors).all())
    print("Normalized:", np.allclose(
        np.linalg.norm(vectors, axis=1),
        1.0,
        atol=1e-4
    ))
    print("Status:", "PASS" if valid else "FAIL")
    print()

    all_valid = all_valid and valid

print("FINAL RESULT:", "ALL CHECKS PASSED" if all_valid else "CHECK FAILED")
