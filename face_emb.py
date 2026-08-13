import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

# =========================
# CONFIG
# =========================
DATASET_DIR = "dataset"
OUTPUT_FILE = "embeddings/embeddings.pkl"

# Create output folder if not exists
os.makedirs("embeddings", exist_ok=True)

# Load InsightFace model
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 for GPU, -1 for CPU

known_embeddings = {}

for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_name)

    if not os.path.isdir(person_path):
        continue

    person_embeddings = []

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read: {img_path}")
            continue

        faces = app.get(img)

        if len(faces) == 0:
            print(f"No face found in: {img_path}")
            continue

        # If multiple faces, select the biggest one
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        embedding = face.embedding
        person_embeddings.append(embedding)

        print(f"Processed: {img_path}")

    if len(person_embeddings) > 0:
        known_embeddings[person_name] = person_embeddings
        print(f"{person_name}: {len(person_embeddings)} embeddings saved")
    else:
        print(f"No valid images for {person_name}")

# Save embeddings
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(known_embeddings, f)

print(f"\nEmbeddings saved to {OUTPUT_FILE}")