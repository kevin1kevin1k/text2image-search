import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from constants import CLIENT_URL, COLLECTION_NAME
from models import PROJECTION_DIM, model, processor
from utils import load_images_and_payloads

qdrant_client = QdrantClient(CLIENT_URL)


def create_and_upload_collection(folder_path: str):
    images, payloads = load_images_and_payloads(folder_path=folder_path)
    inputs = processor(images=images, return_tensors="pt")
    with torch.inference_mode():
        image_embeddings = model.get_image_features(**inputs).numpy()
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=PROJECTION_DIM,
            distance=Distance.COSINE,
        ),
    )

    qdrant_client.upload_collection(
        collection_name=COLLECTION_NAME,
        vectors=image_embeddings,
        payload=payloads,
        batch_size=256,
    )


if __name__ == "__main__":
    create_and_upload_collection(folder_path="./open_images/00000/")
