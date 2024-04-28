import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from torch.utils.data import DataLoader

from constants import CLIENT_URL, COLLECTION_NAME, IMAGES_PATH
from models import PROJECTION_DIM, model, processor
from utils import ImagePayloadDataset

qdrant_client = QdrantClient(CLIENT_URL)


def create_and_upload_collection(folder_path: str):
    if qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.delete_collection(COLLECTION_NAME)
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=PROJECTION_DIM,
            distance=Distance.COSINE,
        ),
    )

    dataset = ImagePayloadDataset(folder_path=folder_path)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    for batch in dataloader:
        images, payloads = batch["image"], batch["payload"]
        payloads = [{"file_stem": stem} for stem in payloads["file_stem"]]
        inputs = processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            image_embeddings = model.get_image_features(**inputs).numpy()

        qdrant_client.upload_collection(
            collection_name=COLLECTION_NAME,
            vectors=image_embeddings,
            payload=payloads,
            batch_size=256,
        )


if __name__ == "__main__":
    print("Creating and uploading collection...")
    create_and_upload_collection(folder_path=IMAGES_PATH)
    print("Done!")
