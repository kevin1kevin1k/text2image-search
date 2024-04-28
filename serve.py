import json

import streamlit as st
from qdrant_client import QdrantClient
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image

from constants import CLIENT_URL, COLLECTION_NAME, IMAGES_PATH
from models import model, tokenizer
from utils import load_json_dict


def search_images(qdrant_client, query_text, score_threshold=0.1):
    inputs = tokenizer([query_text], padding=True, return_tensors="pt")

    text_embedding = model.get_text_features(**inputs)

    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=text_embedding.flatten().tolist(),
        limit=3,
        score_threshold=score_threshold,
    )

    for r in search_results:
        yield r.payload["file_stem"], r.score


def _load_image_and_payload(file_stem):
    image = to_pil_image(read_image(f"{IMAGES_PATH}/{file_stem}.jpg"))
    payload = load_json_dict(f"{IMAGES_PATH}/{file_stem}.json")
    return image, payload


def _transform_json_list(json_list: str):
    return json.loads(json_list.replace("'", '"'))


def run_app():
    qdrant_client = QdrantClient(CLIENT_URL)

    st.title("Neural Image Semantic Search")
    st.write("Search for images using natural language.")

    query = st.text_input("Enter your image search query:")
    score_threshold = st.slider(
        "Similarity score threshold: ",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
    )

    if st.button("Search"):
        if query:
            results = search_images(qdrant_client, query, score_threshold)
            st.subheader("Results:")
            n_cols = 3
            cols = st.columns(n_cols)
            for i, (file_stem, score) in enumerate(results):
                image, payload = _load_image_and_payload(file_stem)
                image_labels = _transform_json_list(payload["image_labels"])
                bbox_labels = _transform_json_list(payload["bbox_labels"])
                cols[i % n_cols].image(image)
                cols[i % n_cols].caption(
                    f"""
                    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
                        <h3 style="margin-top: 0; margin-bottom: 10px;">Score: {score:.2f}</h3>
                        <p><strong>Image Labels:</strong> {', '.join(image_labels)}</p>
                        <p><strong>Bounding Box Labels:</strong> {', '.join(bbox_labels)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning("Please enter a search query.")


if __name__ == "__main__":
    run_app()
