import streamlit as st
from qdrant_client import QdrantClient

from constants import CLIENT_URL, COLLECTION_NAME
from models import model, tokenizer
from utils import load_image_and_payload


def search_images(
    qdrant_client: QdrantClient, query: str, score_threshold: float = 0.1
):
    inputs = tokenizer([query], padding=True, return_tensors="pt")

    text_embedding = model.get_text_features(**inputs)

    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=text_embedding.flatten().tolist(),
        limit=3,
        score_threshold=score_threshold,
    )

    for r in search_results:
        image, payload = load_image_and_payload(r.payload["file_stem"])
        yield r.score, image, payload


def _generate_caption(score, payload):
    return f"""
        <div style="background-color: {st.get_option('theme.primaryColor')}; color: {st.get_option('theme.textColor')}; padding: 10px; border-radius: 5px;">
        <h3 style="margin-top: 0; margin-bottom: 10px;">Score: {score:.2f}</h3>
        <p><strong>Image Labels:</strong> {', '.join(payload["image_labels"])}</p>
        <p><strong>Bounding Box Labels:</strong> {', '.join(payload["bbox_labels"])}</p>
        </div>
    """


def _handle_search(qdrant_client: QdrantClient, query: str, score_threshold: float):
    if query:
        results = search_images(qdrant_client, query, score_threshold)
        st.subheader("Results:")
        n_cols = 3
        cols = st.columns(n_cols)
        for i, (score, image, payload) in enumerate(results):
            cols[i % n_cols].image(image)
            cols[i % n_cols].caption(
                _generate_caption(score, payload),
                unsafe_allow_html=True,
            )
    else:
        st.warning("Please enter a search query.")


def run_app():
    qdrant_client = QdrantClient(CLIENT_URL)

    st.title("Text-to-Image Search")
    st.write("Search for semantically similar images with natural language.")

    score_threshold = st.slider(
        "Similarity score threshold: ",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
    )
    query = st.text_input("Search query:")

    prev_query = ""
    if st.button("Search") or query != prev_query:
        prev_query = query
        _handle_search(qdrant_client, query, score_threshold)


if __name__ == "__main__":
    run_app()
