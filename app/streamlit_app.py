import json
import sys
from pathlib import Path

import numpy as np
import PIL.Image
import streamlit as st
import torch
import transformers

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from src.retrieval_core import (
    DEVICE,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    MODEL_NAME,
    PROJECT_ROOT,
    encode_text_query,
    faiss_retrieve,
    load_clip_model as core_load_clip_model,
    load_faiss_index as core_load_faiss_index,
    load_metadata as core_load_metadata,
)


RESULTS_PATH = PROJECT_ROOT / "data/processed/full_recall_results.json"


@st.cache_resource(show_spinner="Loading CLIP model...")
def load_model():
    try:
        return core_load_clip_model()
    except Exception as exc:
        st.error(
            f"Cannot load CLIP model `{MODEL_NAME}`: {exc}. "
            "Make sure the model is cached locally or that Hugging Face is reachable."
        )
        st.stop()


@st.cache_resource(show_spinner="Loading FAISS index...")
def load_index():
    try:
        return core_load_faiss_index()
    except FileNotFoundError:
        st.error("Run notebook 05 first to create the FAISS index.")
        st.stop()
    except Exception as exc:
        st.error(f"Cannot load FAISS index: {exc}")
        st.stop()


@st.cache_data(show_spinner="Loading metadata...")
def load_metadata():
    try:
        return core_load_metadata()
    except FileNotFoundError:
        st.error("Run notebook 05 first to create the metadata and FAISS index.")
        st.stop()
    except Exception as exc:
        st.error(f"Cannot load metadata: {exc}")
        st.stop()


@st.cache_data(show_spinner=False)
def load_recall_results():
    if not RESULTS_PATH.exists():
        return None

    try:
        with RESULTS_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return None


def retrieve(query, top_k, model, processor, index, metadata):
    return faiss_retrieve(query, top_k, model, processor, index, metadata)


def show_recall_metrics(results):
    if not results:
        return

    recall = results.get("recall", {})
    metric_values = {
        "Recall@1": recall.get("Recall@1"),
        "Recall@5": recall.get("Recall@5"),
        "Recall@10": recall.get("Recall@10"),
    }

    st.sidebar.divider()
    st.sidebar.subheader("Evaluation")
    for name, value in metric_values.items():
        if value is not None:
            st.sidebar.metric(name, f"{value * 100:.2f}%")


def show_environment_warning():
    expected_venv_python = PROJECT_ROOT / ".venv/Scripts/python.exe"
    if expected_venv_python.exists():
        current_python = sys.executable.lower()
        expected_python = str(expected_venv_python).lower()
        if current_python != expected_python:
            st.warning(
                "Streamlit is not running with the project `.venv` Python. "
                f"Current: `{sys.executable}`. Expected: `{expected_venv_python}`."
            )


def show_debug_self_test(model, processor, index, metadata):
    debug_query = "a dog running on the grass"
    debug_results = retrieve(debug_query, 5, model, processor, index, metadata)
    query_embedding = encode_text_query(debug_query, model, processor)

    st.sidebar.divider()
    st.sidebar.subheader("Debug runtime")
    st.sidebar.write(
        {
            "sys.executable": sys.executable,
            "torch.__version__": torch.__version__,
            "transformers.__version__": transformers.__version__,
            "DEVICE": DEVICE,
            "PROJECT_ROOT": str(PROJECT_ROOT),
            "FAISS_INDEX_PATH": str(FAISS_INDEX_PATH),
            "METADATA_PATH": str(METADATA_PATH),
            "FAISS_INDEX_PATH.exists()": FAISS_INDEX_PATH.exists(),
            "METADATA_PATH.exists()": METADATA_PATH.exists(),
            "index.ntotal": index.ntotal,
            "len(metadata)": len(metadata),
            "MODEL_NAME": MODEL_NAME,
            "type(model)": str(type(model)),
            "type(processor)": str(type(processor)),
            "query_embedding shape": query_embedding.shape,
            "query_embedding norm": float(np.linalg.norm(query_embedding)),
            "query_embedding first 5": query_embedding.squeeze(0)[:5].tolist(),
        }
    )

    st.subheader("Debug self-test")
    st.write(f"Debug query: `{debug_query}`")
    st.caption(
        "Expected roughly: Rank 1 2982928615.jpg score ~0.346, "
        "Rank 2 3597924257.jpg score ~0.343, "
        "Rank 3 3540416139.jpg score ~0.339."
    )
    st.write(
        [
            {
                "rank": result["rank"],
                "image_id": result["image_id"],
                "score": result["score"],
                "caption": result["captions"][0] if result["captions"] else "",
            }
            for result in debug_results
        ]
    )
    show_environment_warning()


def show_result_card(result):
    image_path = PROJECT_ROOT / result["image_path"]

    if image_path.exists():
        image = PIL.Image.open(image_path)
        st.image(image, use_container_width=True)
    else:
        st.warning(f"Image file not found: {result['image_path']}")

    st.markdown(f"**Rank:** {result['rank']}")
    st.markdown(f"**Score:** {result['score']:.4f}")
    st.markdown(f"**Image ID:** `{result['image_id']}`")

    captions = result.get("captions", [])
    if captions:
        st.caption(captions[0])

    with st.expander("All captions"):
        if captions:
            for caption in captions:
                st.write(caption)
        else:
            st.write("No captions.")


def main():
    st.set_page_config(page_title="Slide Media Retrieval", layout="wide")

    st.title("Slide Media Retrieval")
    st.caption("Search for matching images from a text description using CLIP + FAISS")

    st.sidebar.subheader("Settings")
    st.sidebar.write(f"**Device:** `{DEVICE}`")
    st.sidebar.write(f"**Model:** `{MODEL_NAME}`")
    top_k = st.sidebar.slider("Top-k", min_value=1, max_value=10, value=5)
    debug_mode = st.sidebar.checkbox("Debug mode", value=False)

    sample_queries = [
        "A dog running through grass",
        "A group of people playing football",
        "A child wearing a red shirt",
        "A man riding a bicycle on the street",
        "People standing near the ocean",
    ]
    selected_sample = st.sidebar.selectbox("Sample query", [""] + sample_queries)

    recall_results = load_recall_results()
    show_recall_metrics(recall_results)

    default_query = selected_sample if selected_sample else ""
    query = st.text_area(
        "Image description",
        value=default_query,
        placeholder="Example: A dog running through grass",
        height=90,
    )

    search_clicked = st.button("Search", type="primary")

    if not search_clicked and not debug_mode:
        return

    model, processor = load_model()
    index = load_index()
    metadata = load_metadata()

    if debug_mode:
        show_debug_self_test(model, processor, index, metadata)

    if not search_clicked:
        return

    query = query.strip()
    if not query:
        st.warning("Enter an image description before searching.")
        return

    with st.spinner("Searching..."):
        try:
            results = retrieve(query, top_k, model, processor, index, metadata)
        except Exception as exc:
            st.error(
                f"Cannot run the search: {exc}. "
                "Check the model, index, and metadata files."
            )
            return

    if not results:
        st.warning("No matching results found.")
        return

    st.subheader("Results")
    columns_per_row = min(top_k, 5)
    rows = [
        results[index : index + columns_per_row]
        for index in range(0, len(results), columns_per_row)
    ]

    for row in rows:
        columns = st.columns(columns_per_row)
        for column, result in zip(columns, row):
            with column:
                show_result_card(result)


if __name__ == "__main__":
    main()
