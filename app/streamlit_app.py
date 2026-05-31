import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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


BASELINE_RESULTS_PATH = PROJECT_ROOT / "data/processed/full_recall_results.json"
BENCHMARK_RESULTS_PATH = (
    PROJECT_ROOT / "data/processed/clip_model_benchmark/benchmark_results.json"
)
FINETUNED_RESULTS_PATH = (
    PROJECT_ROOT / "data/processed/faiss_finetuned_clip_l14/faiss_recall_results.json"
)
SAMPLE_RESULTS_PATH = (
    PROJECT_ROOT / "data/processed/faiss_finetuned_clip_l14/sample_search_results.csv"
)

RESULTS_PATH = FINETUNED_RESULTS_PATH if FINETUNED_RESULTS_PATH.exists() else BASELINE_RESULTS_PATH


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


@st.cache_data(show_spinner=False)
def load_benchmark_results():
    if not BENCHMARK_RESULTS_PATH.exists():
        return pd.DataFrame()

    try:
        rows = json.loads(BENCHMARK_RESULTS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return pd.DataFrame()

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_sample_results():
    if not SAMPLE_RESULTS_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(SAMPLE_RESULTS_PATH)
    except OSError:
        return pd.DataFrame()


def retrieve(query, top_k, model, processor, index, metadata):
    return faiss_retrieve(query, top_k, model, processor, index, metadata)


def normalize_recall_results(results):
    if not results:
        return {}

    recall = results.get("recall", results)
    return {
        name: recall.get(name)
        for name in ["Recall@1", "Recall@5", "Recall@10"]
        if recall.get(name) is not None
    }


def show_recall_metrics(results):
    if not results:
        return

    metric_values = normalize_recall_results(results)

    st.sidebar.divider()
    st.sidebar.subheader("Final evaluation")
    for name, value in metric_values.items():
        st.sidebar.metric(name, f"{value * 100:.2f}%")


def format_percent(value):
    if pd.isna(value):
        return ""
    return f"{value * 100:.2f}%"


def show_project_summary(recall_results):
    st.subheader("Project Overview")
    metric_values = normalize_recall_results(recall_results)

    columns = st.columns(4)
    columns[0].metric("Dataset images", f"{recall_results.get('num_images', 31782):,}")
    columns[1].metric("Text queries", f"{recall_results.get('num_queries', 158910):,}")
    columns[2].metric("Best Recall@10", format_percent(metric_values.get("Recall@10")))
    columns[3].metric("Index type", recall_results.get("index_type", "IndexFlatIP"))

    st.markdown(
        """
        Hệ thống nhận mô tả văn bản, mã hóa bằng CLIP text encoder, sau đó tìm ảnh gần
        nhất trong FAISS index đã build từ image embeddings. Nhóm benchmark nhiều cấu
        hình CLIP và chọn fine-tuned CLIP ViT-L/14 cho kết quả cuối.
        """
    )

    steps = pd.DataFrame(
        [
            {"Step": "1. Metadata", "Output": "image_id, image_path, captions"},
            {"Step": "2. CLIP embeddings", "Output": "normalized image/text vectors"},
            {"Step": "3. FAISS index", "Output": "IndexFlatIP for cosine search"},
            {"Step": "4. Retrieval", "Output": "top-k images ranked by similarity"},
            {"Step": "5. Evaluation", "Output": "Recall@1, Recall@5, Recall@10"},
        ]
    )
    st.dataframe(steps, use_container_width=True, hide_index=True)


def show_benchmark_section(recall_results):
    st.subheader("Model Benchmark")
    benchmark_df = load_benchmark_results()

    if benchmark_df.empty:
        st.info("Benchmark file was not found.")
    else:
        baseline_row = benchmark_df[benchmark_df["short_name"] == "clip_b32"]
        display_df = benchmark_df[
            [
                "short_name",
                "model_name",
                "embedding_dim",
                "Recall@1",
                "Recall@5",
                "Recall@10",
                "elapsed_seconds",
            ]
        ].copy()
        display_df["Recall@1"] = display_df["Recall@1"].map(format_percent)
        display_df["Recall@5"] = display_df["Recall@5"].map(format_percent)
        display_df["Recall@10"] = display_df["Recall@10"].map(format_percent)
        display_df["elapsed_seconds"] = display_df["elapsed_seconds"].map(
            lambda value: f"{value:.1f}s"
        )
        display_df.columns = [
            "Short name",
            "Model",
            "Embedding dim",
            "Recall@1",
            "Recall@5",
            "Recall@10",
            "Eval time",
        ]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        if not baseline_row.empty and recall_results:
            show_improvement_summary(baseline_row.iloc[0], recall_results)

    st.subheader("Final Fine-tuned Result")
    metric_values = normalize_recall_results(recall_results)
    columns = st.columns(3)
    for column, name in zip(columns, ["Recall@1", "Recall@5", "Recall@10"]):
        column.metric(name, format_percent(metric_values.get(name)))

    st.caption(f"Model checkpoint: `{recall_results.get('model_checkpoint', MODEL_NAME)}`")


def show_improvement_summary(baseline_row, recall_results):
    st.subheader("Improvement over Baseline B32")
    final_metrics = normalize_recall_results(recall_results)

    improvement_rows = []
    for metric_name in ["Recall@1", "Recall@5", "Recall@10"]:
        baseline_value = baseline_row.get(metric_name)
        final_value = final_metrics.get(metric_name)
        if baseline_value is None or final_value is None:
            continue

        improvement_rows.append(
            {
                "Metric": metric_name,
                "Baseline B32": format_percent(baseline_value),
                "Fine-tuned L14": format_percent(final_value),
                "Gain": f"+{(final_value - baseline_value) * 100:.2f} pts",
            }
        )

    if improvement_rows:
        st.dataframe(pd.DataFrame(improvement_rows), use_container_width=True, hide_index=True)


def show_sample_section():
    st.subheader("Saved Sample Searches")
    sample_df = load_sample_results()
    if sample_df.empty:
        st.info("Sample search result file was not found.")
        return

    selected_query = st.selectbox("Sample query", sample_df["query"].unique())
    query_df = sample_df[sample_df["query"] == selected_query].copy()
    query_df = query_df[["rank", "score", "image", "caption_example"]]
    query_df["score"] = query_df["score"].map(lambda value: f"{value:.4f}")
    query_df.columns = ["Rank", "Similarity", "Image", "Caption"]
    st.dataframe(query_df, use_container_width=True, hide_index=True)

    show_sample_images(sample_df[sample_df["query"] == selected_query])


def show_sample_images(query_df):
    st.caption("Top-5 retrieved images for the selected sample query.")
    columns = st.columns(min(len(query_df), 5))

    for column, (_, row) in zip(columns, query_df.iterrows()):
        image_path = PROJECT_ROOT / "data/raw/Images" / row["image"]
        with column:
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
            else:
                st.warning(f"Missing image: {row['image']}")

            st.markdown(f"**Rank:** {int(row['rank'])}")
            st.markdown(f"**Similarity:** `{row['score']:.4f}`")
            st.caption(row["caption_example"])


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

    match_label = "Top match" if result["rank"] == 1 else f"Match #{result['rank']}"
    st.markdown(f"**Rank:** {result['rank']} · {match_label}")
    st.markdown(f"**Similarity:** `{result['score']:.4f}`")
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
    st.set_page_config(page_title="CLIP FAISS Retrieval Results", layout="wide")

    st.title("CLIP + FAISS Image Retrieval")
    st.caption("Group result dashboard and live text-to-image retrieval demo")

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

    overview_tab, benchmark_tab, search_tab = st.tabs(
        ["Overview", "Evaluation Results", "Live Search Demo"]
    )

    with overview_tab:
        show_project_summary(recall_results or {})

    with benchmark_tab:
        show_benchmark_section(recall_results or {})
        show_sample_section()

    with search_tab:
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
        st.caption(
            "Similarity is used to rank images in the CLIP embedding space. "
            "It is not a calibrated match percentage."
        )
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
