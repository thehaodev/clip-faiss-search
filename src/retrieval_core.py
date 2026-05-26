import json
from pathlib import Path

import faiss
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_project_root(start_path=None):
    if start_path is None:
        start_path = Path(__file__).resolve()
    else:
        start_path = Path(start_path).resolve()

    candidates = [start_path] if start_path.is_dir() else [start_path.parent]
    candidates.extend(candidates[0].parents)

    for candidate in candidates:
        if (candidate / "data/processed").exists() and (candidate / "src").exists():
            return candidate

    raise FileNotFoundError(f"Could not find project root from start path: {start_path}")


PROJECT_ROOT = find_project_root()
FAISS_INDEX_PATH = PROJECT_ROOT / "data/processed/clip_faiss_index_full.index"
METADATA_PATH = PROJECT_ROOT / "data/processed/clip_full_metadata.json"


def load_metadata(metadata_path=METADATA_PATH):
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}. "
            "Run notebook 05 first to create the metadata and FAISS index."
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_faiss_index(index_path=FAISS_INDEX_PATH):
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}. "
            "Run notebook 05 first to create the FAISS index."
        )

    return faiss.read_index(str(index_path))


def resolve_model_source(model_name=MODEL_NAME):
    model_path = Path(model_name)
    if model_path.exists():
        return str(model_path)

    try:
        return snapshot_download(model_name, local_files_only=True)
    except Exception:
        return model_name


def load_clip_model(model_name=MODEL_NAME, device=DEVICE):
    model_source = resolve_model_source(model_name)
    model = CLIPModel.from_pretrained(model_source, use_safetensors=False)
    processor = CLIPProcessor.from_pretrained(model_source)
    model.to(device)
    model.eval()
    return model, processor


def _extract_projected_text_features(model_output):
    if torch.is_tensor(model_output):
        return model_output

    if hasattr(model_output, "text_embeds") and model_output.text_embeds is not None:
        return model_output.text_embeds

    if hasattr(model_output, "pooler_output") and model_output.pooler_output is not None:
        return model_output.pooler_output

    raise TypeError(f"Cannot extract text features from output type: {type(model_output)}")


def encode_text_query(query, model, processor, device=DEVICE):
    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        text_features = _extract_projected_text_features(
            model.get_text_features(**inputs),
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    query_embedding = text_features.cpu().numpy().astype(np.float32)
    query_embedding = np.ascontiguousarray(query_embedding.reshape(1, -1))
    return query_embedding


def faiss_retrieve(query, top_k, model, processor, index, metadata, device=DEVICE):
    query_embedding = encode_text_query(query, model, processor, device=device)
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for rank, (image_index, score) in enumerate(zip(indices[0], scores[0]), start=1):
        image_index = int(image_index)
        if image_index < 0 or image_index >= len(metadata):
            continue

        item = metadata[image_index]
        results.append(
            {
                "rank": rank,
                "index_id": image_index,
                "image_id": item["image_id"],
                "image_path": item["image_path"],
                "score": float(score),
                "captions": item["captions"],
            }
        )

    return results
