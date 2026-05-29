import argparse
import json
import re
from pathlib import Path

import faiss
import numpy as np
import PIL.Image
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch16"
DEFAULT_METADATA_PATH = Path("data/processed/clip_full_metadata.json")
DEFAULT_OUTPUT_DIR = Path("data/processed/model_benchmarks")
DEFAULT_TOP_K = (1, 5, 10)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode images and evaluate text-to-image retrieval for a CLIP model."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument(
        "--split-name",
        choices=["all", "train", "val", "test"],
        default="all",
    )
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def model_slug(model_name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")


def load_metadata(path, split_path=None, split_name="all", max_images=None):
    with path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if split_name != "all":
        if split_path is None:
            raise ValueError("--split-path is required when --split-name is not all")
        with split_path.open("r", encoding="utf-8") as file:
            split = json.load(file)
        split_key = f"{split_name}_indices"
        metadata = [metadata[index] for index in split[split_key]]

    if max_images is not None:
        metadata = metadata[:max_images]

    return metadata


def load_model(model_name, device):
    model_source = resolve_model_source(model_name)
    model_source_path = Path(model_source)
    model_kwargs = {}
    if not (model_source_path.exists() and (model_source_path / "model.safetensors").exists()):
        model_kwargs["use_safetensors"] = False

    model = CLIPModel.from_pretrained(model_source, **model_kwargs)
    processor = CLIPProcessor.from_pretrained(model_source)
    model.to(device)
    model.eval()
    return model, processor


def resolve_model_source(model_name):
    model_path = Path(model_name)
    if model_path.exists():
        return str(model_path)

    try:
        return snapshot_download(model_name, local_files_only=True)
    except Exception:
        return model_name


def extract_features(model_output, embed_attr, pooler_attr="pooler_output"):
    if torch.is_tensor(model_output):
        return model_output

    if hasattr(model_output, embed_attr) and getattr(model_output, embed_attr) is not None:
        return getattr(model_output, embed_attr)

    if hasattr(model_output, pooler_attr) and getattr(model_output, pooler_attr) is not None:
        return getattr(model_output, pooler_attr)

    raise TypeError(f"Cannot extract features from output type: {type(model_output)}")


def encode_images(metadata, model, processor, device, batch_size):
    embeddings = []

    for start in tqdm(range(0, len(metadata), batch_size), desc="Encoding images"):
        batch_items = metadata[start : start + batch_size]
        images = []

        for item in batch_items:
            image_path = Path(item["image_path"])
            image = PIL.Image.open(image_path).convert("RGB")
            images.append(image)

        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            image_features = extract_features(
                model.get_image_features(**inputs),
                "image_embeds",
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        embeddings.append(image_features.cpu().numpy().astype(np.float32))

    image_embeddings = np.concatenate(embeddings, axis=0)
    return np.ascontiguousarray(image_embeddings)


def build_faiss_index(image_embeddings):
    index = faiss.IndexFlatIP(image_embeddings.shape[1])
    index.add(image_embeddings)
    return index


def iter_caption_batches(metadata, batch_size):
    captions = []
    targets = []

    for image_index, item in enumerate(metadata):
        for caption in item.get("captions", []):
            captions.append(caption)
            targets.append(image_index)

            if len(captions) == batch_size:
                yield captions, np.array(targets, dtype=np.int64)
                captions = []
                targets = []

    if captions:
        yield captions, np.array(targets, dtype=np.int64)


def encode_texts(captions, model, processor, device):
    inputs = processor(
        text=captions,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        text_features = extract_features(
            model.get_text_features(**inputs),
            "text_embeds",
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return np.ascontiguousarray(text_features.cpu().numpy().astype(np.float32))


def evaluate_recall(metadata, model, processor, index, device, batch_size, top_k_values):
    max_k = max(top_k_values)
    hit_counts = {k: 0 for k in top_k_values}
    ranks = []
    total = 0

    batches = iter_caption_batches(metadata, batch_size)
    for captions, targets in tqdm(batches, desc="Evaluating Recall@K"):
        text_embeddings = encode_texts(captions, model, processor, device)
        _, indices = index.search(text_embeddings, max_k)

        for target, retrieved in zip(targets, indices):
            total += 1
            matches = np.where(retrieved == target)[0]
            rank = int(matches[0]) + 1 if len(matches) else None

            if rank is not None:
                ranks.append(rank)
                for k in top_k_values:
                    if rank <= k:
                        hit_counts[k] += 1

    recall = {f"Recall@{k}": hit_counts[k] / total for k in top_k_values}
    mean_rank = float(np.mean(ranks)) if ranks else None
    median_rank = float(np.median(ranks)) if ranks else None
    return {
        "num_queries": total,
        "recall": recall,
        "mean_rank_within_top_k": mean_rank,
        "median_rank_within_top_k": median_rank,
    }


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    slug = model_slug(args.model_name)
    suffix_parts = [slug]
    if args.split_name != "all":
        suffix_parts.append(args.split_name)
    if args.max_images:
        suffix_parts.append(str(args.max_images))
    suffix = "_".join(suffix_parts)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = args.output_dir / f"{suffix}_image_embeddings.npy"
    index_path = args.output_dir / f"{suffix}_faiss.index"
    results_path = args.output_dir / f"{suffix}_recall_results.json"

    metadata = load_metadata(
        args.metadata_path,
        args.split_path,
        args.split_name,
        args.max_images,
    )
    model, processor = load_model(args.model_name, device)

    if embeddings_path.exists() and not args.force:
        image_embeddings = np.load(embeddings_path)
    else:
        image_embeddings = encode_images(
            metadata,
            model,
            processor,
            device,
            args.image_batch_size,
        )
        np.save(embeddings_path, image_embeddings)

    if index_path.exists() and not args.force:
        index = faiss.read_index(str(index_path))
    else:
        index = build_faiss_index(image_embeddings)
        faiss.write_index(index, str(index_path))

    if len(metadata) != index.ntotal:
        raise ValueError(f"metadata length ({len(metadata)}) != index.ntotal ({index.ntotal})")

    results = {
        "model_name": args.model_name,
        "device": device,
        "num_images": len(metadata),
        "embedding_dim": int(index.d),
        "metadata_path": str(args.metadata_path),
        "split_path": str(args.split_path) if args.split_path else None,
        "split_name": args.split_name,
        "embeddings_path": str(embeddings_path),
        "index_path": str(index_path),
    }

    if not args.skip_eval:
        results.update(
            evaluate_recall(
                metadata,
                model,
                processor,
                index,
                device,
                args.text_batch_size,
                DEFAULT_TOP_K,
            )
        )

    with results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
