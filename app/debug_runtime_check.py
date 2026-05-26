import sys
from pathlib import Path


PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

import torch
import transformers

from src.retrieval_core import (
    DEVICE,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    PROJECT_ROOT,
    faiss_retrieve,
    load_clip_model,
    load_faiss_index,
    load_metadata,
)


def print_environment_warning():
    expected_venv_python = PROJECT_ROOT / ".venv/Scripts/python.exe"
    if not expected_venv_python.exists():
        print("WARNING: Project .venv Python was not found; cannot compare app and notebook environments.")
        return

    current_python = sys.executable.lower()
    expected_python = str(expected_venv_python).lower()
    if current_python != expected_python:
        print("WARNING: This script is not running with the project .venv Python.")
        print(f"Current Python:  {sys.executable}")
        print(f"Expected Python: {expected_venv_python}")
        print("If the notebook uses .venv, app results may differ because the runtime environment differs.")


def main():
    print(f"sys.executable: {sys.executable}")
    print(f"torch.__version__: {torch.__version__}")
    print(f"transformers.__version__: {transformers.__version__}")
    print(f"DEVICE: {DEVICE}")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"FAISS_INDEX_PATH: {FAISS_INDEX_PATH}")
    print(f"METADATA_PATH: {METADATA_PATH}")
    print(f"FAISS_INDEX_PATH.exists(): {FAISS_INDEX_PATH.exists()}")
    print(f"METADATA_PATH.exists(): {METADATA_PATH.exists()}")
    print_environment_warning()

    index = load_faiss_index()
    metadata = load_metadata()
    model, processor = load_clip_model()

    print(f"index.ntotal: {index.ntotal}")
    print(f"len(metadata): {len(metadata)}")

    query = "a dog running on the grass"
    results = faiss_retrieve(query, 5, model, processor, index, metadata)

    print()
    print(f"Query: {query}")
    print("Top-5:")
    for result in results:
        caption = result["captions"][0] if result["captions"] else ""
        print(
            f"Rank {result['rank']}: "
            f"{result['image_id']}, "
            f"score={result['score']:.6f}, "
            f"caption={caption}"
        )


if __name__ == "__main__":
    main()
