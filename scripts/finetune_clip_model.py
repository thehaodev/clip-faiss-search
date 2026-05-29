import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


DEFAULT_MODEL_NAME = "openai/clip-vit-large-patch14"
DEFAULT_METADATA_PATH = Path("data/processed/clip_full_metadata.json")
DEFAULT_OUTPUT_DIR = Path("data/processed/finetuned_models")
DEFAULT_SPLIT_PATH = Path("data/processed/splits/clip_retrieval_split_seed42.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP for image-text retrieval.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--train-mode",
        choices=["projection", "last-block", "full"],
        default="projection",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def model_slug(model_name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_metadata(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_model_source(model_name):
    model_path = Path(model_name)
    if model_path.exists():
        return str(model_path)

    try:
        return snapshot_download(model_name, local_files_only=True)
    except Exception:
        return model_name


def create_or_load_split(metadata, split_path, train_ratio, val_ratio, seed):
    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    rng = random.Random(seed)
    indices = list(range(len(metadata)))
    rng.shuffle(indices)

    train_end = int(len(indices) * train_ratio)
    val_end = train_end + int(len(indices) * val_ratio)
    split = {
        "seed": seed,
        "train_indices": indices[:train_end],
        "val_indices": indices[train_end:val_end],
        "test_indices": indices[val_end:],
    }

    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as file:
        json.dump(split, file, ensure_ascii=False, indent=2)

    return split


def subset_metadata(metadata, indices, max_items=None):
    if max_items is not None:
        indices = indices[:max_items]
    return [metadata[index] for index in indices]


class ImageCaptionDataset(Dataset):
    def __init__(self, metadata, random_caption=True):
        self.metadata = metadata
        self.random_caption = random_caption

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        item = self.metadata[index]
        captions = item.get("captions", [])
        if not captions:
            caption = ""
        elif self.random_caption:
            caption = random.choice(captions)
        else:
            caption = captions[0]

        return {
            "image_path": item["image_path"],
            "caption": caption,
        }


def collate_batch(batch, processor):
    images = [
        PIL.Image.open(item["image_path"]).convert("RGB")
        for item in batch
    ]
    captions = [item["caption"] for item in batch]
    return processor(
        text=captions,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )


def configure_trainable_parameters(model, train_mode):
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_name_parts = ["visual_projection", "text_projection", "logit_scale"]

    if train_mode in {"last-block", "full"}:
        trainable_name_parts.extend(
            [
                "text_model.encoder.layers.11",
                "vision_model.encoder.layers.23",
                "text_model.final_layer_norm",
                "vision_model.post_layernorm",
            ]
        )

    if train_mode == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
    else:
        for name, parameter in model.named_parameters():
            if any(part in name for part in trainable_name_parts):
                parameter.requires_grad = True

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def run_validation(model, val_loader, device):
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                outputs = model(**batch, return_loss=True)
            losses.append(float(outputs.loss.detach().cpu()))

    return float(np.mean(losses)) if losses else None


def train(args):
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    metadata = load_metadata(args.metadata_path)
    split = create_or_load_split(
        metadata,
        args.split_path,
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )

    train_metadata = subset_metadata(metadata, split["train_indices"], args.max_train_images)
    val_metadata = subset_metadata(metadata, split["val_indices"], args.max_val_images)

    output_dir = args.output_dir / f"{model_slug(args.model_name)}_{args.train_mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_source = resolve_model_source(args.model_name)
    model = CLIPModel.from_pretrained(model_source, use_safetensors=False)
    processor = CLIPProcessor.from_pretrained(model_source)
    model.to(device)

    trainable, total = configure_trainable_parameters(model, args.train_mode)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable / total:.2%})")

    train_dataset = ImageCaptionDataset(train_metadata, random_caption=True)
    val_dataset = ImageCaptionDataset(val_metadata, random_caption=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_batch(batch, processor),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_batch(batch, processor),
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    history = []
    best_val_loss = None
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer_steps = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(progress, start=1):
            batch = move_batch_to_device(batch, device)

            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                outputs = model(**batch, return_loss=True)
                loss = outputs.loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            running_loss += float(loss.detach().cpu()) * args.gradient_accumulation_steps

            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

            progress.set_postfix(loss=running_loss / step)

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss = run_validation(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "optimizer_steps": optimizer_steps,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if val_loss is not None and (best_val_loss is None or val_loss < best_val_loss):
            best_val_loss = val_loss
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

    if best_val_loss is None:
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)

    summary = {
        "base_model_name": args.model_name,
        "output_dir": str(output_dir),
        "split_path": str(args.split_path),
        "train_mode": args.train_mode,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "train_images": len(train_metadata),
        "val_images": len(val_metadata),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "history": history,
        "best_val_loss": best_val_loss,
    }
    with (output_dir / "finetune_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
