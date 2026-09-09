"""Train EUMU tagging heads.

Florence stays shared and frozen. Use `--profile quality` for the quality
specialized head and `--profile main` for the scene/event head used by the
final assembled checkpoint.

Optional teacher pseudo-label JSONL files can be mixed into the same student
head training path. Thresholds are tuned only on a training holdout or an
explicit calibration JSONL, never on the evaluation val_data split.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoProcessor, get_cosine_schedule_with_warmup, set_seed

from common import NAMESPACES, load_vocab, parse_tag_target, row_annotated_namespaces
from tagging_heads import (
    SCENE_PARENT,
    TaggingHeads,
    default_thresholds,
    encode_image_tokens,
    load_tagging_checkpoint,
    save_tagging_checkpoint,
)

CLASSIFICATION_NAMESPACES = ("scene", "event")
QUALITY_AUGS = (
    "blur",
    "motion_blur",
    "noise",
    "compression_artifacts",
    "low_light",
    "underexposure",
    "overexposure",
    "high_contrast",
    "low_contrast",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAINING_PROFILES = ("quality", "main")


class Lion(torch.optim.Optimizer):
    """Small Lion optimizer implementation to avoid an extra dependency."""

    def __init__(self, params, lr: float = 1e-4, betas=(0.9, 0.99), weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if wd:
                    p.mul_(1 - lr * wd)
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                p.add_(update.sign(), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


def iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def unique_allowed(labels: list[str], allowed: set[str]) -> list[str]:
    out = []
    seen = set()
    for label in labels:
        clean = str(label).strip().lower()
        if clean in allowed and clean not in seen:
            out.append(clean)
            seen.add(clean)
    return out


def raw_labels(value: Any, score_threshold: float = 0.5) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            out.extend(raw_labels(item, score_threshold))
        return out
    if isinstance(value, dict):
        if "labels" in value:
            return raw_labels(value["labels"], score_threshold)
        out = []
        for key, score in value.items():
            if isinstance(score, (int, float)) and float(score) < score_threshold:
                continue
            out.append(str(key))
        return out
    return []


def extract_labels_and_mask(
    row: dict,
    vocab: dict[str, list[str]],
    profile: str,
) -> tuple[dict[str, list[str]], set[str]]:
    target = str(row.get("target", ""))
    labels = parse_tag_target(target, vocab)
    annotated = set(row_annotated_namespaces(row, target))

    for field in ("tags", "teacher_tags", "pseudo_tags"):
        payload = row.get(field)
        if isinstance(payload, dict):
            for ns in NAMESPACES:
                if ns in payload:
                    labels[ns] = raw_labels(payload[ns])
                    annotated.add(ns)

    for ns in NAMESPACES:
        if ns in row:
            labels[ns] = raw_labels(row[ns])
            annotated.add(ns)

    for ns in NAMESPACES:
        labels[ns] = unique_allowed(labels[ns], set(vocab.get(ns, [])))
        if labels[ns]:
            annotated.add(ns)

    if profile == "quality":
        for ns in CLASSIFICATION_NAMESPACES:
            if not labels[ns]:
                annotated.discard(ns)
            elif len(labels[ns]) > 1:
                labels[ns] = labels[ns][:1]
    elif labels["scene"]:
        scene_allowed = set(vocab.get("scene", []))
        expanded = list(labels["scene"])
        for label in list(labels["scene"]):
            parent = SCENE_PARENT.get(label)
            if parent and parent in scene_allowed and parent not in expanded:
                expanded.append(parent)
        labels["scene"] = expanded

    return labels, {ns for ns in annotated if ns in NAMESPACES}


def extract_weight(row: dict, default_weight: float) -> float:
    for key in ("loss_weight", "weight", "confidence", "score"):
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return max(0.05, min(2.0, float(value) * default_weight))
    return float(default_weight)


def apply_quality_aug(image: Image.Image, label: str, rng: random.Random) -> Image.Image:
    if label == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.5, 3.8)))
    if label == "motion_blur":
        kernel = [0.0] * 25
        for i in range(5):
            kernel[2 * 5 + i] = 1.0 / 5.0
        return image.filter(ImageFilter.Kernel((5, 5), kernel, scale=1.0))
    if label == "compression_artifacts":
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=rng.randint(8, 35))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if label == "low_light":
        return ImageEnhance.Brightness(image).enhance(rng.uniform(0.20, 0.55))
    if label == "underexposure":
        return ImageEnhance.Brightness(image).enhance(rng.uniform(0.30, 0.70))
    if label == "overexposure":
        return ImageEnhance.Brightness(image).enhance(rng.uniform(1.45, 2.40))
    if label == "high_contrast":
        return ImageEnhance.Contrast(image).enhance(rng.uniform(1.7, 3.0))
    if label == "low_contrast":
        return ImageEnhance.Contrast(image).enhance(rng.uniform(0.20, 0.60))
    if label == "noise":
        import numpy as np

        arr = np.asarray(image).astype("float32")
        sigma = rng.uniform(10.0, 32.0)
        noise = np.random.default_rng(rng.randrange(2**31)).normal(0.0, sigma, arr.shape)
        arr = (arr + noise).clip(0, 255).astype("uint8")
        return Image.fromarray(arr, mode="RGB")
    return image


class TaggingDataset(Dataset):
    def __init__(
        self,
        sources: Path | list[tuple[Path, float]],
        vocab: dict[str, list[str]],
        max_rows: int = 0,
        synthetic_quality_prob: float = 0.0,
        seed: int = 42,
        profile: str = "main",
    ) -> None:
        self.vocab = vocab
        self.synthetic_quality_prob = float(synthetic_quality_prob)
        self.seed = int(seed)
        self.profile = profile
        self.rows: list[dict] = []
        self.stats = Counter()
        skipped = Counter()

        if isinstance(sources, Path):
            source_items = [(sources, 1.0)]
        else:
            source_items = sources

        for path, default_weight in source_items:
            if not path.exists():
                raise FileNotFoundError(path)
            source_name = path.stem
            for row in iter_jsonl(path):
                image = row.get("image") or row.get("image_path") or row.get("path")
                image_path = Path(str(image)) if image else None
                if (
                    image_path is None
                    or not image_path.exists()
                    or image_path.suffix.lower() not in IMAGE_EXTENSIONS
                ):
                    skipped[f"{source_name}:missing_image"] += 1
                    continue
                labels, annotated = extract_labels_and_mask(row, vocab, profile)
                if not annotated and not any(labels.values()):
                    skipped[f"{source_name}:no_annotation"] += 1
                    continue
                weight = extract_weight(row, default_weight)
                self.rows.append(
                    {
                        "image": str(image),
                        "labels": labels,
                        "mask": annotated,
                        "weight": weight,
                        "source": source_name,
                        "quality_aug": str(row.get("quality_aug", "") or ""),
                    }
                )
                for ns in annotated:
                    self.stats[f"{ns}_rows"] += 1
                self.stats[f"{source_name}_rows"] += 1
                if max_rows and len(self.rows) >= max_rows:
                    break
            if max_rows and len(self.rows) >= max_rows:
                break

        if not self.rows:
            raise RuntimeError(f"no usable tagging rows, skipped={dict(skipped)}")
        print(f"[dataset:tagging] rows={len(self.rows)} stats={dict(self.stats)} skipped={dict(skipped)}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        rng = random.Random(self.seed + index * 1000003)
        image = Image.open(row["image"]).convert("RGB")
        labels = {ns: list(row["labels"][ns]) for ns in NAMESPACES}
        mask = set(row["mask"])

        explicit_quality_aug = str(row.get("quality_aug", "") or "")
        if explicit_quality_aug in QUALITY_AUGS and explicit_quality_aug in self.vocab.get("quality", []):
            image = apply_quality_aug(image, explicit_quality_aug, rng)
            labels["quality"] = [explicit_quality_aug]
            mask.add("quality")
        elif self.synthetic_quality_prob > 0 and rng.random() < self.synthetic_quality_prob:
            choices = [x for x in QUALITY_AUGS if x in self.vocab.get("quality", [])]
            if choices:
                label = rng.choice(choices)
                image = apply_quality_aug(image, label, rng)
                labels["quality"] = [label]
                mask.add("quality")

        return {"image": image, "labels": labels, "mask": mask, "weight": float(row["weight"])}


def dataset_from_rows(
    source: TaggingDataset,
    rows: list[dict],
    synthetic_quality_prob: float,
    seed: int,
    name: str,
) -> TaggingDataset:
    dataset = TaggingDataset.__new__(TaggingDataset)
    dataset.vocab = source.vocab
    dataset.synthetic_quality_prob = float(synthetic_quality_prob)
    dataset.seed = int(seed)
    dataset.profile = source.profile
    dataset.rows = list(rows)
    dataset.stats = Counter()
    for row in dataset.rows:
        for ns in row["mask"]:
            dataset.stats[f"{ns}_rows"] += 1
    if not dataset.rows:
        raise RuntimeError(f"empty tagging split: {name}")
    print(f"[dataset:{name}] rows={len(dataset.rows)} stats={dict(dataset.stats)}")
    return dataset


def split_train_calib(
    dataset: TaggingDataset,
    calib_ratio: float,
    max_calib_rows: int,
    min_calib_per_namespace: int,
    seed: int,
) -> tuple[TaggingDataset, TaggingDataset | None]:
    if calib_ratio <= 0 or len(dataset.rows) < 20:
        return dataset, None

    rng = random.Random(seed)
    selected: set[int] = set()
    for ns in NAMESPACES:
        indices = [idx for idx, row in enumerate(dataset.rows) if ns in row["mask"]]
        if not indices:
            continue
        rng.shuffle(indices)
        take = max(int(round(len(indices) * calib_ratio)), min_calib_per_namespace)
        take = min(take, max(1, len(indices) - 1))
        selected.update(indices[:take])

    if max_calib_rows > 0 and len(selected) > max_calib_rows:
        selected = set(rng.sample(sorted(selected), max_calib_rows))
    if not selected:
        return dataset, None
    if len(selected) >= len(dataset.rows):
        keep = max(1, int(round(len(dataset.rows) * calib_ratio)))
        selected = set(rng.sample(range(len(dataset.rows)), min(keep, len(dataset.rows) - 1)))

    train_rows = [row for idx, row in enumerate(dataset.rows) if idx not in selected]
    calib_rows = [row for idx, row in enumerate(dataset.rows) if idx in selected]
    train = dataset_from_rows(dataset, train_rows, dataset.synthetic_quality_prob, dataset.seed, "tagging_train")
    calib = dataset_from_rows(dataset, calib_rows, 0.0, dataset.seed + 777, "tagging_calib")
    return train, calib


def collate_tagging(batch: list[dict], processor, vocab: dict[str, list[str]], profile: str) -> dict:
    images = [item["image"] for item in batch]
    proc = processor(text=["<CAPTION>"] * len(images), images=images, return_tensors="pt")
    weights = torch.tensor([float(item.get("weight", 1.0)) for item in batch], dtype=torch.float32)

    quality_labels = vocab.get("quality", [])
    quality_index = {label: i for i, label in enumerate(quality_labels)}
    quality_targets = torch.zeros((len(batch), len(quality_labels)), dtype=torch.float32)
    quality_mask = torch.zeros((len(batch),), dtype=torch.bool)
    targets: dict[str, torch.Tensor] = {"quality": quality_targets}
    masks: dict[str, torch.Tensor] = {"quality": quality_mask}

    for row_idx, item in enumerate(batch):
        if "quality" not in item["mask"]:
            continue
        quality_mask[row_idx] = True
        for label in item["labels"]["quality"]:
            if label in quality_index:
                quality_targets[row_idx, quality_index[label]] = 1.0

    if profile == "quality":
        for ns in CLASSIFICATION_NAMESPACES:
            labels = vocab.get(ns, [])
            index = {label: i for i, label in enumerate(labels)}
            y = torch.full((len(batch),), -100, dtype=torch.long)
            m = torch.zeros((len(batch),), dtype=torch.bool)
            for row_idx, item in enumerate(batch):
                if ns not in item["mask"] or not item["labels"][ns]:
                    continue
                label = item["labels"][ns][0]
                if label in index:
                    y[row_idx] = index[label]
                    m[row_idx] = True
            targets[ns] = y
            masks[ns] = m
        return {"pixel_values": proc["pixel_values"], "targets": targets, "masks": masks, "weights": weights}

    for ns in CLASSIFICATION_NAMESPACES:
        labels = vocab.get(ns, [])
        index = {label: i for i, label in enumerate(labels)}
        y = torch.zeros((len(batch), len(labels)), dtype=torch.float32)
        m = torch.zeros((len(batch),), dtype=torch.bool)
        for row_idx, item in enumerate(batch):
            if ns not in item["mask"]:
                continue
            m[row_idx] = True
            for label in item["labels"][ns]:
                if label in index:
                    y[row_idx, index[label]] = 1.0
        targets[ns] = y
        masks[ns] = m

    return {"pixel_values": proc["pixel_values"], "targets": targets, "masks": masks, "weights": weights}


def compute_pos_weights(dataset: TaggingDataset, vocab: dict[str, list[str]], namespace: str, max_weight: float) -> torch.Tensor:
    labels = vocab.get(namespace, [])
    pos = torch.zeros(len(labels), dtype=torch.float32)
    total = 0
    label_to_idx = {label: i for i, label in enumerate(labels)}
    for row in dataset.rows:
        if namespace not in row["mask"]:
            continue
        total += 1
        for label in row["labels"][namespace]:
            if label in label_to_idx:
                pos[label_to_idx[label]] += 1
    neg = max(total, 1) - pos
    return (neg / pos.clamp_min(1.0)).clamp(1.0, max_weight)


def compute_class_weights(
    dataset: TaggingDataset,
    vocab: dict[str, list[str]],
    namespace: str,
    max_weight: float,
) -> torch.Tensor:
    labels = vocab.get(namespace, [])
    counts = torch.zeros(len(labels), dtype=torch.float32)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    total = 0
    for row in dataset.rows:
        if namespace not in row["mask"] or not row["labels"][namespace]:
            continue
        label = row["labels"][namespace][0]
        if label in label_to_idx:
            counts[label_to_idx[label]] += 1
            total += 1
    if total == 0:
        return torch.ones(len(labels), dtype=torch.float32)
    weights = float(total) / (len(labels) * counts.clamp_min(1.0))
    return weights.clamp(0.25, max_weight)


def weighted_mean(losses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(losses.device)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


def namespace_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    sample_weights: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float,
    label_smoothing: float,
) -> torch.Tensor | None:
    mask = mask.to(logits.device)
    if not bool(mask.any()):
        return None
    target = targets.to(logits.device)[mask]
    if label_smoothing > 0:
        target = target * (1.0 - label_smoothing) + 0.5 * label_smoothing
    logit = logits[mask]
    weights = sample_weights.to(logits.device)
    pos_weight = pos_weight.to(logits.device)
    bce = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_weight, reduction="none")
    if gamma > 0:
        prob = torch.sigmoid(logit)
        pt = prob * target + (1.0 - prob) * (1.0 - target)
        bce = (1.0 - pt).clamp_min(1e-6).pow(gamma) * bce
    per_row = bce.mean(dim=1)
    return weighted_mean(per_row, weights[mask])


def classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    sample_weights: torch.Tensor,
    class_weights: torch.Tensor,
    label_smoothing: float,
) -> torch.Tensor | None:
    mask = mask.to(logits.device)
    if not bool(mask.any()):
        return None
    target = targets.to(logits.device)[mask]
    weights = sample_weights.to(logits.device)
    losses = F.cross_entropy(
        logits[mask],
        target,
        weight=class_weights.to(logits.device),
        label_smoothing=label_smoothing,
        reduction="none",
    )
    return weighted_mean(losses, weights[mask])


def tagging_loss(
    profile: str,
    logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    sample_weights: torch.Tensor,
    pos_weights: dict[str, torch.Tensor],
    class_weights: dict[str, torch.Tensor],
    quality_loss_weight: float,
    scene_loss_weight: float,
    event_loss_weight: float,
    focal_gamma: float,
    label_smoothing: float,
    ce_label_smoothing: float,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    q_loss = namespace_bce_loss(
        logits["quality"],
        targets["quality"],
        masks["quality"],
        sample_weights,
        pos_weights["quality"],
        gamma=focal_gamma,
        label_smoothing=label_smoothing,
    )
    if q_loss is not None:
        parts.append(q_loss * quality_loss_weight)

    for ns, weight in (("scene", scene_loss_weight), ("event", event_loss_weight)):
        if profile == "quality":
            loss = classification_loss(
                logits[ns],
                targets[ns],
                masks[ns],
                sample_weights,
                class_weights[ns],
                label_smoothing=ce_label_smoothing,
            )
        else:
            loss = namespace_bce_loss(
                logits[ns],
                targets[ns],
                masks[ns],
                sample_weights,
                pos_weights[ns],
                gamma=focal_gamma,
                label_smoothing=label_smoothing,
            )
        if loss is not None:
            parts.append(loss * weight)

    if not parts:
        return sum(x.sum() for x in logits.values()) * 0.0
    return torch.stack(parts).sum() / max(quality_loss_weight + scene_loss_weight + event_loss_weight, 1e-6)


@torch.no_grad()
def collect_calibration_probs(model, heads, loader, device, dtype, profile: str) -> dict[str, dict[str, torch.Tensor]]:
    heads.eval()
    namespaces = ("quality",) if profile == "quality" else NAMESPACES
    packed = {ns: {"probs": [], "targets": [], "masks": []} for ns in namespaces}
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
        tokens = encode_image_tokens(model, pixel_values).float()
        logits = heads(tokens)
        for ns in namespaces:
            packed[ns]["probs"].append(torch.sigmoid(logits[ns]).cpu())
            packed[ns]["targets"].append(batch["targets"][ns].cpu())
            packed[ns]["masks"].append(batch["masks"][ns].cpu())
    return {
        ns: {
            "probs": torch.cat(items["probs"], dim=0),
            "targets": torch.cat(items["targets"], dim=0),
            "masks": torch.cat(items["masks"], dim=0),
        }
        for ns, items in packed.items()
    }


def tune_thresholds(
    packed: dict[str, dict[str, torch.Tensor]],
    vocab: dict[str, list[str]],
    min_threshold: float,
    max_threshold: float,
) -> dict[str, dict[str, float]]:
    thresholds = default_thresholds(vocab, 0.5)
    start = int(round(min_threshold * 100))
    end = int(round(max_threshold * 100))
    grid = [x / 100.0 for x in range(start, end + 1, 2)]
    for ns in packed:
        ns_pack = packed[ns]
        y = ns_pack["targets"]
        mask = ns_pack["masks"]
        if not bool(mask.any()):
            continue
        y = y[mask]
        p = ns_pack["probs"][mask]
        for idx, label in enumerate(vocab.get(ns, [])):
            true = y[:, idx].bool()
            if not bool(true.any()):
                continue
            best_t, best_f1 = 0.5, -1.0
            for t in grid:
                pred = p[:, idx] >= t
                tp = int((pred & true).sum())
                fp = int((pred & ~true).sum())
                fn = int((~pred & true).sum())
                denom = 2 * tp + fp + fn
                f1 = (2 * tp / denom) if denom else 0.0
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            thresholds[ns][label] = best_t
    return thresholds


def make_optimizer(name: str, params, lr: float, weight_decay: float):
    if name == "lion":
        return Lion(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=torch.cuda.is_available())
    raise ValueError(f"unknown optimizer: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--profile", choices=TRAINING_PROFILES, default="main")
    parser.add_argument("--train-jsonl", type=Path, default=repo / "training_runs" / "data" / "tagging_train.jsonl")
    parser.add_argument("--pseudo-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--pseudo-weight", type=float, default=0.7)
    parser.add_argument("--calib-jsonl", type=Path)
    parser.add_argument("--calib-ratio", type=float, default=0.12)
    parser.add_argument("--max-calib-rows", type=int, default=12000)
    parser.add_argument("--min-calib-per-namespace", type=int, default=128)
    parser.add_argument("--weight-path", type=str, default=str(repo / "model"))
    parser.add_argument("--vocab", type=Path, default=repo / "model" / "task_a_vocab.json")
    parser.add_argument("--init-heads", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--optimizer", choices=("adamw", "lion"), default="adamw")
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--trunk-dim", type=int, default=768)
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--ce-label-smoothing", type=float, default=0.05)
    parser.add_argument("--quality-loss-weight", type=float, default=1.0)
    parser.add_argument("--scene-loss-weight", type=float, default=1.0)
    parser.add_argument("--event-loss-weight", type=float, default=1.5)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--synthetic-quality-prob", type=float, default=0.25)
    parser.add_argument("--max-pos-weight", type=float, default=8.0)
    parser.add_argument("--max-class-weight", type=float, default=6.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.85)
    args = parser.parse_args()
    if args.out is None:
        args.out = repo / "training_runs" / f"tagging_{args.profile}" / "tagging_heads.pt"

    set_seed(args.seed)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    is_dist = world_size > 1
    if is_dist:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            args.device = "cpu"
        device = torch.device(args.device)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype if device.type == "cuda" else "float32"]

    vocab = load_vocab(args.vocab)
    sources: list[tuple[Path, float]] = [(args.train_jsonl, 1.0)]
    sources.extend((path, args.pseudo_weight) for path in args.pseudo_jsonl)
    full_train_ds = TaggingDataset(
        sources,
        vocab,
        max_rows=args.max_train_rows,
        synthetic_quality_prob=args.synthetic_quality_prob,
        seed=args.seed,
        profile=args.profile,
    )
    if args.calib_jsonl:
        train_ds = full_train_ds
        calib_ds = TaggingDataset(
            args.calib_jsonl,
            vocab,
            max_rows=args.max_val_rows,
            synthetic_quality_prob=0.0,
            seed=args.seed + 999,
            profile=args.profile,
        )
    else:
        train_ds, calib_ds = split_train_calib(
            full_train_ds,
            calib_ratio=args.calib_ratio,
            max_calib_rows=args.max_calib_rows,
            min_calib_per_namespace=args.min_calib_per_namespace,
            seed=args.seed + 123,
        )

    processor = AutoProcessor.from_pretrained(args.weight_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.weight_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    hidden_dim = int(getattr(model.config.vision_config, "projection_dim", 768))
    if args.init_heads:
        heads, _meta = load_tagging_checkpoint(args.init_heads, vocab=vocab, map_location="cpu")
        if int(heads.hidden_dim) != hidden_dim:
            raise RuntimeError(
                f"--init-heads hidden_dim={heads.hidden_dim} does not match model hidden_dim={hidden_dim}"
            )
        heads = heads.to(device)
        if rank == 0:
            print(f"initialized tagging heads from {args.init_heads}", flush=True)
    else:
        heads = TaggingHeads(hidden_dim=hidden_dim, vocab=vocab, trunk_dim=args.trunk_dim, dropout=args.dropout).to(device)
    heads_ddp = DDP(heads, device_ids=[local_rank], find_unused_parameters=False) if is_dist else heads
    optimizer = make_optimizer(args.optimizer, heads.parameters(), args.lr, args.weight_decay)
    pos_namespaces = ("quality",) if args.profile == "quality" else NAMESPACES
    pos_weights = {ns: compute_pos_weights(train_ds, vocab, ns, args.max_pos_weight) for ns in pos_namespaces}
    class_weights = (
        {ns: compute_class_weights(train_ds, vocab, ns, args.max_class_weight) for ns in CLASSIFICATION_NAMESPACES}
        if args.profile == "quality"
        else {}
    )
    if rank == 0:
        for ns, weights in pos_weights.items():
            print(f"{ns}_pos_weight={weights.tolist()}", flush=True)
        for ns, weights in class_weights.items():
            print(f"{ns}_class_weight={weights.tolist()}", flush=True)

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        if is_dist else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda b: collate_tagging(b, processor, vocab, args.profile),
    )
    total_steps = max(1, math.ceil(len(train_loader) * args.epochs))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    t0 = time.time()
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        heads_ddp.train()
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
            with torch.no_grad():
                tokens = encode_image_tokens(model, pixel_values).float()
            logits = heads_ddp(tokens)
            loss = tagging_loss(
                args.profile,
                logits,
                batch["targets"],
                batch["masks"],
                batch["weights"],
                pos_weights,
                class_weights,
                quality_loss_weight=args.quality_loss_weight,
                scene_loss_weight=args.scene_loss_weight,
                event_loss_weight=args.event_loss_weight,
                focal_gamma=args.focal_gamma,
                label_smoothing=args.label_smoothing,
                ce_label_smoothing=args.ce_label_smoothing,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach().cpu())
            steps += 1
        if is_dist:
            loss_t = torch.tensor([total_loss, float(steps)], device=device)
            dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
            avg_loss = float(loss_t[0]) / max(float(loss_t[1]), 1.0)
        else:
            avg_loss = total_loss / max(steps, 1)
        if rank == 0:
            elapsed = (time.time() - t0) / 60.0
            print(
                f"epoch {epoch + 1}/{args.epochs} loss={avg_loss:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} elapsed={elapsed:.1f}m",
                flush=True,
            )

    thresholds = default_thresholds(vocab, 0.5)
    if calib_ds is not None and rank == 0:
        calib_loader = DataLoader(
            calib_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=lambda b: collate_tagging(b, processor, vocab, args.profile),
        )
        packed = collect_calibration_probs(model, heads, calib_loader, device, dtype, args.profile)
        thresholds = tune_thresholds(packed, vocab, args.threshold_min, args.threshold_max)
        source = str(args.calib_jsonl) if args.calib_jsonl else "train split"
        print(f"thresholds tuned on {source}", flush=True)

    if rank != 0:
        if is_dist:
            dist.barrier()
            dist.destroy_process_group()
        return

    save_tagging_checkpoint(
        args.out,
        heads,
        thresholds=thresholds,
        extra={
            "profile": args.profile,
            "train_jsonl": str(args.train_jsonl),
            "pseudo_jsonl": [str(path) for path in args.pseudo_jsonl],
            "pseudo_weight": args.pseudo_weight,
            "calib_jsonl": str(args.calib_jsonl) if args.calib_jsonl else None,
            "calib_ratio": args.calib_ratio,
            "max_calib_rows": args.max_calib_rows,
            "epochs": args.epochs,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "focal_gamma": args.focal_gamma,
            "label_smoothing": args.label_smoothing,
            "ce_label_smoothing": args.ce_label_smoothing,
            "quality_loss_weight": args.quality_loss_weight,
            "scene_loss_weight": args.scene_loss_weight,
            "event_loss_weight": args.event_loss_weight,
            "synthetic_quality_prob": args.synthetic_quality_prob,
        },
    )
    print(f"saved {args.out}", flush=True)
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
