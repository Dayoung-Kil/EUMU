"""Search per-label tagging thresholds on a validation split.

This script is used after tagging head training. It does not update model
weights. It runs the frozen Florence backbone and trained tagging heads on a
local validation split, then writes a JSON summary with tuned thresholds.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from common import load_vocab
from tagging_heads import encode_image_tokens, load_tagging_checkpoint


def image_path_for(root: Path, image_id: str) -> Path:
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        path = root / f"{image_id}{ext}"
        if path.exists():
            return path
    hits = list(root.glob(f"{image_id}.*"))
    if hits:
        return hits[0]
    raise FileNotFoundError(root / image_id)


def load_subset_rows(val_root: Path, subset: str, namespace: str) -> list[dict[str, Any]]:
    gt = json.loads((val_root / subset / "gt.json").read_text())
    image_root = val_root / subset / "images"
    rows = []
    for image_id, entry in gt.items():
        rows.append(
            {
                "image_id": image_id,
                "image": image_path_for(image_root, image_id),
                "labels": list(entry.get(namespace, [])),
            }
        )
    return rows


@torch.no_grad()
def collect_probs(
    rows: list[dict[str, Any]],
    namespace: str,
    processor,
    model,
    heads,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> torch.Tensor:
    packed = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        images = [Image.open(row["image"]).convert("RGB") for row in batch]
        inputs = processor(
            text=["<CAPTION>"] * len(images),
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(device)
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
        tokens = encode_image_tokens(model, inputs["pixel_values"]).float()
        logits = heads(tokens)
        packed.append(torch.sigmoid(logits[namespace]).cpu())
        print(f"collected {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return torch.cat(packed, dim=0)


def f1_macro(pred_sets: list[list[str]], true_sets: list[list[str]], labels: list[str]) -> tuple[float, dict[str, float]]:
    per_label = {}
    for label in labels:
        tp = fp = fn = 0
        for pred, true in zip(pred_sets, true_sets):
            p = label in pred
            t = label in true
            if p and t:
                tp += 1
            elif p and not t:
                fp += 1
            elif not p and t:
                fn += 1
        denom = 2 * tp + fp + fn
        per_label[label] = (2 * tp / denom) if denom else 0.0
    return sum(per_label.values()) / max(len(labels), 1), per_label


def predict_from_thresholds(
    probs: torch.Tensor,
    labels: list[str],
    thresholds: dict[str, float],
    max_labels: int,
    fallback_top_k: int,
) -> list[list[str]]:
    out = []
    for p_tensor in probs:
        values = [float(x) for x in p_tensor.tolist()]
        selected = [label for idx, label in enumerate(labels) if values[idx] >= thresholds.get(label, 0.5)]
        if not selected and fallback_top_k > 0:
            order = sorted(range(len(labels)), key=lambda idx: values[idx], reverse=True)
            selected = [labels[idx] for idx in order[:fallback_top_k]]
        if max_labels > 0 and len(selected) > max_labels:
            selected = sorted(selected, key=lambda label: values[labels.index(label)], reverse=True)[:max_labels]
        out.append(selected)
    return out


def coordinate_tune(
    probs: torch.Tensor,
    labels: list[str],
    true_sets: list[list[str]],
    initial: dict[str, float],
    grid: list[float],
    max_labels: int,
    fallback_top_k: int,
    rounds: int,
) -> tuple[dict[str, float], float, dict[str, float], list[dict[str, Any]]]:
    thresholds = dict(initial)
    pred_sets = predict_from_thresholds(probs, labels, thresholds, max_labels, fallback_top_k)
    best_macro, best_per = f1_macro(pred_sets, true_sets, labels)
    trace: list[dict[str, Any]] = []
    initial_macro = best_macro
    print(f"initial macro={initial_macro:.6f}", flush=True)
    for round_idx in range(rounds):
        changed = False
        for label in labels:
            label_best = (best_macro, thresholds[label], best_per)
            for value in grid:
                trial = dict(thresholds)
                trial[label] = value
                pred_sets = predict_from_thresholds(probs, labels, trial, max_labels, fallback_top_k)
                macro, per = f1_macro(pred_sets, true_sets, labels)
                if macro > label_best[0] + 1e-12:
                    label_best = (macro, value, per)
            if label_best[1] != thresholds[label]:
                thresholds[label] = label_best[1]
                best_macro, best_per = label_best[0], label_best[2]
                changed = True
                item = {"label": label, "threshold": thresholds[label], "macro_f1": best_macro}
                trace.append(item)
                print(f"  {label}: threshold={thresholds[label]:.2f} macro={best_macro:.6f}", flush=True)
        print(f"round {round_idx + 1}/{rounds} macro={best_macro:.6f}", flush=True)
        if not changed:
            break
    return thresholds, initial_macro, best_macro, best_per, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--weight-path", type=Path, default=repo / "model")
    parser.add_argument("--config-path", type=Path, default=repo / "model" / "config.yaml")
    parser.add_argument("--head-in", type=Path, default=repo / "training_runs" / "final_tagging" / "tagging_heads.pt")
    parser.add_argument("--vocab", type=Path, default=repo / "model" / "task_a_vocab.json")
    parser.add_argument("--val-root", type=Path, default=repo / "data" / "val_data")
    parser.add_argument("--namespace", choices=("quality", "scene", "event"), required=True)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-labels", type=int, default=2)
    parser.add_argument("--fallback-top-k", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--grid-min", type=float, default=0.01)
    parser.add_argument("--grid-max", type=float, default=0.99)
    parser.add_argument("--grid-step", type=float, default=0.01)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config_path.read_text()) or {}
    infer_cfg = cfg.get("inference", {}) or {}
    device = torch.device("cuda" if torch.cuda.is_available() and str(infer_cfg.get("device", "cuda")).startswith("cuda") else "cpu")
    dtype_name = str(infer_cfg.get("dtype", "float16"))
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(
        dtype_name if device.type == "cuda" else "float32",
        torch.float32,
    )

    subset = args.subset or args.namespace
    vocab = load_vocab(args.vocab)
    labels = vocab[args.namespace]
    heads, meta = load_tagging_checkpoint(args.head_in, vocab=vocab, map_location=device)
    heads = heads.to(device).eval()
    thresholds = meta.get("thresholds") or {}
    initial = {
        label: float((thresholds.get(args.namespace) or {}).get(label, 0.5))
        for label in labels
    }

    rows = load_subset_rows(args.val_root, subset, args.namespace)
    true_sets = [row["labels"] for row in rows]
    processor = AutoProcessor.from_pretrained(str(args.weight_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.weight_path),
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device).eval()
    probs = collect_probs(rows, args.namespace, processor, model, heads, device, dtype, args.batch_size)
    grid_steps = int(round((args.grid_max - args.grid_min) / args.grid_step))
    grid = [round(args.grid_min + i * args.grid_step, 4) for i in range(grid_steps + 1)]
    best, initial_macro, best_macro, per_label, trace = coordinate_tune(
        probs,
        labels,
        true_sets,
        initial,
        grid,
        max_labels=args.max_labels,
        fallback_top_k=args.fallback_top_k,
        rounds=args.rounds,
    )
    pred_sets = predict_from_thresholds(probs, labels, best, args.max_labels, args.fallback_top_k)
    summary = {
        "namespace": args.namespace,
        "subset": subset,
        "initial_macro_f1": initial_macro,
        "best_macro_f1": best_macro,
        "max_labels": args.max_labels,
        "fallback_top_k": args.fallback_top_k,
        "thresholds": best,
        "per_label_f1": per_label,
        "pred_counts": dict(Counter(label for row in pred_sets for label in row)),
        "trace": trace,
    }
    out = args.out or (repo / "training_runs" / "threshold_search" / f"{args.namespace}_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
