"""Integrated multi-label tagging heads for EUMU."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from common import NAMESPACES

HEAD_TYPE = "task_a_multilabel_bce_v7"
LEGACY_HEAD_TYPES = {"quality_bce_scene_event_ce"}

SCENE_PARENT: dict[str, str] = {
    "airport": "indoor",
    "bathroom": "indoor",
    "bedroom": "indoor",
    "classroom": "indoor",
    "factory": "indoor",
    "hotel": "indoor",
    "kitchen": "indoor",
    "living_room": "indoor",
    "museum": "indoor",
    "office": "indoor",
    "restaurant": "indoor",
    "shopping_mall": "indoor",
    "stage": "indoor",
    "supermarket": "indoor",
    "train_station": "indoor",
    "beach": "outdoor",
    "bridge": "outdoor",
    "coast": "outdoor",
    "desert": "outdoor",
    "farm": "outdoor",
    "forest": "outdoor",
    "garden": "outdoor",
    "highway": "outdoor",
    "lake": "outdoor",
    "mountain": "outdoor",
    "ocean": "outdoor",
    "park": "outdoor",
    "parking_lot": "outdoor",
    "river": "outdoor",
    "rooftop": "outdoor",
    "snow": "outdoor",
    "stadium": "outdoor",
    "street": "outdoor",
}


class AttentionPool(nn.Module):
    """Pool Florence visual tokens with mean, max, and learned attention."""

    def __init__(self, hidden_dim: int, attn_dim: int = 256) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.GELU(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise RuntimeError(f"expected image tokens [B,N,C], got {tuple(tokens.shape)}")
        x = self.norm(tokens.float())
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1).unsqueeze(-1)
        attn = (x * weights).sum(dim=1)
        mean = x.mean(dim=1)
        maxv = x.max(dim=1).values
        return torch.cat([attn, mean, maxv], dim=-1)


def make_mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, max(128, hidden_dim // 4)),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(max(128, hidden_dim // 4), out_dim),
    )


class TaggingHeads(nn.Module):
    """All tagging namespaces are emitted as multi-label tag logits."""

    def __init__(
        self,
        hidden_dim: int,
        vocab: dict[str, list[str]],
        trunk_dim: int = 768,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.trunk_dim = int(trunk_dim)
        self.vocab = {ns: list(vocab.get(ns, [])) for ns in NAMESPACES}
        self.head_type = HEAD_TYPE
        self.pool = AttentionPool(self.hidden_dim)
        pooled_dim = self.hidden_dim * 3
        self.quality = make_mlp(pooled_dim, self.trunk_dim, len(self.vocab["quality"]), dropout)
        self.scene = make_mlp(pooled_dim, self.trunk_dim, len(self.vocab["scene"]), dropout)
        self.event = make_mlp(pooled_dim, self.trunk_dim, len(self.vocab["event"]), dropout)

    def forward(self, image_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = self.pool(image_tokens)
        return {
            "quality": self.quality(pooled),
            "scene": self.scene(pooled),
            "event": self.event(pooled),
        }


def get_florence_core(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


@torch.no_grad()
def encode_image_tokens(model: torch.nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    core = get_florence_core(model)
    if not hasattr(core, "_encode_image"):
        raise AttributeError("Florence model does not expose _encode_image")
    tokens = core._encode_image(pixel_values)
    if tokens.ndim != 3:
        raise RuntimeError(f"expected image features [B,N,C], got shape {tuple(tokens.shape)}")
    return tokens


def default_thresholds(vocab: dict[str, list[str]], value: float = 0.5) -> dict[str, dict[str, float]]:
    return {ns: {label: float(value) for label in vocab.get(ns, [])} for ns in NAMESPACES}


def _threshold_for(
    thresholds: dict[str, Any] | None,
    namespace: str,
    label: str,
    default: float,
) -> float:
    if not thresholds:
        return default
    value = thresholds.get(namespace)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        item = value.get(label, value.get("_default", default))
        if isinstance(item, (int, float)):
            return float(item)
    return default


def _topk_sigmoid_labels(logits: torch.Tensor, labels: list[str], k: int, min_prob: float) -> list[str]:
    if not labels or k <= 0:
        return []
    probs = torch.sigmoid(logits.detach().float().cpu())[0]
    values, indices = torch.topk(probs, k=min(k, len(labels)))
    out = []
    for value, idx in zip(values.tolist(), indices.tolist()):
        if float(value) >= min_prob:
            out.append(labels[int(idx)])
    if not out and labels:
        out.append(labels[int(indices[0])])
    return out


def _topk_softmax_labels(logits: torch.Tensor, labels: list[str], k: int, min_prob: float) -> list[str]:
    if not labels or k <= 0:
        return []
    probs = torch.softmax(logits.detach().float().cpu(), dim=-1)[0]
    values, indices = torch.topk(probs, k=min(k, len(labels)))
    out = []
    for value, idx in zip(values.tolist(), indices.tolist()):
        if float(value) >= min_prob:
            out.append(labels[int(idx)])
    if not out and labels:
        out.append(labels[int(indices[0])])
    return out


def _cap_labels(selected: list[str], probs: torch.Tensor, labels: list[str], max_labels: int) -> list[str]:
    if max_labels <= 0 or len(selected) <= max_labels:
        return selected
    label_to_prob = {label: float(probs[idx]) for idx, label in enumerate(labels)}
    return sorted(selected, key=lambda label: label_to_prob.get(label, 0.0), reverse=True)[:max_labels]


def _thresholded_labels(
    logits: torch.Tensor,
    vocab: dict[str, list[str]],
    thresholds: dict[str, Any] | None,
    namespace: str,
    default_threshold: float,
    fallback_top_k: int,
    fallback_min_prob: float,
    max_labels: int,
) -> list[str]:
    labels = vocab.get(namespace, [])
    if not labels:
        return []
    probs = torch.sigmoid(logits.detach().float().cpu())[0]
    selected: list[str] = []
    for idx, label in enumerate(labels):
        threshold = _threshold_for(thresholds, namespace, label, default_threshold)
        if float(probs[idx]) >= threshold:
            selected.append(label)
    if not selected and fallback_top_k > 0:
        selected = _topk_sigmoid_labels(logits, labels, fallback_top_k, fallback_min_prob)
    return _cap_labels(selected, probs, labels, max_labels)


def _add_scene_parents(selected: list[str], logits: torch.Tensor, vocab: dict[str, list[str]], max_labels: int) -> list[str]:
    labels = vocab.get("scene", [])
    if not selected or not labels:
        return selected
    probs = torch.sigmoid(logits.detach().float().cpu())[0]
    label_to_prob = {label: float(probs[idx]) for idx, label in enumerate(labels)}
    out = list(selected)
    for label in selected:
        parent = SCENE_PARENT.get(label)
        if parent and parent in labels and parent not in out:
            out.append(parent)
    if max_labels <= 0 or len(out) <= max_labels:
        return out
    parents = [label for label in ("indoor", "outdoor") if label in out]
    specifics = [label for label in out if label not in {"indoor", "outdoor"}]
    specifics = sorted(specifics, key=lambda label: label_to_prob.get(label, 0.0), reverse=True)
    capped = specifics[: max(0, max_labels - len(parents))] + parents
    return capped[:max_labels]


def _drop_scene_parent_when_specific(
    selected: list[str],
    logits: torch.Tensor,
    vocab: dict[str, list[str]],
    max_specific_labels: int,
) -> list[str]:
    if not selected:
        return selected
    labels = vocab.get("scene", [])
    probs = torch.sigmoid(logits.detach().float().cpu())[0]
    label_to_prob = {label: float(probs[idx]) for idx, label in enumerate(labels)}
    specifics = [label for label in selected if label not in {"indoor", "outdoor"}]
    if not specifics:
        return selected[:1]
    specifics = sorted(specifics, key=lambda label: label_to_prob.get(label, 0.0), reverse=True)
    return specifics[: max(1, max_specific_labels)]


def predict_tags_from_logits(
    logits: dict[str, torch.Tensor],
    vocab: dict[str, list[str]],
    thresholds: dict[str, Any] | None = None,
    default_threshold: float = 0.5,
    quality_default_good: bool = True,
    scene_top_k: int = 1,
    scene_min_prob: float = 0.0,
    scene_max_labels: int = 1,
    scene_specific_max_labels: int = 1,
    event_top_k: int = 1,
    event_min_prob: float = 0.0,
    event_max_labels: int = 1,
    quality_max_labels: int = 2,
    legacy_scene_event_softmax: bool = False,
) -> dict[str, list[str]]:
    out = {ns: [] for ns in NAMESPACES}

    quality_labels = vocab.get("quality", [])
    quality_probs = torch.sigmoid(logits["quality"]).detach().float().cpu()[0]
    for idx, label in enumerate(quality_labels):
        prob = float(quality_probs[idx])
        thresh = _threshold_for(thresholds, "quality", label, default_threshold)
        if prob >= thresh:
            out["quality"].append(label)

    defects = [label for label in out["quality"] if label != "quality_good"]
    if defects:
        out["quality"] = _cap_labels(defects, quality_probs, quality_labels, quality_max_labels)
    elif quality_default_good and "quality_good" in quality_labels:
        out["quality"] = ["quality_good"]

    if legacy_scene_event_softmax:
        out["scene"] = _topk_softmax_labels(
            logits["scene"],
            vocab.get("scene", []),
            min(scene_top_k, scene_max_labels) if scene_max_labels > 0 else scene_top_k,
            scene_min_prob,
        )
        out["event"] = _topk_softmax_labels(
            logits["event"],
            vocab.get("event", []),
            min(event_top_k, event_max_labels) if event_max_labels > 0 else event_top_k,
            event_min_prob,
        )
    else:
        out["scene"] = _thresholded_labels(
            logits["scene"],
            vocab,
            thresholds,
            "scene",
            scene_min_prob if scene_min_prob > 0 else default_threshold,
            scene_top_k,
            scene_min_prob,
            scene_max_labels,
        )
        out["scene"] = _drop_scene_parent_when_specific(
            out["scene"],
            logits["scene"],
            vocab,
            scene_specific_max_labels,
        )
        out["event"] = _thresholded_labels(
            logits["event"],
            vocab,
            thresholds,
            "event",
            event_min_prob if event_min_prob > 0 else default_threshold,
            event_top_k,
            event_min_prob,
            event_max_labels,
        )
    return out


def save_tagging_checkpoint(
    path: str | Path,
    heads: TaggingHeads,
    thresholds: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "state_dict": heads.state_dict(),
        "hidden_dim": heads.hidden_dim,
        "trunk_dim": heads.trunk_dim,
        "vocab": heads.vocab,
        "head_type": HEAD_TYPE,
        "thresholds": thresholds or default_thresholds(heads.vocab),
    }
    if extra:
        payload["extra"] = extra
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_tagging_checkpoint(
    path: str | Path,
    vocab: dict[str, list[str]] | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[TaggingHeads, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location)
    if payload.get("head_type") not in (None, HEAD_TYPE, *LEGACY_HEAD_TYPES):
        raise ValueError(f"unsupported tagging head_type={payload.get('head_type')!r}")
    ckpt_vocab = payload.get("vocab")
    if vocab is None:
        vocab = ckpt_vocab
    if vocab is None:
        raise ValueError(f"tagging checkpoint has no vocab: {path}")
    heads = TaggingHeads(
        hidden_dim=int(payload.get("hidden_dim", 768)),
        trunk_dim=int(payload.get("trunk_dim", 768)),
        vocab=vocab,
    )
    heads.load_state_dict(payload["state_dict"], strict=True)
    meta = {
        "thresholds": payload.get("thresholds") or default_thresholds(vocab),
        "extra": payload.get("extra", {}),
        "head_type": payload.get("head_type", HEAD_TYPE),
    }
    return heads, meta
