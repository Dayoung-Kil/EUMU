"""Self-contained EUMU inference API for the MUMU Challenge 2026.

Required public API:

    predictor = EUMUPredictor(weight_path="model", config_path="model/config.yaml")
    output = predictor.predict("/path/to/image.jpg")

The predictor is one unified package:
  - Microsoft Florence-2-base shared model.
  - Integrated attention-pooling tagging heads loaded from model/task_a_heads.pt.
  - Florence decoder for object detection and captioning.
"""
from __future__ import annotations

import json
import math
import copy
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml
from PIL import Image, ImageStat
from torch import nn
from transformers import AutoModelForCausalLM, AutoProcessor

GENERIC_DETECTION_PHRASE_LABELS = {
    "human",
    "person",
    "man",
    "woman",
    "boy",
    "girl",
    "table",
    "plate",
    "cup",
    "shirt",
    "group",
    "object",
    "thing",
    "image",
    "photo",
    "picture",
    "part",
    "piece",
}

NAMESPACES = ("quality", "scene", "event")
HEAD_TYPE = "task_a_multilabel_bce"
LEGACY_HEAD_TYPES = {"task_a_multilabel_bce_v7", "quality_bce_scene_event_ce"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EVAL_TOKEN_RE = re.compile(r"[\w][\w'\-]*", re.UNICODE)
WORD_RE = re.compile(r"[a-z0-9]+")
SPACE_RE = re.compile(r"\s+")


def configure_deterministic_runtime(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
QUALITY_GOOD_BLOCKERS = {
    "blur",
    "compression_artifacts",
    "low_light",
    "motion_blur",
    "noise",
    "overexposure",
    "underexposure",
}
CAPTION_BAD_TAIL_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "around",
    "as",
    "at",
    "be",
    "been",
    "being",
    "beside",
    "behind",
    "by",
    "covered",
    "for",
    "from",
    "had",
    "has",
    "have",
    "holding",
    "in",
    "into",
    "is",
    "located",
    "looking",
    "near",
    "of",
    "on",
    "or",
    "over",
    "placed",
    "playing",
    "riding",
    "sitting",
    "showing",
    "standing",
    "surrounded",
    "the",
    "to",
    "under",
    "using",
    "walking",
    "was",
    "wearing",
    "were",
    "with",
    "without",
}

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


def load_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text())


def load_vocab(path: str | Path) -> dict[str, list[str]]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"vocab must be a JSON object: {path}")
    return {ns: list(raw.get(ns, [])) for ns in NAMESPACES}


def evaluator_tokens(text: str) -> list[str]:
    normalised = unicodedata.normalize("NFKC", str(text or "")).lower()
    return EVAL_TOKEN_RE.findall(normalised)


def repair_caption_tail(text: str, min_tokens: int = 6) -> str:
    words = str(text or "").strip(" ,;:.!?").split()
    while len(words) > min_tokens:
        tail = re.sub(r"[^\w'\-]+", "", words[-1].lower())
        if tail not in CAPTION_BAD_TAIL_TOKENS:
            break
        words.pop()
    return " ".join(words).strip(" ,;:")


def sanitize_caption(text: str, char_limit: int = 300, token_limit: int = 30) -> str:
    text = " ".join(str(text or "").replace("\n", " ").replace("\r", " ").split())
    if not text:
        return "an image"
    if len(text) > char_limit:
        text = text[:char_limit].rstrip(" ,;:")
    while len(evaluator_tokens(text)) > token_limit:
        parts = text.rsplit(" ", 1)
        if len(parts) < 2:
            break
        text = parts[0].rstrip(" ,;:")
    text = text.strip()
    if not text:
        return "an image"
    text = repair_caption_tail(text)
    if not text:
        return "an image"
    if text[-1] not in ".!?":
        if len(text) >= char_limit:
            text = text[: max(1, char_limit - 1)].rstrip(" ,;:")
        text += "."
    return text


def normalise_label_text(label: str) -> str:
    text = unicodedata.normalize("NFKC", str(label or "")).lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 /]+", " ", text)
    text = SPACE_RE.sub(" ", text).strip()
    for article in ("a ", "an ", "the "):
        if text.startswith(article):
            text = text[len(article):].strip()
            break
    return text


def normalise_alias_value(label: str) -> str:
    return " ".join(str(label or "").replace("\n", " ").replace("\r", " ").lower().split())


def load_label_map(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    raw = load_json(p)
    if not isinstance(raw, dict):
        raise ValueError(f"label map must be a JSON object: {p}")
    out: dict[str, str] = {}
    for key, value in raw.items():
        k = normalise_label_text(str(key))
        v = normalise_alias_value(str(value))
        if k and v:
            out[k] = v
            out.setdefault(normalise_label_text(v), v)
    return out


def canonicalize_label(label: str, label_map: dict[str, str] | None = None) -> str:
    clean = normalise_label_text(label)
    if label_map and clean in label_map:
        return label_map[clean]
    return clean


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def nms_detections(detections: list[dict], iou_thresh: float = 0.65) -> list[dict]:
    by_label: dict[str, list[dict]] = {}
    for det in detections:
        by_label.setdefault(det["label"], []).append(det)
    def sort_key(det: dict) -> tuple:
        bbox = det.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
        return (
            -float(det.get("score", 0.0)),
            str(det.get("label", "")),
            tuple(round(float(x), 4) for x in bbox),
        )
    kept: list[dict] = []
    for items in by_label.values():
        local: list[dict] = []
        for det in sorted(items, key=sort_key):
            if all(bbox_iou(det["bbox_xyxy"], prev["bbox_xyxy"]) <= iou_thresh for prev in local):
                local.append(det)
        kept.extend(local)
    kept.sort(key=sort_key)
    return kept


def scale_detection_bboxes(
    detections: list[dict],
    image_width: int,
    image_height: int,
    bbox_scale_by_label: dict[str, float] | None = None,
) -> list[dict]:
    if not bbox_scale_by_label:
        return detections
    w, h = float(image_width), float(image_height)
    out: list[dict] = []
    for det in detections:
        factor = float(bbox_scale_by_label.get(str(det.get("label", "")), 1.0))
        if not math.isfinite(factor) or factor <= 0.0 or abs(factor - 1.0) < 1e-9:
            out.append(det)
            continue
        x1, y1, x2, y2 = (float(v) for v in det["bbox_xyxy"])
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        bw = (x2 - x1) * factor
        bh = (y2 - y1) * factor
        scaled = [
            max(0.0, min(w, cx - bw * 0.5)),
            max(0.0, min(h, cy - bh * 0.5)),
            max(0.0, min(w, cx + bw * 0.5)),
            max(0.0, min(h, cy + bh * 0.5)),
        ]
        if scaled[2] <= scaled[0] or scaled[3] <= scaled[1]:
            out.append(det)
            continue
        updated = dict(det)
        updated["bbox_xyxy"] = scaled
        out.append(updated)
    return out


def normalise_detections(
    detections: list[dict],
    image_width: int,
    image_height: int,
    label_map: dict[str, str] | None = None,
    score_floor: float = 0.0,
    iou_thresh: float = 0.65,
    limit: int = 300,
    bbox_scale_by_label: dict[str, float] | None = None,
) -> list[dict]:
    w, h = float(image_width), float(image_height)
    out: list[dict] = []
    for det in detections:
        label = canonicalize_label(str(det.get("label", "")), label_map)
        if not label:
            continue
        bbox = det.get("bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            continue
        x1 = max(0.0, min(w, x1))
        x2 = max(0.0, min(w, x2))
        y1 = max(0.0, min(h, y1))
        y2 = max(0.0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        score = det.get("score", 0.0)
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            continue
        score = max(0.0, min(1.0, float(score)))
        if score < score_floor:
            continue
        out.append({"label": label, "score": score, "bbox_xyxy": [x1, y1, x2, y2]})
    out = nms_detections(out, iou_thresh=iou_thresh)
    out = scale_detection_bboxes(out, image_width, image_height, bbox_scale_by_label)
    return out[:limit]


def keyword_tags(text: str, vocab: dict[str, list[str]]) -> dict[str, list[str]]:
    tokens = set(WORD_RE.findall(unicodedata.normalize("NFKC", str(text or "")).lower()))
    out = {ns: [] for ns in NAMESPACES}
    for ns in NAMESPACES:
        for label in vocab.get(ns, []):
            words = str(label).lower().split("_")
            if words and all(word in tokens for word in words):
                out[ns].append(label)
    defects = [x for x in out["quality"] if x != "quality_good"]
    if defects:
        out["quality"] = defects
    elif "quality_good" in vocab.get("quality", []):
        out["quality"] = ["quality_good"]
    return out


def refine_quality_tags_with_image_stats(
    tags: dict[str, list[str]],
    image: Image.Image,
    max_labels: int = 2,
    add_rules: list[dict] | None = None,
    drop_rules: list[dict] | None = None,
    sequence_rules: list[dict] | None = None,
) -> dict[str, list[str]]:
    refined = {ns: list(tags.get(ns, [])) for ns in NAMESPACES}
    gray = image.convert("L").resize((224, 224))
    stat = ImageStat.Stat(gray)
    mean = float(stat.mean[0]) / 255.0
    std = float(stat.stddev[0]) / 255.0
    hist = gray.histogram()
    total = float(gray.width * gray.height)
    dark_ratio = sum(hist[:46]) / total
    highlight_ratio = sum(hist[235:]) / total
    rgb = image.convert("RGB").resize((224, 224))
    rgb_arr = np.asarray(rgb).astype(np.float32) / 255.0
    saturation = rgb_arr.max(axis=2) - rgb_arr.min(axis=2)
    sat_mean = float(saturation.mean())
    sat_std = float(saturation.std())

    stable_labels = {
        "blur",
        "motion_blur",
        "noise",
        "compression_artifacts",
        "quality_good",
        "low_contrast",
        "low_light",
        "underexposure",
        "overexposure",
        "high_contrast",
    }
    quality = [label for label in refined.get("quality", []) if label in stable_labels]
    for label, enabled in (
        ("low_contrast", std < 0.16),
        ("low_light", mean < 0.22),
        ("underexposure", mean < 0.28),
        ("overexposure", mean > 0.70),
        ("high_contrast", std > 0.32),
    ):
        if not enabled or label in quality:
            continue
        quality = [x for x in quality if x != "quality_good"]
        quality.append(label)
        if max_labels > 0 and len(quality) > max_labels:
            quality = quality[-max_labels:]

    if not quality:
        quality = ["quality_good"]
    elif "quality_good" not in quality and not (set(quality) & QUALITY_GOOD_BLOCKERS):
        quality.append("quality_good")
    elif (
        "quality_good" not in quality
        and "blur" not in quality
        and mean >= 0.35
        and dark_ratio <= 0.25
        and highlight_ratio >= 0.05
    ):
        quality.append("quality_good")

    gray_arr = np.asarray(gray).astype(np.float32) / 255.0
    gx = np.zeros_like(gray_arr)
    gy = np.zeros_like(gray_arr)
    gx[:, 1:-1] = (gray_arr[:, 2:] - gray_arr[:, :-2]) * 0.5
    gy[1:-1, :] = (gray_arr[2:, :] - gray_arr[:-2, :]) * 0.5
    grad_std = float(np.sqrt(gx * gx + gy * gy).std())
    grad_aniso = float(abs(gx.var() - gy.var()) / (gx.var() + gy.var() + 1e-8))
    residual_pad = np.pad(gray_arr, 1, mode="edge")
    neighbor_avg = (
        residual_pad[:-2, 1:-1]
        + residual_pad[2:, 1:-1]
        + residual_pad[1:-1, :-2]
        + residual_pad[1:-1, 2:]
    ) / 4.0
    residual = gray_arr - neighbor_avg
    noise_std = float(residual.std())
    noise_mad = float(np.median(np.abs(residual)))
    vertical = np.abs(gray_arr[:, 1:] - gray_arr[:, :-1])
    horizontal = np.abs(gray_arr[1:, :] - gray_arr[:-1, :])
    v_boundary = vertical[:, 7::8]
    h_boundary = horizontal[7::8, :]
    v_mask = np.ones(vertical.shape[1], dtype=bool)
    h_mask = np.ones(horizontal.shape[0], dtype=bool)
    v_mask[7::8] = False
    h_mask[7::8] = False
    block_boundary = float(v_boundary.mean() + h_boundary.mean())
    block_nonboundary = float(vertical[:, v_mask].mean() + horizontal[h_mask, :].mean())
    blockiness = block_boundary - block_nonboundary

    for label, enabled in (
        ("overexposure", noise_std <= 0.017085185423493385),
        ("low_contrast", dark_ratio >= 0.5301299426020408),
        ("underexposure", mean >= 0.7257036149501802),
        ("low_light", grad_aniso >= 0.43856335669755936),
        ("high_contrast", std <= 0.1114960539340973),
        ("blur", grad_std <= 0.021133261639624836),
    ):
        if enabled and label not in quality:
            quality.append(label)

    stats = {
        "mean": mean,
        "std": std,
        "dark_ratio": dark_ratio,
        "highlight_ratio": highlight_ratio,
        "grad_std": grad_std,
        "grad_aniso": grad_aniso,
        "noise_std": noise_std,
        "noise_mad": noise_mad,
        "sat_mean": sat_mean,
        "sat_std": sat_std,
        "blockiness": blockiness,
        "block_boundary": block_boundary,
        "block_nonboundary": block_nonboundary,
    }
    if add_rules:
        for rule in add_rules:
            if not isinstance(rule, dict):
                continue
            label = str(rule.get("label") or "").strip()
            feature = str(rule.get("feature") or rule.get("feat") or "").strip()
            op = str(rule.get("op") or rule.get("dir") or "").strip()
            try:
                threshold = float(rule.get("threshold", rule.get("thr")))
            except (TypeError, ValueError):
                continue
            value = stats.get(feature)
            if label not in stable_labels or value is None or not math.isfinite(value):
                continue
            if (op == "<=" and value <= threshold) or (op == ">=" and value >= threshold):
                if label not in quality:
                    if label != "quality_good":
                        quality = [item for item in quality if item != "quality_good"]
                    quality.append(label)
                    if max_labels > 0 and len(quality) > max_labels:
                        quality = quality[-max_labels:]

    if drop_rules:
        for rule in drop_rules:
            if not isinstance(rule, dict):
                continue
            label = str(rule.get("label") or "").strip()
            feature = str(rule.get("feature") or rule.get("feat") or "").strip()
            op = str(rule.get("op") or rule.get("dir") or "").strip()
            try:
                threshold = float(rule.get("threshold", rule.get("thr")))
            except (TypeError, ValueError):
                continue
            value = stats.get(feature)
            if label not in quality or value is None or not math.isfinite(value):
                continue
            if (op == "<=" and value <= threshold) or (op == ">=" and value >= threshold):
                quality = [item for item in quality if item != label]
        if not quality:
            quality = ["quality_good"]

    if sequence_rules:
        for rule in sequence_rules:
            if not isinstance(rule, dict):
                continue
            action = str(rule.get("action") or rule.get("op_type") or "").strip()
            label = str(rule.get("label") or "").strip()
            feature = str(rule.get("feature") or rule.get("feat") or "").strip()
            op = str(rule.get("op") or rule.get("dir") or "").strip()
            try:
                threshold = float(rule.get("threshold", rule.get("thr")))
            except (TypeError, ValueError):
                continue
            value = stats.get(feature)
            if label not in stable_labels or value is None or not math.isfinite(value):
                continue
            enabled = (op == "<=" and value <= threshold) or (op == ">=" and value >= threshold)
            if not enabled:
                continue
            if action == "add":
                if label not in quality:
                    if label != "quality_good":
                        quality = [item for item in quality if item != "quality_good"]
                    quality.append(label)
                    if max_labels > 0 and len(quality) > max_labels:
                        quality = quality[-max_labels:]
            elif action == "drop" and label in quality:
                quality = [item for item in quality if item != label]
                if not quality:
                    quality = ["quality_good"]

    refined["quality"] = quality
    return refined


def refine_quality_tags_with_image_stats_legacy(
    tags: dict[str, list[str]],
    image: Image.Image,
    max_labels: int = 2,
) -> dict[str, list[str]]:
    refined = {ns: list(tags.get(ns, [])) for ns in NAMESPACES}
    gray = image.convert("L").resize((224, 224))
    stat = ImageStat.Stat(gray)
    mean = float(stat.mean[0]) / 255.0
    std = float(stat.stddev[0]) / 255.0

    stable_labels = {"blur", "quality_good", "low_contrast", "low_light", "high_contrast"}
    quality = [label for label in refined.get("quality", []) if label in stable_labels]
    for label, enabled in (
        ("low_contrast", std < 0.24),
        ("low_light", mean < 0.22),
        ("high_contrast", std > 0.32),
    ):
        if not enabled or label in quality:
            continue
        quality = [x for x in quality if x != "quality_good"]
        quality.append(label)
        if max_labels > 0 and len(quality) > max_labels:
            quality = quality[-max_labels:]

    if not quality:
        quality = ["quality_good"]
    refined["quality"] = quality
    return refined


def _evidence_text(tags: dict[str, list[str]], detections: list[dict], caption: str) -> str:
    parts = [caption]
    for ns in NAMESPACES:
        parts.extend(tags.get(ns, []))
    parts.extend(str(det.get("label", "")) for det in detections)
    return " ".join(parts).lower().replace("_", " ").replace("-", " ")


def _has_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _has_any_term(text: str, terms: list[str]) -> bool:
    for term in terms:
        term = str(term).strip()
        if not term:
            continue
        if " " in term:
            if term in text:
                return True
        elif re.search(rf"\b{re.escape(term)}\b", text):
            return True
    return False


def refine_scene_tags_with_caption_overrides(
    tags: dict[str, list[str]],
    detections: list[dict],
    caption: str,
    policy: dict | None,
) -> dict[str, list[str]]:
    if not policy:
        return tags
    refined = {ns: list(tags.get(ns, [])) for ns in NAMESPACES}
    rules = policy.get("rules") if isinstance(policy, dict) else None
    if not isinstance(rules, list):
        return refined
    text = _evidence_text(refined, detections, caption)
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        label = str(rule.get("label") or "").strip()
        terms = rule.get("terms") or rule.get("phrases") or []
        if not label or not isinstance(terms, list):
            continue
        if _has_any_term(text, [str(term) for term in terms]):
            refined["scene"] = [label]
    return refined


def _sports_event_from_evidence(tags: dict[str, list[str]], detections: list[dict], caption: str) -> bool:
    text = _evidence_text(tags, detections, caption)
    scene = set(tags.get("scene", []))
    det_labels = {str(det.get("label", "")) for det in detections}
    score = 0
    if "stadium" in scene:
        score += 3
    if _has_any(
        text,
        [
            "wrestling",
            "wrestlers",
            "soccer",
            "football",
            "basketball",
            "tennis",
            "volleyball",
            "gymnastics",
            "marathon",
            "skating",
            "skiing",
            "ice rink",
        ],
    ):
        score += 3
    if "baseball player" in text or "baseball bat" in text or "baseball base" in text:
        score += 3
    if _has_any(
        text,
        ["player", "players", "athlete", "running on a track", "riding skis", "skating on a ice rink", "figure skating"],
    ):
        score += 1
    if _has_any(text, [" track", " field", " ring", " rink", " slope", " skis", " ski "]):
        score += 1
    if det_labels & {"baseball_bat", "baseball_base", "ski", "ski_boot", "volleyball", "basketball"}:
        score += 2
    if _has_any(text, ["graduation gown", "graduation gowns", "graduation cap", "cap and gown"]):
        score -= 4
    if "wedding dress" in text:
        score -= 3
    return score >= 3


def _exhibition_from_evidence(tags: dict[str, list[str]], detections: list[dict], caption: str) -> bool:
    text = _evidence_text(tags, detections, caption)
    scene = set(tags.get("scene", []))
    score = 0
    if "museum" in scene:
        score += 2
    if _has_any(text, ["gallery", "exhibit", "exhibition"]):
        score += 3
    if _has_any(
        text,
        [
            "painting",
            "paintings",
            "art ",
            " artwork",
            "poster",
            "display",
            "manuscript",
            "show card",
            "sculpture",
            "statue",
            "clothes rack",
            "neon lights",
            "dress hanging",
        ],
    ):
        score += 2
    if "wall" in text and _has_any(text, ["poster", "painting", "sign", "covered", "hanging", "photo"]):
        score += 1
    if _has_any(text, ["police", "protest", "flag", "banner", "crowd", "racial profiling", "can t breathe"]):
        score -= 3
    if _has_any(text, ["graduation gown", "graduation gowns", "graduation cap", "cap and gown", "pinning a medal"]):
        score -= 3
    if "wedding dress" in text:
        score -= 2
    if _sports_event_from_evidence(tags, detections, caption):
        score -= 2
    return score >= 2


HIDDEN_EVENT_LABELS = {
    "award_ceremony",
    "birthday",
    "celebration",
    "ceremony",
    "cooking",
    "festival",
    "marathon",
    "parade",
    "party",
    "performance",
    "religious_event",
}


def _event_evidence_text_without_event_tags(
    tags: dict[str, list[str]],
    detections: list[dict],
    caption: str,
) -> str:
    parts = [caption]
    parts.extend(tags.get("quality", []))
    parts.extend(tags.get("scene", []))
    parts.extend(str(det.get("label", "")) for det in detections[:50])
    return " ".join(parts).lower().replace("_", " ").replace("-", " ")


def _hidden_events_from_evidence(
    tags: dict[str, list[str]],
    detections: list[dict],
    caption: str,
) -> set[str]:
    text = _event_evidence_text_without_event_tags(tags, detections, caption)
    out: set[str] = set()

    if _has_any(text, ["birthday", "birthday cake", "birthday party", "happy birthday", "blowing out candles"]) or (
        _has_any(text, ["cake"]) and _has_any(text, ["candle", "candles"])
    ):
        out.add("birthday")
    if "wedding" in text:
        out.discard("birthday")

    if _has_any(
        text,
        ["award ceremony", "awards ceremony", "receiving an award", "presenting an award", "medal ceremony"],
    ) or (_has_any(text, ["trophy", "medal", "award"]) and _has_any(text, ["podium", "stage", "ceremony"])):
        out.add("award_ceremony")

    if _has_any(text, ["ribbon cutting", "formal ceremony", "ceremonial procession", "memorial service"]):
        out.add("ceremony")

    if _has_any(
        text,
        [
            "cooking",
            "chef preparing",
            "chef cooking",
            "preparing food",
            "cutting vegetables",
            "chopping vegetables",
            "frying",
            "stove top",
        ],
    ) or (_has_any(text, ["stove", "oven", "frying pan"]) and _has_any(text, ["kitchen", "preparing", "cooking"])):
        out.add("cooking")

    if _has_any(text, ["festival", "carnival", "street fair", "fairground", "outdoor festival", "food festival", "fireworks display"]):
        out.add("festival")

    if _has_any(text, ["marathon", "road race", "running race", "runners racing", "runner wearing a bib", "people running in a race"]) or (
        _has_any(text, ["finish line", "race bib"]) and _has_any(text, ["runner", "runners", "running"])
    ):
        out.add("marathon")

    if _has_any(text, ["parade", "marching band", "parade float", "people marching in a parade", "marching down the street"]):
        out.add("parade")

    if _has_any(text, ["birthday party", "people at a party", "party table", "dance floor", "night club"]) or (
        _has_any(text, ["party"]) and _has_any(text, ["balloons", "dancing", "drinks"])
    ):
        out.add("party")

    if _has_any(
        text,
        [
            "performing on stage",
            "performer on stage",
            "performers on stage",
            "band performing",
            "singing on stage",
            "dance performance",
            "theater performance",
        ],
    ) or (
        _has_any(text, ["stage"])
        and _has_any(text, ["microphone", "guitar", "singing", "dancer", "dancers", "musician", "musicians", "band"])
    ):
        out.add("performance")

    if _has_any(text, ["church service", "religious ceremony", "religious service", "worship", "prayer", "praying", "priest", "altar", "mosque", "temple"]):
        out.add("religious_event")

    if _has_any(text, ["people celebrating", "crowd celebrating", "confetti", "cheering crowd"]) or (
        _has_any(text, ["celebrating", "celebration"]) and not _has_any(text, ["protest", "demonstration"])
    ):
        out.add("celebration")

    if "protest" in tags.get("event", []):
        out.difference_update({"celebration", "festival", "parade"})
    return out


def refine_event_tags_with_evidence(
    tags: dict[str, list[str]],
    detections: list[dict],
    caption: str,
) -> dict[str, list[str]]:
    refined = {ns: list(tags.get(ns, [])) for ns in NAMESPACES}
    events = refined.setdefault("event", [])
    supported_hidden = _hidden_events_from_evidence(tags, detections, caption)
    events[:] = [label for label in events if label not in HIDDEN_EVENT_LABELS or label in supported_hidden]
    for label, enabled in (
        ("sports_event", _sports_event_from_evidence(tags, detections, caption)),
        ("exhibition", _exhibition_from_evidence(tags, detections, caption)),
    ):
        if enabled and label not in events:
            events.append(label)
    for label in sorted(supported_hidden):
        if label not in events:
            events.append(label)
    return refined


def refine_event_tags_with_caption_additions(
    tags: dict[str, list[str]],
    detections: list[dict],
    caption: str,
    policy: dict | None = None,
) -> dict[str, list[str]]:
    if not isinstance(policy, dict):
        return {ns: list(tags.get(ns, [])) for ns in NAMESPACES}

    enabled = {str(item).strip() for item in (policy.get("enabled") or [])}
    if not enabled:
        return {ns: list(tags.get(ns, [])) for ns in NAMESPACES}

    refined = {ns: list(tags.get(ns, [])) for ns in NAMESPACES}
    events = refined.setdefault("event", [])
    text = _event_evidence_text_without_event_tags(tags, detections, caption)

    additions: list[str] = []
    if (
        "performance_stage_music" in enabled
        and "performance" not in events
        and "protest" not in events
        and "sports_event" not in events
        and _has_any_term(text, ["stage"])
        and _has_any_term(
            text,
            [
                "guitar",
                "singing",
                "dancer",
                "dancers",
                "musician",
                "musicians",
                "band",
                "drum",
                "saxophone",
                "violin",
                "musical instrument",
                "musical instruments",
                "dance performance",
                "theater performance",
                "performing on stage",
                "performer on stage",
            ],
        )
        and not _has_any_term(text, ["protest", "sports event", "baseball", "soccer", "football", "basketball"])
    ):
        additions.append("performance")

    if (
        "celebration_strict" in enabled
        and "celebration" not in events
        and "protest" not in events
        and _has_any_term(text, ["people celebrating", "crowd celebrating", "confetti", "cheering crowd", "celebration"])
        and not _has_any_term(text, ["protest", "demonstration"])
    ):
        additions.append("celebration")

    if (
        "celebration_life_visual" in enabled
        and "celebration" not in events
        and "protest" not in events
        and any(label in events for label in ("party", "festival", "wedding", "graduation", "birthday"))
        and _has_any_term(
            text,
            [
                "confetti",
                "balloon",
                "balloons",
                "cake",
                "candles",
                "dancing",
                "dance floor",
                "fireworks",
                "cheering",
                "ceremony",
            ],
        )
    ):
        additions.append("celebration")

    if (
        "religious_strict" in enabled
        and "religious_event" not in events
        and _has_any_term(
            text,
            [
                "church service",
                "religious ceremony",
                "religious service",
                "worship",
                "prayer",
                "praying",
                "priest",
                "altar",
                "mosque",
                "temple",
            ],
        )
    ):
        additions.append("religious_event")

    if (
        "ceremony_strict" in enabled
        and "ceremony" not in events
        and _has_any_term(text, ["ribbon cutting", "formal ceremony", "ceremonial procession", "memorial service"])
    ):
        additions.append("ceremony")

    if (
        "marathon_strict" in enabled
        and "marathon" not in events
        and "wedding" not in events
        and "conference" not in events
        and "protest" not in events
        and (
            _has_any_term(text, ["marathon", "road race", "running race"])
            or (
                _has_any_term(text, ["race bib", "finish line"])
                and _has_any_term(text, ["runner", "runners", "running"])
            )
        )
        and not _has_any_term(text, ["parade float", "marching band"])
    ):
        additions.append("marathon")

    if (
        "ceremony_caption" in enabled
        and "ceremony" not in events
        and _has_any_term(text, ["ceremony", "ceremonial"])
        and not _has_any_term(text, ["conference table", "meeting room"])
        and (
            any(label in events for label in ("wedding", "graduation", "award_ceremony", "religious_event"))
            or _has_any_term(text, ["ribbon cutting", "memorial service", "formal ceremony"])
        )
    ):
        additions.append("ceremony")

    if (
        "religious_place_group" in enabled
        and "religious_event" not in events
        and "conference" not in events
        and "protest" not in events
        and "exhibition" not in events
        and _has_any_term(text, ["church", "mosque", "temple", "cathedral", "altar"])
        and _has_any_term(text, ["people gathered", "group of people", "service", "ceremony", "wedding", "prayer", "praying"])
        and not _has_any_term(text, ["building appears", "building is", "exterior", "photograph of a large"])
    ):
        additions.append("religious_event")

    for label in additions:
        if label not in events:
            events.append(label)

    if (
        "sports_ice_rink" in enabled
        and "sports_event" not in events
        and _has_any_term(text, ["ice rink", "figure skater", "figure skating", "skater performing", "skating routine"])
    ):
        events.append("sports_event")

    if (
        "sports_running_race" in enabled
        and "sports_event" not in events
        and _has_any_term(
            text,
            ["running in a marathon", "running in a race", "marathon", "race bib", "finish line", "running on a street"],
        )
    ):
        events.append("sports_event")

    if (
        "drop_exhibition_sports_event" in enabled
        and "exhibition" in events
        and "sports_event" in events
    ):
        events[:] = [label for label in events if label != "exhibition"]

    if (
        "drop_conference_gallery" in enabled
        and "conference" in events
        and _has_any_term(text, ["art gallery", "gallery", "art installation", "row of paintings"])
    ):
        events[:] = [label for label in events if label != "conference"]

    if (
        "drop_conference_exhibition" in enabled
        and "conference" in events
        and _has_any_term(text, ["scientist biology exhibition", "exhibition hall"])
    ):
        events[:] = [label for label in events if label != "conference"]

    if (
        "drop_exhibition_road" in enabled
        and "exhibition" in events
        and _has_any_term(text, ["road", "street at night", "snowy street"])
    ):
        events[:] = [label for label in events if label != "exhibition"]

    max_labels = int(policy.get("max_event_labels", 0) or 0)
    if max_labels > 0 and len(events) > max_labels:
        events[:] = events[:max_labels]
    return refined


def refine_event_tags_with_evidence_legacy(
    tags: dict[str, list[str]],
    detections: list[dict],
    caption: str,
) -> dict[str, list[str]]:
    refined = {ns: list(tags.get(ns, [])) for ns in NAMESPACES}
    events = refined.setdefault("event", [])
    for label, enabled in (
        ("sports_event", _sports_event_from_evidence(tags, detections, caption)),
        ("exhibition", _exhibition_from_evidence(tags, detections, caption)),
    ):
        if enabled and label not in events:
            events.append(label)
    return refined


class AttentionPool(nn.Module):
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


def load_tagging_checkpoint(
    path: str | Path,
    vocab: dict[str, list[str]] | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[TaggingHeads, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    ckpt_vocab = payload.get("vocab")
    if vocab is None:
        vocab = ckpt_vocab
    if vocab is None:
        raise ValueError(f"tagging checkpoint has no vocab: {path}")
    if payload.get("head_type") not in (None, HEAD_TYPE, *LEGACY_HEAD_TYPES):
        raise ValueError(f"unsupported tagging head_type={payload.get('head_type')!r}")
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


def _resolve_path(path: str | None, config_dir: Path, repo_root: Path) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    for base in (config_dir, repo_root, Path.cwd()):
        candidate = base / p
        if candidate.exists():
            return candidate
    return config_dir / p


class EUMUPredictor:
    def __init__(self, weight_path: str, config_path: str) -> None:
        self.weight_path = str(weight_path)
        self.config_path = str(config_path)
        self.config_dir = Path(config_path).resolve().parent
        self.repo_root = self.config_dir.parent
        cfg = yaml.safe_load(Path(config_path).read_text()) or {}

        infer_cfg = cfg.get("inference", {}) or {}
        self.deterministic_seed = int(infer_cfg.get("deterministic_seed", 0) or 0)
        if self.deterministic_seed:
            configure_deterministic_runtime(self.deterministic_seed)
        device_pref = str(infer_cfg.get("device", "cuda"))
        if device_pref.startswith("cuda") and not torch.cuda.is_available():
            device_pref = "cpu"
        self.device = torch.device(device_pref)
        dtype_name = str(infer_cfg.get("dtype", "float16"))
        self.dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype_name if self.device.type == "cuda" else "float32", torch.float32)
        self.num_beams = int(infer_cfg.get("num_beams", 3))

        tagging_cfg = cfg.get("task_a", {}) or {}
        vocab_path = _resolve_path(tagging_cfg.get("vocab_path") or "task_a_vocab.json", self.config_dir, self.repo_root)
        if vocab_path is None or not vocab_path.exists():
            vocab_path = Path(weight_path) / "task_a_vocab.json"
        self.vocab = load_vocab(vocab_path)

        detection_cfg = cfg.get("detection", {}) or {}
        self.detection_prompt = str(detection_cfg.get("prompt", "<OD>"))
        self.detection_max_new_tokens = int(detection_cfg.get("max_new_tokens", 1024))
        self.detection_limit = int(detection_cfg.get("limit", 300))
        self.detection_nms_iou = float(detection_cfg.get("nms_iou", 0.65))
        self.detection_score_floor = float(detection_cfg.get("score_floor", 0.0))
        self.detection_score_decimals = int(detection_cfg.get("score_decimals", -1))
        self.detection_bbox_grid = float(detection_cfg.get("bbox_grid", 0.0))
        self.detection_final_score_floor = float(detection_cfg.get("final_score_floor", 0.0))
        self.detection_bbox_scale_by_label: dict[str, float] = {}
        scale_cfg = detection_cfg.get("bbox_scale_by_label", {}) or {}
        if isinstance(scale_cfg, dict):
            for label, value in scale_cfg.items():
                try:
                    scale = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(scale) and scale > 0.0:
                    self.detection_bbox_scale_by_label[str(label).strip().lower()] = scale
        self.detection_batch_size = max(1, int(detection_cfg.get("batch_size", 1)))
        label_map_path = _resolve_path(detection_cfg.get("label_map_path"), self.config_dir, self.repo_root)
        self.label_map = load_label_map(label_map_path)
        final_alias_cfg = detection_cfg.get("final_label_aliases", {}) or {}
        self.detection_final_label_aliases: dict[str, str] = {}
        if isinstance(final_alias_cfg, dict):
            self.detection_final_label_aliases = {
                str(src).strip().lower(): str(dst).strip().lower()
                for src, dst in final_alias_cfg.items()
                if str(src).strip() and str(dst).strip()
            }
        duplicate_alias_cfg = detection_cfg.get("final_label_duplicate_aliases", {}) or {}
        self.detection_final_label_duplicate_score_multiplier = float(
            detection_cfg.get("final_label_duplicate_score_multiplier", 1.0)
        )
        duplicate_multiplier_cfg = detection_cfg.get("final_label_duplicate_score_multipliers", {}) or {}
        self.detection_final_label_duplicate_score_multipliers: dict[tuple[str, str], float] = {}
        if isinstance(duplicate_multiplier_cfg, dict):
            for src, values in duplicate_multiplier_cfg.items():
                src_label = str(src).strip().lower()
                if not src_label or not isinstance(values, dict):
                    continue
                for dst, value in values.items():
                    dst_label = str(dst).strip().lower()
                    if not dst_label:
                        continue
                    try:
                        multiplier = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(multiplier) and multiplier >= 0.0:
                        self.detection_final_label_duplicate_score_multipliers[(src_label, dst_label)] = multiplier
        self.detection_final_label_duplicate_aliases: dict[str, list[str]] = {}
        if isinstance(duplicate_alias_cfg, dict):
            for src, dst in duplicate_alias_cfg.items():
                src_label = str(src).strip().lower()
                if not src_label:
                    continue
                raw_targets = dst if isinstance(dst, list) else [dst]
                targets = [
                    str(target).strip().lower()
                    for target in raw_targets
                    if str(target).strip()
                ]
                if targets:
                    self.detection_final_label_duplicate_aliases[src_label] = targets
        extra_prompts = detection_cfg.get("extra_prompts", []) or []
        self.detection_extra_prompts = [
            item for item in extra_prompts
            if isinstance(item, dict) and item.get("prompt")
        ]
        self._phrase_label_items = self._build_phrase_label_items()
        grounding_cfg = detection_cfg.get("phrase_grounding", {}) or {}
        self.detection_phrase_grounding = (
            grounding_cfg
            if isinstance(grounding_cfg, dict) and bool(grounding_cfg.get("enabled", False))
            else {}
        )
        selective_ovd_cfg = detection_cfg.get("selective_ovd", {}) or {}
        self.detection_selective_ovd = (
            selective_ovd_cfg
            if isinstance(selective_ovd_cfg, dict) and bool(selective_ovd_cfg.get("enabled", False))
            else {}
        )
        targeted_generic_cfg = detection_cfg.get("targeted_generic_ovd", {}) or {}
        self.detection_targeted_generic_ovd = (
            targeted_generic_cfg
            if isinstance(targeted_generic_cfg, dict) and bool(targeted_generic_cfg.get("enabled", False))
            else {}
        )
        source_policy_cfg = detection_cfg.get("source_policy_by_label", {}) or {}
        self.detection_source_policy_by_label: dict[str, str] = {}
        if isinstance(source_policy_cfg, dict):
            self.detection_source_policy_by_label = {
                str(label).strip().lower(): str(source).strip()
                for label, source in source_policy_cfg.items()
                if str(label).strip() and str(source).strip()
            }
        source_variant_cfg = detection_cfg.get("source_variants", {}) or {}
        self.detection_source_variants = source_variant_cfg if isinstance(source_variant_cfg, dict) else {}

        caption_cfg = cfg.get("caption", {}) or {}
        prompts = caption_cfg.get("prompts", ["<CAPTION>"])
        self.caption_prompts = list(prompts) if isinstance(prompts, list) else ["<CAPTION>"]
        self.caption_max_new_tokens = int(caption_cfg.get("max_new_tokens", 128))
        self.caption_char_limit = int(caption_cfg.get("char_limit", 300))
        self.caption_token_limit = int(caption_cfg.get("token_limit", 30))

        self.tagging_cfg = tagging_cfg
        self.fallback_tag_context_prompt = tagging_cfg.get("fallback_tag_context_prompt", "<MORE_DETAILED_CAPTION>")
        self.fallback_tag_context_max_new_tokens = int(tagging_cfg.get("fallback_tag_context_max_new_tokens", 192))

        self.processor = AutoProcessor.from_pretrained(self.weight_path, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            self.weight_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        lora_cfg = cfg.get("lora", {}) or {}
        adapter_path = _resolve_path(lora_cfg.get("adapter_path"), self.config_dir, self.repo_root)
        if adapter_path and (adapter_path / "adapter_config.json").exists():
            from peft import PeftModel

            base = PeftModel.from_pretrained(base, str(adapter_path))
            if bool(lora_cfg.get("merge_on_load", True)):
                base = base.merge_and_unload()
        self.model = base.to(self.device).eval()

        self.tagging_heads = None
        self.tagging_thresholds: dict[str, Any] | None = None
        if tagging_cfg.get("head_checkpoint"):
            head_path = _resolve_path(tagging_cfg.get("head_checkpoint"), self.config_dir, self.repo_root)
        else:
            head_path = self.config_dir / "task_a_heads.pt"
        if head_path and head_path.exists():
            heads, meta = load_tagging_checkpoint(head_path, vocab=self.vocab, map_location="cpu")
            self.tagging_heads = heads.to(self.device).eval()
            self.tagging_thresholds = meta.get("thresholds")
            threshold_overrides = tagging_cfg.get("threshold_overrides", {}) or {}
            if isinstance(threshold_overrides, dict):
                merged = {
                    ns: dict((self.tagging_thresholds or {}).get(ns, {}) or {})
                    for ns in NAMESPACES
                }
                for ns, values in threshold_overrides.items():
                    if ns not in merged or not isinstance(values, dict):
                        continue
                    for label, value in values.items():
                        if isinstance(value, (int, float)):
                            merged[ns][str(label)] = float(value)
                self.tagging_thresholds = merged

    def _inputs(self, image: Image.Image, prompt: str):
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)
        return inputs

    def _inputs_batch(self, images: list[Image.Image], prompts: list[str]):
        if len(images) != len(prompts):
            raise ValueError(f"images/prompts batch mismatch: {len(images)} != {len(prompts)}")
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)
        return inputs

    @torch.no_grad()
    def _generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        inputs = self._inputs(image, prompt)
        ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=self.num_beams,
        )
        return self.processor.batch_decode(ids, skip_special_tokens=False)[0]

    @torch.no_grad()
    def _generate_with_scores(self, image: Image.Image, prompt: str, max_new_tokens: int):
        inputs = self._inputs(image, prompt)
        generate_kwargs = {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"],
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": self.num_beams,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        out = self.model.generate(**generate_kwargs)
        scores = self.model.compute_transition_scores(
            sequences=out.sequences,
            scores=out.scores,
            beam_indices=getattr(out, "beam_indices", None),
            normalize_logits=True,
        )
        return out.sequences[0], scores[0]

    @torch.no_grad()
    def _generate_with_scores_batch(
        self,
        images: list[Image.Image],
        prompts: list[str],
        max_new_tokens: int,
    ):
        inputs = self._inputs_batch(images, prompts)
        generate_kwargs = {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"],
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": self.num_beams,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        out = self.model.generate(**generate_kwargs)
        scores = self.model.compute_transition_scores(
            sequences=out.sequences,
            scores=out.scores,
            beam_indices=getattr(out, "beam_indices", None),
            normalize_logits=True,
        )
        return list(zip(out.sequences, scores))

    @torch.no_grad()
    def _generate_batch(
        self,
        images: list[Image.Image],
        prompts: list[str],
        max_new_tokens: int,
        skip_special_tokens: bool = False,
    ) -> list[str]:
        inputs = self._inputs_batch(images, prompts)
        ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=self.num_beams,
        )
        return self.processor.batch_decode(ids, skip_special_tokens=skip_special_tokens)

    def _run_generation_task(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        text = self._generate(image, prompt, max_new_tokens)
        parsed = self.processor.post_process_generation(
            text,
            task=prompt,
            image_size=(image.width, image.height),
        )
        if isinstance(parsed, dict):
            value = parsed.get(prompt, "")
            if isinstance(value, str):
                return value
        return ""

    def _build_phrase_label_items(self) -> list[tuple[str, str]]:
        items: set[tuple[str, str]] = set()
        for key, value in self.label_map.items():
            clean = normalise_label_text(key)
            target = self.label_map.get(normalise_label_text(value), value)
            if not clean or not target:
                continue
            items.add((clean, target))
            if len(clean) > 3 and clean.endswith("s"):
                items.add((clean[:-1], target))
            if len(clean) > 4 and clean.endswith("y"):
                items.add((f"{clean[:-1]}ies", target))
            elif len(clean) > 3 and not clean.endswith(("s", "x", "z")):
                items.add((f"{clean}s", target))
        return sorted(items, key=lambda x: (-len(x[0].split()), -len(x[0]), x[0], x[1]))

    def _exact_extra_detection_label(self, label: str) -> str:
        clean = normalise_label_text(label)
        if not clean:
            return ""
        return self.label_map.get(clean, "")

    def _phrase_extra_detection_label(self, label: str, allow_generic: bool) -> str:
        exact = self._exact_extra_detection_label(label)
        if exact:
            return exact
        clean = normalise_label_text(label)
        if not clean:
            return ""
        padded = f" {clean} "
        for key, target in self._phrase_label_items:
            if not allow_generic and key in GENERIC_DETECTION_PHRASE_LABELS:
                continue
            if f" {key} " in padded:
                return target
        return ""

    def _map_extra_detection_label(
        self,
        label: str,
        label_mode: str,
        keep_unknown: bool,
        allow_generic: bool,
    ) -> str:
        if label_mode == "phrase":
            mapped = self._phrase_extra_detection_label(label, allow_generic=allow_generic)
        elif label_mode == "exact":
            mapped = self._exact_extra_detection_label(label)
        else:
            mapped = ""
        if mapped:
            return mapped
        return str(label) if keep_unknown else ""

    def _parse_detection_payload(
        self,
        payload: dict,
        default_score: float,
        label_mode: str = "raw",
        keep_unknown: bool = True,
        allow_generic: bool = False,
        score_multiplier: float = 1.0,
    ) -> list[dict]:
        raw: list[dict] = []
        bboxes = payload.get("bboxes") or []
        labels = payload.get("labels") or payload.get("bboxes_labels") or []
        scores = payload.get("scores") or []
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            label_text = str(label)
            if label_mode != "raw":
                label_text = self._map_extra_detection_label(
                    label_text,
                    label_mode=label_mode,
                    keep_unknown=keep_unknown,
                    allow_generic=allow_generic,
                )
                if not label_text:
                    continue
            score = default_score
            if i < len(scores) and isinstance(scores[i], (int, float)) and math.isfinite(float(scores[i])):
                score = float(scores[i])
            score *= score_multiplier
            raw.append({"label": label_text, "score": score, "bbox_xyxy": list(bbox)})
        return raw

    def _is_generic_phrase_label(self, phrase: str, target: str, allow_generic: bool) -> bool:
        if allow_generic:
            return False
        phrase_clean = normalise_label_text(phrase)
        target_clean = normalise_label_text(target)
        return phrase_clean in GENERIC_DETECTION_PHRASE_LABELS or target_clean in GENERIC_DETECTION_PHRASE_LABELS

    def _add_grounding_phrase(
        self,
        phrases: list[str],
        seen_targets: set[str],
        phrase: str,
        target: str,
        allow_generic: bool,
    ) -> None:
        phrase_clean = normalise_label_text(phrase)
        target_clean = normalise_label_text(target)
        if not phrase_clean or not target_clean or len(phrase_clean) < 3:
            return
        if self._is_generic_phrase_label(phrase_clean, target_clean, allow_generic=allow_generic):
            return
        if target_clean in seen_targets:
            return
        phrases.append(phrase_clean)
        seen_targets.add(target_clean)

    def _grounding_candidate_phrases(
        self,
        context: str,
        raw_detections: list[dict],
        max_phrases: int,
        include_detection_labels: bool,
        include_context_phrases: bool,
        include_unordered_context_phrases: bool,
        include_inferred_compound_phrases: bool,
        semantic_expansion_blocklist: set[str],
        allow_generic: bool,
    ) -> list[str]:
        phrases: list[str] = []
        seen_targets: set[str] = set()

        if include_detection_labels:
            ordered = sorted(
                raw_detections,
                key=lambda d: (
                    -float(d.get("score", 0.0)),
                    str(d.get("label", "")),
                    tuple(round(float(x), 4) for x in d.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])),
                ),
            )
            for det in ordered:
                raw_label = str(det.get("label", ""))
                target = self._phrase_extra_detection_label(raw_label, allow_generic=True)
                if not target:
                    continue
                self._add_grounding_phrase(phrases, seen_targets, raw_label, target, allow_generic=allow_generic)
                if len(phrases) >= max_phrases:
                    return phrases

        if include_inferred_compound_phrases:
            for phrase in self._inferred_compound_phrases(context, raw_detections):
                target = self._phrase_extra_detection_label(phrase, allow_generic=True)
                if not target:
                    continue
                if normalise_label_text(target) in semantic_expansion_blocklist:
                    continue
                self._add_grounding_phrase(phrases, seen_targets, phrase, target, allow_generic=allow_generic)
                if len(phrases) >= max_phrases:
                    return phrases

        if include_context_phrases and context:
            padded = f" {normalise_label_text(context)} "
            for phrase, target in self._phrase_label_items:
                if len(phrase) < 3:
                    continue
                if self._is_generic_phrase_label(phrase, target, allow_generic=allow_generic):
                    continue
                if normalise_label_text(target) in seen_targets:
                    continue
                if f" {phrase} " not in padded:
                    continue
                self._add_grounding_phrase(phrases, seen_targets, phrase, target, allow_generic=allow_generic)
                if len(phrases) >= max_phrases:
                    return phrases

            if include_unordered_context_phrases:
                context_text = normalise_label_text(
                    " ".join([context, *[str(det.get("label", "")) for det in raw_detections]])
                )
                context_tokens = set(context_text.split())
                for phrase, target in self._phrase_label_items:
                    words = phrase.split()
                    if not 2 <= len(words) <= 4:
                        continue
                    if self._is_generic_phrase_label(phrase, target, allow_generic=allow_generic):
                        continue
                    if normalise_label_text(target) in semantic_expansion_blocklist:
                        continue
                    if normalise_label_text(target) in seen_targets:
                        continue
                    if all(word in context_tokens for word in words):
                        self._add_grounding_phrase(phrases, seen_targets, phrase, target, allow_generic=allow_generic)
                        if len(phrases) >= max_phrases:
                            return phrases
        return phrases

    def _inferred_compound_phrases(self, context: str, raw_detections: list[dict]) -> list[str]:
        text = normalise_label_text(
            " ".join([context, *[str(det.get("label", "")) for det in raw_detections]])
        )
        tokens = set(text.split())
        labels = {normalise_label_text(str(det.get("label", ""))) for det in raw_detections}

        def has_any(values: tuple[str, ...]) -> bool:
            return any(value in labels or value in tokens for value in values)

        phrases: list[str] = []
        if has_any(("sink", "stove", "oven", "microwave oven", "faucet", "cabinet")):
            phrases.extend(["kitchen table", "dining table"])
        if has_any(("racket", "tennis racket")) or "tennis" in tokens:
            phrases.append("tennis ball")
        if "baseball" in tokens and has_any(("glove", "baseball glove")):
            phrases.append("baseball glove")
        if has_any(("bath mat", "bathtub", "toilet", "shower")) or ("bathroom" in tokens and has_any(("sink", "faucet"))):
            phrases.append("bath towel")
        if has_any(("lamp", "lampshade")) and has_any(("desk", "coffee table", "table", "sofa")):
            phrases.append("table lamp")
        if "salad" in tokens and has_any(("bowl", "plate")):
            phrases.extend(["salad bowl", "salad plate"])
        if has_any(("orange juice", "juice", "blender")) and has_any(("orange fruit", "orange", "cup", "glass")):
            phrases.extend(["fruit juice", "smoothie"])
        if has_any(("traffic light", "street sign", "bus vehicle", "car automobile")) and "street" in tokens:
            phrases.extend(["streetlight", "traffic sign"])
        if any(word in tokens for word in ("police", "officer", "arrest")):
            phrases.append("handcuff")

        out: list[str] = []
        seen: set[str] = set()
        for phrase in phrases:
            clean = normalise_label_text(phrase)
            if clean and clean not in seen:
                out.append(clean)
                seen.add(clean)
        return out

    def _append_selective_ovd_detections(
        self,
        images: list[Image.Image],
        raw_by_image: list[list[dict]],
        cfg: dict | None = None,
    ) -> None:
        cfg = self.detection_selective_ovd if cfg is None else cfg
        if not cfg:
            return
        prompt_token = str(cfg.get("prompt", "<OPEN_VOCABULARY_DETECTION>"))
        separator = str(cfg.get("separator", "<and>"))
        context_prompt = str(cfg.get("context_prompt", "<MORE_DETAILED_CAPTION>") or "")
        max_phrases = max(1, int(cfg.get("max_phrases", 24)))
        include_detection_labels = bool(cfg.get("include_detection_labels", True))
        include_context_phrases = bool(cfg.get("include_context_phrases", True))
        allow_generic = bool(cfg.get("allow_generic_phrase_labels", False))

        jobs: list[tuple[int, Image.Image, str]] = []
        for idx, image in enumerate(images):
            context = ""
            if context_prompt:
                context = self._run_generation_task(
                    image,
                    context_prompt,
                    int(cfg.get("context_max_new_tokens", self.caption_max_new_tokens)),
                )
            phrases = self._grounding_candidate_phrases(
                context,
                raw_by_image[idx],
                max_phrases=max_phrases,
                include_detection_labels=include_detection_labels,
                include_context_phrases=include_context_phrases,
                include_unordered_context_phrases=bool(cfg.get("include_unordered_context_phrases", False)),
                include_inferred_compound_phrases=bool(cfg.get("include_inferred_compound_phrases", False)),
                semantic_expansion_blocklist={
                    normalise_label_text(str(item))
                    for item in (cfg.get("semantic_expansion_blocklist", []) or [])
                },
                allow_generic=allow_generic,
            )
            if phrases:
                jobs.append((idx, image, f"{prompt_token}{separator.join(phrases)}"))

        if not jobs:
            return
        ovd = self._run_detection_jobs(
            jobs,
            prompt_token,
            int(cfg.get("max_new_tokens", 512)),
            float(cfg.get("default_score", 0.35)),
            len(images),
            output_scores=bool(cfg.get("output_scores", False)),
            label_mode="phrase",
            keep_unknown=bool(cfg.get("keep_unknown", False)),
            allow_generic=allow_generic,
            score_multiplier=float(cfg.get("score_multiplier", 0.16)),
        )
        for idx, items in enumerate(ovd):
            raw_by_image[idx].extend(items)

    def _append_phrase_grounding_detections(
        self,
        images: list[Image.Image],
        raw_by_image: list[list[dict]],
        cfg: dict | None = None,
    ) -> None:
        cfg = self.detection_phrase_grounding if cfg is None else cfg
        if not cfg:
            return
        prompt_token = str(cfg.get("prompt", "<CAPTION_TO_PHRASE_GROUNDING>"))
        context_prompt = str(cfg.get("context_prompt", "<MORE_DETAILED_CAPTION>") or "")
        max_phrases = max(1, int(cfg.get("max_phrases", 32)))
        include_detection_labels = bool(cfg.get("include_detection_labels", True))
        include_context_phrases = bool(cfg.get("include_context_phrases", True))
        allow_generic = bool(cfg.get("allow_generic_phrase_labels", False))

        jobs: list[tuple[int, Image.Image, str]] = []
        for idx, image in enumerate(images):
            context = ""
            if context_prompt:
                context = self._run_generation_task(
                    image,
                    context_prompt,
                    int(cfg.get("context_max_new_tokens", self.caption_max_new_tokens)),
                )
            phrases = self._grounding_candidate_phrases(
                context,
                raw_by_image[idx],
                max_phrases=max_phrases,
                include_detection_labels=include_detection_labels,
                include_context_phrases=include_context_phrases,
                include_unordered_context_phrases=bool(cfg.get("include_unordered_context_phrases", False)),
                include_inferred_compound_phrases=bool(cfg.get("include_inferred_compound_phrases", False)),
                semantic_expansion_blocklist={
                    normalise_label_text(str(item))
                    for item in (cfg.get("semantic_expansion_blocklist", []) or [])
                },
                allow_generic=allow_generic,
            )
            if phrases:
                jobs.append((idx, image, f"{prompt_token}{', '.join(phrases)}"))

        if not jobs:
            return
        grounded = self._run_detection_jobs(
            jobs,
            prompt_token,
            int(cfg.get("max_new_tokens", 512)),
            float(cfg.get("default_score", 0.30)),
            len(images),
            output_scores=bool(cfg.get("output_scores", False)),
            label_mode="phrase",
            keep_unknown=bool(cfg.get("keep_unknown", False)),
            allow_generic=allow_generic,
            score_multiplier=float(cfg.get("score_multiplier", 0.10)),
        )
        for idx, items in enumerate(grounded):
            raw_by_image[idx].extend(items)

    def _append_targeted_generic_ovd_detections(
        self,
        images: list[Image.Image],
        raw_by_image: list[list[dict]],
    ) -> None:
        cfg = self.detection_targeted_generic_ovd
        if not cfg:
            return
        targets = [
            normalise_label_text(str(item))
            for item in (cfg.get("targets", []) or [])
            if normalise_label_text(str(item))
        ]
        if not targets:
            return

        evidence_terms_cfg = cfg.get("evidence_terms", {}) or {}
        evidence_terms: dict[str, list[str]] = {}
        if isinstance(evidence_terms_cfg, dict):
            for target in targets:
                raw_terms = evidence_terms_cfg.get(target, []) or [target]
                evidence_terms[target] = [
                    normalise_label_text(str(term))
                    for term in raw_terms
                    if normalise_label_text(str(term))
                ]
        else:
            evidence_terms = {target: [target] for target in targets}

        prompt_token = str(cfg.get("prompt", "<OPEN_VOCABULARY_DETECTION>"))
        separator = str(cfg.get("separator", "<and>"))
        context_prompt = str(cfg.get("context_prompt", "<MORE_DETAILED_CAPTION>") or "")
        jobs: list[tuple[int, Image.Image, str]] = []
        allowed_by_image: list[set[str]] = [set() for _ in images]
        for idx, image in enumerate(images):
            context = ""
            if context_prompt:
                context = self._run_generation_task(
                    image,
                    context_prompt,
                    int(cfg.get("context_max_new_tokens", self.caption_max_new_tokens)),
                )
            evidence_text = normalise_label_text(
                " ".join([context, *[str(det.get("label", "")) for det in raw_by_image[idx]]])
            )
            selected: list[str] = []
            padded = f" {evidence_text} "
            for target in targets:
                if any(f" {term} " in padded for term in evidence_terms.get(target, [target])):
                    selected.append(target)
                    allowed_by_image[idx].add(target)
            if selected:
                jobs.append((idx, image, f"{prompt_token}{separator.join(selected)}"))

        if not jobs:
            return
        ovd = self._run_detection_jobs(
            jobs,
            prompt_token,
            int(cfg.get("max_new_tokens", 256)),
            float(cfg.get("default_score", 0.25)),
            len(images),
            output_scores=bool(cfg.get("output_scores", False)),
            label_mode="phrase",
            keep_unknown=False,
            allow_generic=True,
            score_multiplier=float(cfg.get("score_multiplier", 0.20)),
        )
        for idx, items in enumerate(ovd):
            allowed = allowed_by_image[idx]
            if not allowed:
                continue
            raw_by_image[idx].extend(
                item for item in items
                if normalise_label_text(str(item.get("label", ""))) in allowed
            )

    def _run_detection_jobs(
        self,
        jobs: list[tuple[int, Image.Image, str]],
        task: str,
        max_new_tokens: int,
        default_score: float,
        out_count: int,
        output_scores: bool = True,
        label_mode: str = "raw",
        keep_unknown: bool = True,
        allow_generic: bool = False,
        score_multiplier: float = 1.0,
    ) -> list[list[dict]]:
        raw_by_image: list[list[dict]] = [[] for _ in range(out_count)]
        for start in range(0, len(jobs), self.detection_batch_size):
            batch = jobs[start : start + self.detection_batch_size]
            images = [image for _, image, _ in batch]
            prompts = [prompt for _, _, prompt in batch]
            if output_scores:
                generated = self._generate_with_scores_batch(images, prompts, max_new_tokens)
                parsed_items = []
                for seq, trans_scores in generated:
                    parsed_items.append((seq, trans_scores))
            else:
                parsed_items = self._generate_batch(images, prompts, max_new_tokens)

            for parsed_item, (image_idx, image, _) in zip(parsed_items, batch):
                if output_scores:
                    seq, trans_scores = parsed_item
                    parsed = self.processor.post_process_generation(
                        sequence=seq,
                        transition_beam_score=trans_scores,
                        task=task,
                        image_size=(image.width, image.height),
                    )
                else:
                    parsed = self.processor.post_process_generation(
                        parsed_item,
                        task=task,
                        image_size=(image.width, image.height),
                    )
                payload = parsed.get(task, {}) if isinstance(parsed, dict) else {}
                if isinstance(payload, dict):
                    raw_by_image[image_idx].extend(
                        self._parse_detection_payload(
                            payload,
                            default_score,
                            label_mode=label_mode,
                            keep_unknown=keep_unknown,
                            allow_generic=allow_generic,
                            score_multiplier=score_multiplier,
                        )
                    )
        return raw_by_image

    def _merged_source_subconfig(self, base: dict, override: Any) -> dict:
        if not isinstance(override, dict):
            return copy.deepcopy(base)
        merged = copy.deepcopy(base)
        for key, value in override.items():
            merged[key] = value
        if not bool(merged.get("enabled", False)):
            return {}
        return merged

    def _detect_many_base_only(self, images: list[Image.Image]) -> list[list[dict]]:
        jobs = [
            (idx, image, self.detection_prompt)
            for idx, image in enumerate(images)
        ]
        raw_by_image = self._run_detection_jobs(
            jobs,
            self.detection_prompt,
            self.detection_max_new_tokens,
            0.5,
            len(images),
            output_scores=True,
        )
        return [
            normalise_detections(
                raw,
                image_width=image.width,
                image_height=image.height,
                label_map=self.label_map,
                score_floor=self.detection_score_floor,
                iou_thresh=self.detection_nms_iou,
                limit=self.detection_limit,
                bbox_scale_by_label=None,
            )
            for raw, image in zip(raw_by_image, images)
        ]

    def _detect_many_single_source(
        self,
        images: list[Image.Image],
        selective_ovd_cfg: dict | None = None,
        phrase_grounding_cfg: dict | None = None,
    ) -> list[list[dict]]:
        jobs = [
            (idx, image, self.detection_prompt)
            for idx, image in enumerate(images)
        ]
        raw_by_image = self._run_detection_jobs(
            jobs,
            self.detection_prompt,
            self.detection_max_new_tokens,
            0.5,
            len(images),
            output_scores=True,
        )
        for extra_cfg in self.detection_extra_prompts:
            prompt = str(extra_cfg.get("prompt", ""))
            task = str(extra_cfg.get("task", prompt))
            if not prompt or not task:
                continue
            jobs = [(idx, image, prompt) for idx, image in enumerate(images)]
            extra_raw = self._run_detection_jobs(
                jobs,
                task,
                int(extra_cfg.get("max_new_tokens", self.detection_max_new_tokens)),
                float(extra_cfg.get("default_score", 0.35)),
                len(images),
                output_scores=bool(extra_cfg.get("output_scores", True)),
                label_mode=str(extra_cfg.get("label_mode", "raw")),
                keep_unknown=bool(extra_cfg.get("keep_unknown", True)),
                allow_generic=bool(extra_cfg.get("allow_generic_phrase_labels", False)),
                score_multiplier=float(extra_cfg.get("score_multiplier", 1.0)),
            )
            for idx, items in enumerate(extra_raw):
                raw_by_image[idx].extend(items)

        self._append_selective_ovd_detections(images, raw_by_image, selective_ovd_cfg)
        self._append_phrase_grounding_detections(images, raw_by_image, phrase_grounding_cfg)
        self._append_targeted_generic_ovd_detections(images, raw_by_image)

        return [
            normalise_detections(
                raw,
                image_width=image.width,
                image_height=image.height,
                label_map=self.label_map,
                score_floor=self.detection_score_floor,
                iou_thresh=self.detection_nms_iou,
                limit=self.detection_limit,
                bbox_scale_by_label=None,
            )
            for raw, image in zip(raw_by_image, images)
        ]

    def _detect_many_with_source_policy(self, images: list[Image.Image]) -> list[list[dict]]:
        by_source: dict[str, list[list[dict]]] = {
            "primary": self._detect_many_single_source(
                images,
                selective_ovd_cfg=self.detection_selective_ovd,
                phrase_grounding_cfg=self.detection_phrase_grounding,
            )
        }
        for name, variant in self.detection_source_variants.items():
            if not isinstance(variant, dict):
                continue
            selective_cfg = self._merged_source_subconfig(
                self.detection_selective_ovd,
                variant.get("selective_ovd", {}),
            )
            grounding_cfg = self._merged_source_subconfig(
                self.detection_phrase_grounding,
                variant.get("phrase_grounding", {}),
            )
            by_source[str(name)] = self._detect_many_single_source(
                images,
                selective_ovd_cfg=selective_cfg,
                phrase_grounding_cfg=grounding_cfg,
            )

        out: list[list[dict]] = []
        for image_idx, image in enumerate(images):
            merged: list[dict] = []
            for source_name, detections_by_image in by_source.items():
                detections = detections_by_image[image_idx]
                for det in detections:
                    label = str(det.get("label", "")).strip().lower()
                    wanted = self.detection_source_policy_by_label.get(label, "primary")
                    if wanted == source_name:
                        merged.append(det)
            merged.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
            merged = scale_detection_bboxes(
                merged,
                image_width=image.width,
                image_height=image.height,
                bbox_scale_by_label=self.detection_bbox_scale_by_label,
            )
            out.append(merged[: self.detection_limit])
        return out

    def _apply_final_label_aliases(self, detections_by_image: list[list[dict]]) -> list[list[dict]]:
        if not self.detection_final_label_aliases and not self.detection_final_label_duplicate_aliases:
            return detections_by_image
        out: list[list[dict]] = []
        for detections in detections_by_image:
            rows: list[dict] = []
            for det in detections:
                row = dict(det)
                label = str(row.get("label", "")).strip().lower()
                final_label = self.detection_final_label_aliases.get(label, row.get("label", label))
                row["label"] = final_label
                row["score"] = self._final_score(row.get("score", 0.0))
                row["bbox_xyxy"] = self._final_bbox(row.get("bbox_xyxy", [0.0, 0.0, 1.0, 1.0]))
                duplicate_src = str(final_label).strip().lower()
                rows.append(row)
                for duplicate_label in self.detection_final_label_duplicate_aliases.get(duplicate_src, []):
                    if duplicate_label == row["label"]:
                        continue
                    multiplier = self.detection_final_label_duplicate_score_multipliers.get(
                        (duplicate_src, duplicate_label),
                        self.detection_final_label_duplicate_score_multiplier,
                    )
                    duplicate = dict(row)
                    duplicate["label"] = duplicate_label
                    duplicate["score"] = self._final_score(float(det.get("score", 0.0)) * multiplier)
                    rows.append(duplicate)
            rows = sorted(rows, key=self._final_detection_sort_key)
            if self.detection_final_score_floor > 0.0:
                rows = [
                    row for row in rows
                    if float(row.get("score", 0.0)) >= self.detection_final_score_floor
                ]
            if len(rows) > self.detection_limit:
                rows = rows[: self.detection_limit]
            out.append(rows)
        return out

    def _detect_many_pre_final(self, images: list[Image.Image]) -> list[list[dict]]:
        if self.detection_source_policy_by_label:
            return self._detect_many_with_source_policy(images)
        return [
            scale_detection_bboxes(
                dets,
                image_width=image.width,
                image_height=image.height,
                bbox_scale_by_label=self.detection_bbox_scale_by_label,
            )
            for dets, image in zip(self._detect_many_single_source(images), images)
        ]

    def _detect_many(self, images: list[Image.Image]) -> list[list[dict]]:
        return self._apply_final_label_aliases(self._detect_many_pre_final(images))

    def _detect_pre_final(self, image: Image.Image) -> list[dict]:
        return self._detect_many_pre_final([image])[0]

    def _detect(self, image: Image.Image) -> list[dict]:
        return self._detect_many([image])[0]

    def _final_score(self, score: float) -> float:
        score = max(0.0, min(1.0, float(score)))
        if self.detection_score_decimals >= 0:
            factor = 10 ** self.detection_score_decimals
            score = math.floor(score * factor) / factor
        return score

    def _final_bbox(self, bbox: list[float]) -> list[float]:
        values = [float(x) for x in bbox]
        if self.detection_bbox_grid > 0.0:
            grid = self.detection_bbox_grid
            values = [math.floor(x / grid) * grid for x in values]
            if values[2] <= values[0]:
                values[2] = values[0] + grid
            if values[3] <= values[1]:
                values[3] = values[1] + grid
        return values

    @staticmethod
    def _final_detection_sort_key(item: dict) -> tuple:
        bbox = item.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
        return (
            -float(item.get("score", 0.0)),
            str(item.get("label", "")),
            tuple(round(float(x), 4) for x in bbox),
        )

    def _caption(self, image: Image.Image, detections: list[dict]) -> tuple[str, str]:
        candidates: list[str] = []
        for prompt in self.caption_prompts:
            candidates.append(self._run_generation_task(image, str(prompt), self.caption_max_new_tokens))
        candidates = [sanitize_caption(c, self.caption_char_limit, self.caption_token_limit) for c in candidates]

        det_words = set()
        for det in detections[:20]:
            det_words.update(str(det.get("label", "")).split())

        def score(text: str) -> tuple[int, int, int]:
            tokens = set(text.lower().replace("-", " ").split())
            coverage = len(tokens & det_words)
            length_score = -abs(len(text.split()) - 12)
            return (coverage, length_score, len(text))

        best = max(candidates or ["an image"], key=score)
        return best, " ".join(candidates)

    @torch.no_grad()
    def _tags_with_heads(self, image: Image.Image) -> dict[str, list[str]]:
        if self.tagging_heads is None:
            raise RuntimeError("tagging heads are not loaded")
        inputs = self._inputs(image, "<CAPTION>")
        tokens = encode_image_tokens(self.model, inputs["pixel_values"]).float()
        logits = self.tagging_heads(tokens)
        return predict_tags_from_logits(
            logits,
            self.vocab,
            thresholds=self.tagging_thresholds,
            default_threshold=float(self.tagging_cfg.get("default_threshold", 0.5)),
            quality_default_good=bool(self.tagging_cfg.get("quality_default_good", True)),
            scene_top_k=int(self.tagging_cfg.get("scene_top_k", 1 if self.tagging_cfg.get("scene_top1", True) else 0)),
            scene_min_prob=float(self.tagging_cfg.get("scene_min_prob", 0.0)),
            scene_max_labels=int(self.tagging_cfg.get("scene_max_labels", 1)),
            scene_specific_max_labels=int(self.tagging_cfg.get("scene_specific_max_labels", 1)),
            event_top_k=int(self.tagging_cfg.get("event_top_k", 1 if self.tagging_cfg.get("event_top1", True) else 0)),
            event_min_prob=float(self.tagging_cfg.get("event_min_prob", 0.0)),
            event_max_labels=int(self.tagging_cfg.get("event_max_labels", 1)),
            quality_max_labels=int(self.tagging_cfg.get("quality_max_labels", 2)),
            legacy_scene_event_softmax=bool(self.tagging_cfg.get("legacy_scene_event_softmax", False)),
        )

    def _fallback_tags(self, image: Image.Image, caption_context: str) -> dict[str, list[str]]:
        context = caption_context
        if self.fallback_tag_context_prompt:
            context = " ".join(
                [
                    context,
                    self._run_generation_task(
                        image,
                        str(self.fallback_tag_context_prompt),
                        self.fallback_tag_context_max_new_tokens,
                    ),
                ]
            )
        return keyword_tags(context, self.vocab)

    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        if self.deterministic_seed:
            configure_deterministic_runtime(self.deterministic_seed)
        image = Image.open(image_path).convert("RGB")
        pre_final_detections = self._detect_pre_final(image)
        detections = self._apply_final_label_aliases([pre_final_detections])[0]
        caption, caption_context = self._caption(image, detections)
        if self.tagging_heads is not None:
            tags = self._tags_with_heads(image)
        else:
            tags = self._fallback_tags(image, caption_context)
        if bool(self.tagging_cfg.get("quality_stat_fusion", False)):
            if bool(self.tagging_cfg.get("legacy_quality_stat_fusion", False)):
                tags = refine_quality_tags_with_image_stats_legacy(
                    tags,
                    image,
                    max_labels=int(self.tagging_cfg.get("quality_max_labels", 2)),
                )
            else:
                tags = refine_quality_tags_with_image_stats(
                    tags,
                    image,
                    max_labels=int(self.tagging_cfg.get("quality_max_labels", 2)),
                    add_rules=self.tagging_cfg.get("quality_stat_add_rules"),
                    drop_rules=self.tagging_cfg.get("quality_stat_drop_rules"),
                    sequence_rules=self.tagging_cfg.get("quality_stat_sequence_rules"),
                )
        if bool(self.tagging_cfg.get("event_evidence_fusion", False)):
            evidence_view = str(self.tagging_cfg.get("event_evidence_detection_view", "final")).strip().lower()
            if evidence_view in {"base", "base_od", "base-od", "od"}:
                event_detections = self._detect_many_base_only([image])[0]
            elif evidence_view in {"pre_final", "pre-final", "prefinal"}:
                event_detections = pre_final_detections
            else:
                event_detections = detections
            event_caption = caption
            event_prompt = self.tagging_cfg.get("event_evidence_context_prompt")
            if event_prompt:
                event_caption = self._run_generation_task(
                    image,
                    str(event_prompt),
                    int(self.tagging_cfg.get("event_evidence_context_max_new_tokens", 128)),
                )
            if bool(self.tagging_cfg.get("legacy_event_evidence_fusion", False)):
                tags = refine_event_tags_with_evidence_legacy(tags, event_detections, event_caption)
            else:
                tags = refine_event_tags_with_evidence(tags, event_detections, event_caption)
        event_additions = self.tagging_cfg.get("event_caption_additions")
        if event_additions:
            tags = refine_event_tags_with_caption_additions(tags, detections, caption, event_additions)
        scene_overrides = self.tagging_cfg.get("scene_caption_overrides")
        if scene_overrides:
            tags = refine_scene_tags_with_caption_overrides(tags, detections, caption, scene_overrides)
        return {
            "tags": {ns: list(tags.get(ns, [])) for ns in NAMESPACES},
            "detections": detections,
            "caption": caption,
        }


def iter_images(folder: str | Path) -> Iterable[Path]:
    root = Path(folder)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


UnifiedPredictor = EUMUPredictor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    predictor = EUMUPredictor(args.weight_path, args.config_path)
    print(json.dumps(predictor.predict(args.image), ensure_ascii=False, indent=2))
