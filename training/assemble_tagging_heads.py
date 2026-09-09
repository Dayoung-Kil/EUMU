"""Assemble the final EUMU tagging checkpoint.

The final submission checkpoint uses the main attention-pooling trunk and
scene/event heads, the quality-specialized head, and validation-selected
thresholds.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from common import load_vocab
from tagging_heads import load_tagging_checkpoint, save_tagging_checkpoint

FINAL_SCENE_THRESHOLDS = {
    "airport": 0.9,
    "bathroom": 0.47,
    "beach": 0.83,
    "bedroom": 0.72,
    "bridge": 0.45,
    "classroom": 0.16,
    "coast": 0.65,
    "desert": 0.9,
    "factory": 0.89,
    "farm": 0.37,
    "forest": 0.87,
    "garden": 0.85,
    "highway": 0.53,
    "hotel": 0.74,
    "indoor": 0.59,
    "kitchen": 0.88,
    "lake": 0.77,
    "living_room": 0.91,
    "mountain": 0.84,
    "museum": 0.58,
    "ocean": 0.81,
    "office": 0.91,
    "outdoor": 0.05,
    "park": 0.71,
    "parking_lot": 0.01,
    "restaurant": 0.85,
    "river": 0.87,
    "rooftop": 0.5,
    "shopping_mall": 0.8,
    "snow": 0.81,
    "stadium": 0.96,
    "stage": 0.56,
    "street": 0.9,
    "supermarket": 0.96,
    "train_station": 0.85,
}

FINAL_EVENT_THRESHOLDS = {
    "award_ceremony": 0.44,
    "birthday": 0.83,
    "celebration": 0.51,
    "ceremony": 0.57,
    "concert": 0.82,
    "conference": 0.88,
    "cooking": 0.87,
    "exhibition": 0.02,
    "festival": 0.04,
    "graduation": 0.92,
    "marathon": 0.02,
    "parade": 0.39,
    "party": 0.02,
    "performance": 0.42,
    "protest": 0.98,
    "religious_event": 0.06,
    "sports_event": 0.02,
    "wedding": 0.9,
}


def load_threshold_summary(path: Path | None) -> tuple[dict[str, float] | None, float | None]:
    if path is None:
        return None, None
    summary = json.loads(path.read_text())
    thresholds = summary.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError(f"{path} has no thresholds object")
    macro = summary.get("best_macro_f1")
    return {str(k): float(v) for k, v in thresholds.items()}, float(macro) if isinstance(macro, (int, float)) else None


def copy_namespace(dst_state: dict[str, torch.Tensor], src_state: dict[str, torch.Tensor], namespace: str) -> None:
    prefix = f"{namespace}."
    for key, value in src_state.items():
        if key.startswith(prefix):
            if key not in dst_state:
                raise KeyError(f"destination checkpoint has no key {key}")
            if tuple(dst_state[key].shape) != tuple(value.shape):
                raise RuntimeError(f"shape mismatch for {key}: {tuple(dst_state[key].shape)} vs {tuple(value.shape)}")
            dst_state[key] = value.clone()


def main() -> None:
    parser = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--main-head", type=Path, default=repo / "training_runs" / "tagging_main" / "tagging_heads.pt")
    parser.add_argument("--quality-head", type=Path, default=repo / "training_runs" / "tagging_quality" / "tagging_heads.pt")
    parser.add_argument("--vocab", type=Path, default=repo / "model" / "task_a_vocab.json")
    parser.add_argument("--scene-summary", type=Path)
    parser.add_argument("--event-summary", type=Path)
    parser.add_argument("--out", type=Path, default=repo / "training_runs" / "final_tagging" / "tagging_heads.pt")
    parser.add_argument("--install-to-model", action="store_true")
    args = parser.parse_args()

    vocab = load_vocab(args.vocab)
    main_heads, main_meta = load_tagging_checkpoint(args.main_head, vocab=vocab, map_location="cpu")
    quality_heads, quality_meta = load_tagging_checkpoint(args.quality_head, vocab=vocab, map_location="cpu")

    main_state = main_heads.state_dict()
    quality_state = quality_heads.state_dict()
    copy_namespace(main_state, quality_state, "quality")
    main_heads.load_state_dict(main_state, strict=True)

    thresholds: dict[str, Any] = {
        ns: dict((main_meta.get("thresholds") or {}).get(ns, {}))
        for ns in ("quality", "scene", "event")
    }
    thresholds["quality"] = dict((quality_meta.get("thresholds") or {}).get("quality", {}))

    scene_thresholds, scene_macro = load_threshold_summary(args.scene_summary)
    event_thresholds, event_macro = load_threshold_summary(args.event_summary)
    thresholds["scene"] = scene_thresholds or dict(FINAL_SCENE_THRESHOLDS)
    thresholds["event"] = event_thresholds or dict(FINAL_EVENT_THRESHOLDS)
    thresholds["quality"]["blur"] = 0.76

    extra = {
        "merged_from": {
            "quality": str(args.quality_head),
            "scene": str(args.main_head),
            "event": str(args.main_head),
            "pool": str(args.main_head),
        },
        "scene_threshold_search": str(args.scene_summary) if args.scene_summary else "built-in final thresholds",
        "scene_threshold_macro_f1": scene_macro,
        "final_threshold_merge": {
            "quality_change": "blur threshold only from fixed quality threshold search",
            "blur_threshold": 0.76,
            "event_source": str(args.event_summary) if args.event_summary else "built-in final thresholds",
            "event_macro_f1": event_macro,
        },
        "final_runtime_update": "no new tagging training; runtime config adds drop_exhibition_sports_event",
    }
    save_tagging_checkpoint(args.out, main_heads, thresholds=thresholds, extra=extra)
    print(f"saved {args.out}", flush=True)

    if args.install_to_model:
        target = repo / "model" / "task_a_heads.pt"
        shutil.copy2(args.out, target)
        print(f"installed {target}", flush=True)


if __name__ == "__main__":
    main()
