"""Prepare all tagging training rows for EUMU.

The script uses only official training images and captions. It writes the base
tagging rows, synthetic quality rows, and caption-keyword event pseudo rows used
by the final tagging recipe.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

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

EVENT_RULES: dict[str, dict[str, object]] = {
    "birthday": {
        "positive": [
            "birthday",
            "birthday cake",
            "birthday party",
            "blowing out candles",
            "candles on a cake",
        ],
        "negative": ["wedding", "graduation"],
        "confidence": 0.80,
    },
    "cooking": {
        "positive": [
            "cooking",
            "cook",
            "chef",
            "frying",
            "preparing food",
            "cutting vegetables",
        ],
        "negative": ["restaurant", "dining table"],
        "confidence": 0.62,
    },
    "parade": {
        "positive": ["parade", "marching band", "float in a parade", "procession"],
        "negative": ["wedding", "protest"],
        "confidence": 0.78,
    },
    "marathon": {
        "positive": ["marathon", "race bib", "finish line", "running race", "runners racing"],
        "negative": ["horse race", "ski race"],
        "confidence": 0.70,
    },
    "sports_event": {
        "positive": [
            "soccer game",
            "football game",
            "baseball game",
            "basketball game",
            "tennis match",
            "volleyball game",
            "players on a field",
            "player on a field",
            "baseball player",
            "stadium",
        ],
        "negative": ["graduation", "wedding"],
        "confidence": 0.62,
    },
    "exhibition": {
        "positive": [
            "museum",
            "gallery",
            "exhibit",
            "exhibition",
            "art gallery",
            "paintings on a wall",
            "sculpture",
        ],
        "negative": ["protest", "wedding", "graduation"],
        "confidence": 0.62,
    },
    "performance": {
        "positive": [
            "performing on stage",
            "performer on stage",
            "dancer on stage",
            "dancers on stage",
            "singing on stage",
            "band playing",
            "microphone on stage",
        ],
        "negative": ["conference", "wedding", "graduation"],
        "confidence": 0.58,
    },
    "religious_event": {
        "positive": ["church service", "worship", "priest", "altar", "mosque", "temple"],
        "negative": ["tourists", "museum"],
        "confidence": 0.70,
    },
    "festival": {
        "positive": ["festival", "carnival", "fairground", "fireworks", "street fair"],
        "negative": ["baseball", "wedding"],
        "confidence": 0.60,
    },
    "party": {
        "positive": ["party", "people at a party", "party table", "balloons and cake"],
        "negative": ["wedding", "conference"],
        "confidence": 0.56,
    },
    "celebration": {
        "positive": ["celebration", "celebrating", "confetti", "people celebrating"],
        "negative": ["wedding", "graduation", "protest"],
        "confidence": 0.54,
    },
    "ceremony": {
        "positive": ["ceremony", "ceremonial", "ribbon cutting"],
        "negative": ["wedding", "graduation"],
        "confidence": 0.54,
    },
    "award_ceremony": {
        "positive": ["award ceremony", "trophy", "medal ceremony", "podium"],
        "negative": ["sports field"],
        "confidence": 0.60,
    },
}


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def clean_text(text: object) -> str:
    return " ".join(str(text or "").lower().replace("_", " ").split())


def row_text(row: dict) -> str:
    parts = [row.get("target", "")]
    captions = row.get("all_captions")
    if isinstance(captions, list):
        parts.extend(str(x) for x in captions)
    return clean_text(" ".join(str(x) for x in parts))


def phrase_hit(text: str, phrase: str) -> bool:
    phrase = clean_text(phrase)
    if not phrase:
        return False
    if " " in phrase:
        return phrase in text
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def existing_image(row: dict, image_index: dict[str, str] | None = None) -> str | None:
    image = row.get("image") or row.get("image_path") or row.get("path")
    if image:
        path = Path(str(image))
        if path.exists() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return str(path)
    name = row.get("image_fname")
    if image_index and name and str(name) in image_index:
        return image_index[str(name)]
    return None


def collect_missing_filenames(paths: list[Path]) -> set[str]:
    names: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            if existing_image(row):
                continue
            name = row.get("image_fname")
            if name and Path(str(name)).suffix.lower() in IMAGE_EXTENSIONS:
                names.add(str(name))
    return names


def build_image_index(needed: set[str], roots: list[Path]) -> dict[str, str]:
    index: dict[str, str] = {}
    if not needed:
        return index
    for root in roots:
        if not root.exists():
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in filenames:
                if filename in needed and filename not in index:
                    index[filename] = str(Path(dirpath) / filename)
            if len(index) == len(needed):
                return index
    return index


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_base_rows(source_dir: Path, out: Path, image_index: dict[str, str]) -> None:
    sources = [
        source_dir / "train_tag_quality.jsonl",
        source_dir / "train_tag_scene.jsonl",
        source_dir / "train_tag_event.jsonl",
    ]
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(source)

    rows: list[dict] = []
    stats = Counter()
    skipped = Counter()
    for source in sources:
        for row in iter_jsonl(source):
            image = existing_image(row, image_index)
            if not image:
                raise RuntimeError(f"row has no resolvable image path: {row}")
            if Path(image).suffix.lower() not in IMAGE_EXTENSIONS:
                skipped["non_image"] += 1
                continue
            row["image"] = image
            stats[str(row.get("namespace", source.stem))] += 1
            rows.append(row)

    write_jsonl(out, rows)
    print(f"wrote {out} rows={len(rows)}")
    print(json.dumps(dict(stats), indent=2, sort_keys=True))
    if skipped:
        print(f"skipped={dict(skipped)}")


def make_quality_rows(
    caption_jsonl: Path,
    train_out: Path,
    calib_out: Path,
    per_label: int,
    calib_per_label: int,
    seed: int,
    image_index: dict[str, str],
) -> None:
    images = []
    seen = set()
    for row in iter_jsonl(caption_jsonl):
        image = existing_image(row, image_index)
        if image and image not in seen:
            images.append(image)
            seen.add(image)

    rng = random.Random(seed)
    rng.shuffle(images)
    if not images:
        raise RuntimeError(f"no caption images resolved from {caption_jsonl}")

    train_rows: list[dict] = []
    calib_rows: list[dict] = []
    cursor = 0
    for label in QUALITY_AUGS:
        needed = per_label + calib_per_label
        if cursor + needed > len(images):
            cursor = 0
            rng.shuffle(images)
        selected = images[cursor: cursor + needed]
        cursor += needed

        for split, target_rows, split_images in (
            ("train", train_rows, selected[:per_label]),
            ("calib", calib_rows, selected[per_label:]),
        ):
            for image in split_images:
                target_rows.append(
                    {
                        "image": image,
                        "target": f"quality: {label} | scene: none | event: none",
                        "namespace": "quality",
                        "source": f"synthetic_quality_{split}",
                        "quality_aug": label,
                        "confidence": 1.60,
                    }
                )

    write_jsonl(train_out, train_rows)
    write_jsonl(calib_out, calib_rows)
    print(f"wrote {train_out} rows={len(train_rows)}")
    print(f"wrote {calib_out} rows={len(calib_rows)}")


def make_event_rows(
    caption_jsonl: Path,
    out: Path,
    max_per_label: int,
    seed: int,
    image_index: dict[str, str],
) -> None:
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in iter_jsonl(caption_jsonl):
        image = existing_image(row, image_index)
        if not image:
            continue
        text = row_text(row)
        candidates = []
        for label, rule in EVENT_RULES.items():
            positives = list(rule["positive"])  # type: ignore[index]
            negatives = list(rule["negative"])  # type: ignore[index]
            pos_hits = sum(1 for phrase in positives if phrase_hit(text, str(phrase)))
            if pos_hits == 0:
                continue
            if any(phrase_hit(text, str(phrase)) for phrase in negatives):
                continue
            confidence = float(rule["confidence"])  # type: ignore[index]
            candidates.append((pos_hits, confidence, label))
        if not candidates:
            continue
        candidates.sort(reverse=True)
        _hits, confidence, label = candidates[0]
        buckets[label].append(
            {
                "image": image,
                "target": f"quality: none | scene: none | event: {label}",
                "namespace": "event",
                "source": "caption_keyword_event_pseudo",
                "confidence": confidence,
                "pseudo_text": text[:240],
            }
        )

    rows: list[dict] = []
    counts = {}
    for label in sorted(EVENT_RULES):
        items = buckets.get(label, [])
        rng.shuffle(items)
        capped = items[:max_per_label]
        rows.extend(capped)
        counts[label] = len(capped)
    rng.shuffle(rows)
    write_jsonl(out, rows)
    print(f"wrote {out} rows={len(rows)}")
    print(json.dumps(counts, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--source-dir", type=Path, default=repo / "data" / "train_data")
    parser.add_argument("--caption-jsonl", type=Path, default=repo / "data" / "train_data" / "train_caption.jsonl")
    parser.add_argument("--out-dir", type=Path, default=repo / "training_runs" / "data")
    parser.add_argument("--base-out", type=Path)
    parser.add_argument("--quality-train-out", type=Path)
    parser.add_argument("--quality-calib-out", type=Path)
    parser.add_argument("--event-out", type=Path)
    parser.add_argument("--image-root", type=Path, action="append", default=[])
    parser.add_argument("--quality-per-label", type=int, default=1800)
    parser.add_argument("--quality-calib-per-label", type=int, default=350)
    parser.add_argument("--event-max-per-label", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20270827)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_out = args.base_out or args.out_dir / "tagging_train.jsonl"
    quality_train_out = args.quality_train_out or args.out_dir / "quality_synthetic_train.jsonl"
    quality_calib_out = args.quality_calib_out or args.out_dir / "quality_synthetic_calib.jsonl"
    event_out = args.event_out or args.out_dir / "event_caption_pseudo.jsonl"

    source_paths = [
        args.source_dir / "train_tag_quality.jsonl",
        args.source_dir / "train_tag_scene.jsonl",
        args.source_dir / "train_tag_event.jsonl",
        args.caption_jsonl,
    ]
    needed = collect_missing_filenames(source_paths)
    image_index = build_image_index(needed, args.image_root)
    missing = sorted(name for name in needed if name not in image_index)
    if missing:
        raise RuntimeError(f"could not resolve {len(missing)} image_fname values, e.g. {missing[:5]}")

    make_base_rows(args.source_dir, base_out, image_index)
    make_quality_rows(
        args.caption_jsonl,
        quality_train_out,
        quality_calib_out,
        per_label=args.quality_per_label,
        calib_per_label=args.quality_calib_per_label,
        seed=args.seed,
        image_index=image_index,
    )
    make_event_rows(
        args.caption_jsonl,
        event_out,
        max_per_label=args.event_max_per_label,
        seed=args.seed + 17,
        image_index=image_index,
    )


if __name__ == "__main__":
    main()
