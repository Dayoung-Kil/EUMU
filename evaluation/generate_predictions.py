"""Generate EUMU predictions for a folder of evaluation images."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from inference import EUMUPredictor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_INFO = {
    "parameters_m": 239.169,
    "gflops_224": 23.947,
    "peak_memory_gb": 4.5,
}


def default_model_path() -> Path:
    model_dir = REPO_ROOT / "model"
    return model_dir if model_dir.exists() else REPO_ROOT


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def resolve_images(image_dir: Path, ids_file: Path | None) -> list[tuple[str, Path]]:
    if ids_file is None:
        return [(path.stem, path) for path in iter_images(image_dir)]

    stem_to_path: dict[str, Path] = {}
    for path in iter_images(image_dir):
        stem_to_path.setdefault(path.stem, path)

    pairs: list[tuple[str, Path]] = []
    missing: list[str] = []
    for line in ids_file.read_text().splitlines():
        image_id = line.strip()
        if not image_id or image_id.startswith("#"):
            continue
        path = stem_to_path.get(image_id)
        if path is None:
            missing.append(image_id)
        else:
            pairs.append((image_id, path))
    if missing:
        raise FileNotFoundError(f"{len(missing)} image ids were not found, e.g. {missing[:5]}")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path)
    model_path = default_model_path()
    parser.add_argument("--weight-path", type=Path, default=model_path)
    parser.add_argument("--config-path", type=Path, default=model_path / "config.yaml")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "predictions.generated.json")
    args = parser.parse_args()

    predictor = EUMUPredictor(weight_path=str(args.weight_path), config_path=str(args.config_path))
    predictions = []
    for image_id, image_path in resolve_images(args.image_dir, args.ids_file):
        predictions.append({"image_id": image_id, **predictor.predict(str(image_path))})

    payload = {
        "schema_version": "1.0",
        "model_info": MODEL_INFO,
        "predictions": predictions,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False))
    print(f"wrote {args.out} predictions={len(predictions)}", flush=True)


if __name__ == "__main__":
    main()
