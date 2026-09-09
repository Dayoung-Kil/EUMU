"""Small shared helpers for EUMU tagging recipe scripts."""
from __future__ import annotations

import json
from pathlib import Path

NAMESPACES = ("quality", "scene", "event")


def load_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text())


def load_vocab(path: str | Path) -> dict[str, list[str]]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"vocab must be a JSON object: {path}")
    return {ns: list(raw.get(ns, [])) for ns in NAMESPACES}


def parse_tag_target(target: str, vocab: dict[str, list[str]]) -> dict[str, list[str]]:
    allowed = {ns: set(vocab.get(ns, [])) for ns in NAMESPACES}
    out = {ns: [] for ns in NAMESPACES}
    for part in str(target or "").split("|"):
        if ":" not in part:
            continue
        ns, values = part.split(":", 1)
        ns = ns.strip().lower()
        if ns not in out:
            continue
        seen: set[str] = set()
        for value in values.split(","):
            label = value.strip().lower()
            if not label or label == "none":
                continue
            if label in allowed[ns] and label not in seen:
                out[ns].append(label)
                seen.add(label)
    return out


def row_annotated_namespaces(row: dict, target: str | None = None) -> set[str]:
    if isinstance(row.get("namespace"), str) and row["namespace"] in NAMESPACES:
        return {row["namespace"]}
    if isinstance(row.get("namespaces"), list):
        return {str(x) for x in row["namespaces"] if str(x) in NAMESPACES}
    if isinstance(row.get("annotated_namespaces"), list):
        return {str(x) for x in row["annotated_namespaces"] if str(x) in NAMESPACES}

    annotated: set[str] = set()
    for ns in NAMESPACES:
        if isinstance(row.get(ns), (list, str)):
            annotated.add(ns)

    if not annotated and target:
        for part in str(target).split("|"):
            if ":" not in part:
                continue
            ns, values = part.split(":", 1)
            ns = ns.strip().lower()
            labels = [x.strip().lower() for x in values.split(",")]
            has_label = any(label and label != "none" for label in labels)
            if ns in NAMESPACES and has_label:
                annotated.add(ns)
    return annotated
