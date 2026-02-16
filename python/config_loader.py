"""Load config.yaml and expose XAI_LEVEL_NAMES."""

from pathlib import Path
from typing import Dict, List, Tuple

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _load_xai_level_list() -> List[Tuple[str, str]]:
    """
    Parse XAI_LEVEL_NAMES from config as ordered list of (name, name).
    Uses task name as key directly (no numeric IDs).
    """
    if not CONFIG_PATH.exists():
        return [
            ("Positive and Negative Response Preference", "Positive and Negative Response Preference"),
            ("Response Attribution", "Response Attribution"),
        ]

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw = data.get("XAI_LEVEL_NAMES")
    if not raw:
        return []

    result: List[Tuple[str, str]] = []

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            for k, v in item.items():
                key = str(k).strip()
                if key.upper().startswith("LEVEL_") and isinstance(v, list):
                    for name in v:
                        label = str(name).strip()
                        if not label:
                            continue
                        result.append((label, label))
                else:
                    result.append((str(k).strip(), str(v).strip()))
    elif isinstance(raw, dict):
        for k, v in raw.items():
            result.append((str(k).strip(), str(v).strip()))
    return result


def get_xai_level_names_grouped() -> Dict[int, List[Tuple[str, str]]]:
    """
    Return levels grouped by category index (0, 1, 2) for layout.
    Values are (name, name) - name used as key directly.
    """
    ordered = _load_xai_level_list()
    grouped: Dict[int, List[Tuple[str, str]]] = {}
    group_idx = 0
    if not ordered:
        return {}

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("XAI_LEVEL_NAMES") or []

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            for k, v in item.items():
                if str(k).strip().upper().startswith("LEVEL_") and isinstance(v, list):
                    items: List[Tuple[str, str]] = []
                    for name in v:
                        label = str(name).strip()
                        if label:
                            items.append((label, label))
                    if items:
                        grouped[group_idx] = items
                        group_idx += 1

    return grouped


def get_xai_level_names() -> Dict[str, str]:
    """
    Return task name -> display name mapping.
    Uses name as key (same as value).
    """
    ordered = _load_xai_level_list()
    return dict(ordered)


def get_dataset_groups() -> Dict[str, List[str]]:
    """Return dataset groups from config.yaml DATASETS section."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("DATASETS") or {}
    if isinstance(raw, dict):
        out: Dict[str, List[str]] = {}
        for k, v in raw.items():
            if isinstance(v, list):
                out[str(k).strip()] = [str(x).strip() for x in v if str(x).strip()]
        return out
    return {}


def get_dataset_list() -> List[str]:
    """Flattened dataset list from config.yaml DATASETS section."""
    groups = get_dataset_groups()
    flat: List[str] = []
    for _, items in groups.items():
        flat.extend(items)
    return flat


def get_analyzer_list() -> List[str]:
    """Return list of analyzer names from config.yaml ANALYZER section."""
    if not CONFIG_PATH.exists():
        return []
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("ANALYZER") or []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []
