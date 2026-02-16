"""
XAI Level 1 API handlers: Response Attribution + adversarial text generation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List

from python.xai_1.adversarial_text_generation import (
    DEFAULT_ITERATIONS,
    DEFAULT_PREFIX_LEN,
    MAX_ITERATIONS,
    MAX_PREFIX_LEN,
    find_adversarial_prefix,
)


def _parse_integer(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return default
    return max(min(iv, max_value), min_value)


def _parse_seed_ids(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,\s]+", value.strip())
    elif isinstance(value, Iterable):
        parts = [str(x).strip() for x in value]
    else:
        parts = [str(value).strip()]
    ids: List[int] = []
    for part in parts:
        if not part:
            continue
        try:
            num = int(part)
        except (TypeError, ValueError):
            continue
        ids.append(num)
    return ids


def _normalize_rows(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    parsed: List[Dict[str, Any]] = []
    raw_rows = []
    if isinstance(value, str):
        try:
            raw_rows = json.loads(value)
        except json.JSONDecodeError:
            raw_rows = []
    elif isinstance(value, (list, tuple)):
        raw_rows = list(value)
    elif isinstance(value, dict):
        raw_rows = [value]
    else:
        return []

    for idx, entry in enumerate(raw_rows):
        if not isinstance(entry, dict):
            continue
        row_id = entry.get("row_id") or entry.get("id") or f"row_{idx}"
        input_string = (entry.get("input_string") or entry.get("input") or "").strip()
        target_text = (entry.get("target_text") or entry.get("target") or "").strip()
        seed_value = entry.get("seed_ids") or entry.get("initial_prefix_ids") or entry.get("seeds")
        parsed.append(
            {
                "row_id": row_id,
                "input_string": input_string,
                "target_text": target_text,
                "seed_ids": _parse_seed_ids(seed_value),
                "seed_text": entry.get("seed_text") or entry.get("seed_sentence") or "",
            }
        )
    return parsed


def run_attribution(
    *,
    model: str,
    treatment: str,
    current_model: str,
    input_string: str,
    system_instruction: str,
    attribution_method: str,
    input_setting: Dict[str, Any],
) -> tuple[Dict[str, Any], int]:
    """
    Handle response attribution (1.0.1).
    Returns (result_dict, status_code).
    """
    try:
        temperature = float(input_setting.get("temperature", 0.7))
        max_new_tokens = int(input_setting.get("max_new_tokens", 256))
        top_p = float(input_setting.get("top_p", 1.0))
        top_k = int(input_setting.get("top_k", 50))
    except (TypeError, ValueError):
        return ({"error": "Invalid input_setting: temperature, max_new_tokens, top_p, top_k must be numbers"}, 400)

    try:
        from python.xai_1.input_attribution import compute_input_attribution

        result = compute_input_attribution(
            model_key=current_model,
            input_string=input_string,
            system_instruction=system_instruction,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            attribution_method=attribution_method,
        )
        result["status"] = "ok"
        result["model"] = model
        result["treatment"] = treatment
        return (result, 200)
    except Exception as e:
        return ({"error": str(e)}, 500)


def run_adversarial_text_generation(
    *,
    model: str,
    treatment: str,
    current_model: str,
    input_setting: Dict[str, Any],
) -> tuple[Dict[str, Any], int]:
    rows = _normalize_rows(input_setting.get("adversarial_rows"))
    if not rows:
        return ({"error": "adversarial_rows is required (list of {input_string, target_text})."}, 400)

    prefix_length = _parse_integer(input_setting.get("prefix_length"), DEFAULT_PREFIX_LEN, 1, MAX_PREFIX_LEN)
    iterations = _parse_integer(input_setting.get("iterations"), DEFAULT_ITERATIONS, 1, MAX_ITERATIONS)

    results: List[Dict[str, Any]] = []
    sanitized_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        row_id = row.get("row_id") or f"row_{idx}"
        input_string = (row.get("input_string") or "").strip()
        target_text = (row.get("target_text") or "").strip()
        seeds = row.get("seed_ids", [])
        seed_text = row.get("seed_text") or ""

        sanitized_rows.append(
            {
                "row_id": row_id,
                "input_string": input_string,
                "target_text": target_text,
                "seed_ids": seeds,
            }
        )

        if not input_string or not target_text:
            results.append(
                {
                    "row_id": row_id,
                    "row_index": idx,
                    "input_string": input_string,
                    "target_text": target_text,
                    "error": "input_string and target_text are required",
                }
            )
            continue

        try:
            attack_result = find_adversarial_prefix(
                model_key=current_model,
                input_string=input_string,
                target_text=target_text,
                prefix_length=prefix_length,
                iterations=iterations,
                initial_prefix_ids=seeds,
                initial_prefix_text=seed_text,
            )
            attack_result["row_id"] = row_id
            attack_result["row_index"] = idx
            attack_result["initial_prefix_ids"] = seeds
            attack_result["seed_text"] = seed_text
            attack_result["prefix_length"] = prefix_length
            attack_result["iterations"] = iterations
            results.append(attack_result)
        except Exception as exc:
            results.append(
                {
                    "row_id": row_id,
                    "row_index": idx,
                    "input_string": input_string,
                    "target_text": target_text,
                    "error": str(exc),
                }
            )

    response = {
        "status": "ok",
        "model": model,
        "treatment": treatment,
        "prefix_length": prefix_length,
        "iterations": iterations,
        "adversarial_rows": sanitized_rows,
        "adversarial_results": results,
    }
    return (response, 200)
