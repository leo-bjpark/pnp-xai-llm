"""
Dataset pipeline persistence for Data Management.
- Pipelines: load from Hugging Face datasets, apply Data Processing, use in tasks.
- Stored in data/dataset_pipelines.json.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PIPELINES_FILE = DATA_DIR / "dataset_pipelines.json"


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_pipelines() -> List[Dict[str, Any]]:
    _ensure_data_dir()
    if not PIPELINES_FILE.exists():
        return []
    try:
        with open(PIPELINES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        return []


def _save_pipelines(pipelines: List[Dict[str, Any]]) -> None:
    _ensure_data_dir()
    with open(PIPELINES_FILE, "w", encoding="utf-8") as f:
        json.dump(pipelines, f, indent=2, ensure_ascii=False)


def get_pipelines() -> List[Dict[str, Any]]:
    """Return all dataset pipelines."""
    return _load_pipelines()


def get_pipeline_by_id(pipeline_id: str) -> Optional[Dict[str, Any]]:
    """Get a single pipeline by ID."""
    for p in _load_pipelines():
        if p.get("id") == pipeline_id:
            return p
    return None


def add_pipeline(name: str, status: str = "empty") -> str:
    """Create a new pipeline. Returns pipeline_id."""
    pipelines = _load_pipelines()
    pipeline_id = f"pipe_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(pipelines)}"
    pipelines.append({
        "id": pipeline_id,
        "name": name or "Unnamed",
        "status": status,
        "hf_dataset_path": "",
        "hf_load_options": {},
        "config": {},
        "dataset_info": None,
        "processing_code": "",
        "processed_dataset_info": None,
        "created_at": datetime.now().isoformat(),
    })
    _save_pipelines(pipelines)
    return pipeline_id


def update_pipeline(
    pipeline_id: str,
    name: Optional[str] = None,
    status: Optional[str] = None,
    hf_dataset_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    hf_load_options: Optional[Dict[str, Any]] = None,
    dataset_info: Optional[Dict[str, Any]] = None,
    processing_code: Optional[str] = None,
    processed_dataset_info: Optional[Dict[str, Any]] = None,
) -> bool:
    """Update pipeline fields."""
    pipelines = _load_pipelines()
    for p in pipelines:
        if p.get("id") == pipeline_id:
            if name is not None:
                p["name"] = name
            if status is not None:
                p["status"] = status
            if hf_dataset_path is not None:
                p["hf_dataset_path"] = hf_dataset_path
            if config is not None:
                p["config"] = config
            if hf_load_options is not None:
                p["hf_load_options"] = hf_load_options
            if dataset_info is not None:
                p["dataset_info"] = dataset_info
            if processing_code is not None:
                p["processing_code"] = processing_code
            if processed_dataset_info is not None:
                p["processed_dataset_info"] = processed_dataset_info
            _save_pipelines(pipelines)
            return True
    return False


def delete_pipeline(pipeline_id: str) -> bool:
    """Delete a pipeline by ID."""
    pipelines = _load_pipelines()
    for i, p in enumerate(pipelines):
        if p.get("id") == pipeline_id:
            pipelines.pop(i)
            _save_pipelines(pipelines)
            return True
    return False
