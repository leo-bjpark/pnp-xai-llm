from flask import Blueprint, jsonify, request

from python.memory import cache_store
from python.model_load import (
    clear_model_cache,
    get_config_models,
    get_config_models_grouped,
    get_model_status,
    load_model,
)


session_bp = Blueprint("session", __name__)


@session_bp.get("/api/session")
def api_get_session():
    """Return current session state: loaded_model, treatment."""
    sess = cache_store.get_session()
    return jsonify(
        {
            "loaded_model": sess["loaded_model"],
            "treatment": sess["treatment"],
        }
    )


@session_bp.post("/api/session")
def api_set_session():
    """Set session (after user confirms model load)."""
    data = request.get_json(force=True) or {}
    model = data.get("loaded_model")
    treatment = data.get("treatment", "")
    cache_store.set_session(model, treatment)
    return jsonify({"status": "ok"})


@session_bp.get("/api/models")
def api_models():
    return jsonify({"models": get_config_models()})


@session_bp.post("/api/load_model")
def api_load_model():
    """Load model and update session."""
    data = request.get_json(force=True) or {}
    model_key = data.get("model", "")
    treatment = data.get("treatment", cache_store.get_session().get("treatment") or "")
    if not model_key:
        return jsonify({"error": "model required"}), 400
    try:
        load_model(model_key)
        cache_store.set_session(model_key, treatment)
        # Apply Simple Steering treatment hooks, if treatment is a JSON config.
        try:
            from python.treatments.simple_steering import apply_simple_steering_from_string

            apply_simple_steering_from_string(model_key, treatment)
        except Exception:
            # Treatment errors should not break model loading.
            pass
        return jsonify({"status": "ok", "model": model_key})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@session_bp.get("/api/model_status")
def api_model_status():
    model_key = request.args.get("model")
    if not model_key:
        return jsonify({"error": "model query param required"}), 400
    try:
        status = get_model_status(model_key)
        return jsonify(status)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@session_bp.get("/api/cuda_env")
def api_get_cuda_env():
    """Return current CUDA_VISIBLE_DEVICES (echo result)."""
    import os

    value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return jsonify({"CUDA_VISIBLE_DEVICES": value, "echo": value})


@session_bp.post("/api/cuda_env")
def api_set_cuda_env():
    """Set CUDA_VISIBLE_DEVICES and return new echo result."""
    import os

    data = request.get_json(force=True) or {}
    value = data.get("value", "")
    if value is None:
        value = ""
    value = str(value).strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    return jsonify({"CUDA_VISIBLE_DEVICES": value, "echo": value})

