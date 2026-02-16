from flask import Blueprint, jsonify

from python.analyzer.layer_residual_pca import compute_residual_pca_singular_values
from python.memory.variable import get_residual_variable, has_variable_pickle, load_variable_pickle


analyzer_bp = Blueprint("analyzer", __name__)


@analyzer_bp.get("/api/analyzer/layer-residual-pca/<path:var_id>")
def api_layer_residual_pca(var_id: str):
  """
  Run Layer Residual PCA on a saved residual variable.

  Loads variable from pickle (if available) or from residual store, then
  computes singular values of stacked directions.
  """
  payload = None
  if has_variable_pickle(var_id):
    payload = load_variable_pickle(var_id)
  if not isinstance(payload, dict):
    rv = get_residual_variable(var_id)
    if not rv:
      return jsonify({"error": "Variable not found"}), 404
    payload = rv

  directions = payload.get("directions") or {}
  # Backwards-compat: some variables may store direction_vectors + layer_names
  if not directions and payload.get("direction_vectors") and payload.get("layer_names"):
    directions = {
      name: vec for name, vec in zip(payload["layer_names"], payload["direction_vectors"])
    }

  if not isinstance(directions, dict) or not directions:
    return jsonify({"error": "No directions in variable"}), 400

  try:
    result = compute_residual_pca_singular_values(directions)
  except ValueError as exc:
    return jsonify({"error": str(exc)}), 400
  except Exception as exc:  # noqa: BLE001
    return jsonify({"error": f"PCA failed: {exc}"}), 500

  result["variable_id"] = var_id
  result["task_name"] = payload.get("task_name")
  result["model"] = payload.get("model")
  return jsonify(result)

