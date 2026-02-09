"""
Simple Steering treatment.

Applies residual-direction steering vectors saved in residual_variables.json
as forward hooks on the loaded HF model.

Treatment format (stored as JSON string in the `treatment` session field):

{
  "type": "simple_steering",
  "residual_var": "<variable name>",
  "normalize": true,
  "alpha": 1.0,
  "delta": 0.0,  # optional: apply only when |cos sim| ≥ delta
  "layer_keys": [
    "model.layers.0.mlp_block_out",
    "model.layers.5.mlp_block_out"
  ]
}

Notes:
- Only keys ending with ".mlp_block_out" are currently supported.
- Hooks are attached to the corresponding layer module (e.g. "model.layers.0").
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import torch

from python.memory.variable import get_residual_variable
from python.model_load import load_llm


_ACTIVE_HOOKS: Dict[str, List[torch.utils.hooks.RemovableHandle]] = {}


def _clear_all_hooks() -> None:
    """Remove all registered steering hooks (for all models)."""
    for handles in _ACTIVE_HOOKS.values():
        for h in handles:
            try:
                h.remove()
            except Exception:
                # Best-effort cleanup
                pass
    _ACTIVE_HOOKS.clear()


def _get_param_dtype(model: torch.nn.Module) -> torch.dtype:
    for p in model.parameters():
        return p.dtype
    return torch.float32


def _apply_vector(
    h: torch.Tensor,
    v: torch.Tensor,
    alpha: float,
    normalize: bool,
    delta: float,
) -> torch.Tensor:
    """
    Core steering operation:
        h <- h + alpha * v
    and, when normalize=True, per-token norm is preserved.
    """
    v_local = v.to(device=h.device, dtype=h.dtype)
    while v_local.dim() < h.dim():
        v_local = v_local.unsqueeze(0)

    # Conditional steering based on |cosine similarity| between h and v.
    # When delta == 0, this reduces to always applying steering.
    eps = 1e-6
    # Normalize v along the last dimension (after broadcasting shape).
    v_norm = v_local.norm(dim=-1, keepdim=True).clamp(min=eps)
    v_unit = v_local / v_norm
    # Per-token norm and cosine similarity
    h_norm = h.norm(dim=-1, keepdim=True).clamp(min=eps)
    cos_sim = (h * v_unit).sum(dim=-1, keepdim=True) / h_norm
    mask = (cos_sim.abs() >= float(delta)).to(h.dtype)

    h_orig = h
    # Only positions with |cos_sim| >= delta get the full alpha; others get 0.
    masked_alpha = alpha * mask
    h_new = h_orig + masked_alpha * v_local
    if normalize:
        orig_norm = h_norm
        new_norm = h_new.norm(dim=-1, keepdim=True).clamp(min=eps)
        scale = orig_norm / new_norm
        h_new = h_new * scale
    return h_new


def _make_forward_hook(
    v: torch.Tensor,
    alpha: float,
    normalize: bool,
    delta: float,
) -> Any:
    """Forward hook: modify module output."""

    v = v.detach().clone()

    def hook(_mod, _inp, outp):
        if outp is None:
            return outp
        t = outp[0] if isinstance(outp, (tuple, list)) else outp
        if t is None:
            return outp

        h_new = _apply_vector(t, v, alpha, normalize, delta)
        if isinstance(outp, tuple):
            return (h_new,) + outp[1:]
        if isinstance(outp, list):
            outp[0] = h_new
            return outp
        return h_new

    return hook


def _make_forward_pre_hook(
    v: torch.Tensor,
    alpha: float,
    normalize: bool,
    delta: float,
) -> Any:
    """Forward-pre hook: modify module input (tuple or tensor)."""

    v = v.detach().clone()

    def hook(_mod, inp):
        if inp is None:
            return inp
        x = inp[0] if isinstance(inp, (tuple, list)) else inp
        if x is None:
            return inp

        h_new = _apply_vector(x, v, alpha, normalize, delta)
        if isinstance(inp, tuple):
            return (h_new,) + inp[1:]
        if isinstance(inp, list):
            inp[0] = h_new
            return inp
        return (h_new,)

    return hook


def _discover_attn_mlp_modules(
    model: torch.nn.Module,
    layer_prefix: str,
) -> tuple[Optional[torch.nn.Module], Optional[torch.nn.Module]]:
    """
    Heuristically find attention and MLP modules under a given layer prefix.

    This mirrors the logic used when detecting layer structure:
    - first child whose name contains 'attn' is treated as attention block
    - first child whose name contains 'mlp' is treated as MLP block
    """
    full_prefix = layer_prefix + "."
    attn_mod = None
    mlp_mod = None
    for name, mod in model.named_modules():
        if not name or not name.startswith(full_prefix):
            continue
        rest = name[len(full_prefix) :]
        head = rest.split(".")[0] if "." in rest else rest
        low = head.lower()
        if attn_mod is None and "attn" in low:
            attn_mod = mod
        if mlp_mod is None and "mlp" in low:
            mlp_mod = mod
        if attn_mod is not None and mlp_mod is not None:
            break
    return attn_mod, mlp_mod


def apply_simple_steering(model_key: str, cfg: Dict[str, Any]) -> None:
    """
    Apply Simple Steering hooks for a given model and config dict.
    Existing steering hooks (for any model) are cleared first.
    """
    _clear_all_hooks()

    if not isinstance(cfg, dict):
        return
    if cfg.get("type") != "simple_steering":
        return

    residual_var = (cfg.get("residual_var") or "").strip()
    if not residual_var:
        return

    rv = get_residual_variable(residual_var)
    if not rv:
        return

    directions = rv.get("directions") or {}
    if not isinstance(directions, dict) or not directions:
        return

    try:
        alpha = float(cfg.get("alpha", 1.0))
    except (TypeError, ValueError):
        alpha = 1.0
    normalize = bool(cfg.get("normalize", True))
    try:
        delta = float(cfg.get("delta", 0.0))
    except (TypeError, ValueError):
        delta = 0.0
    if delta < 0:
        delta = 0.0
    if delta > 1:
        delta = 1.0

    raw_layer_keys = cfg.get("layer_keys")
    if isinstance(raw_layer_keys, list) and raw_layer_keys:
        layer_keys = [str(k) for k in raw_layer_keys if str(k) in directions]
    else:
        # Default: all supported keys from residual variable
        layer_keys = [k for k in directions.keys()]

    if not layer_keys:
        return

    # Get shared model instance from load_llm cache
    _, model = load_llm(model_key)
    dtype = _get_param_dtype(model)
    handles: List[torch.utils.hooks.RemovableHandle] = []

    named_modules = dict(model.named_modules())

    for key in layer_keys:
        vec = directions.get(key)
        if not isinstance(vec, (list, tuple)):
            continue
        try:
            v_tensor = torch.tensor(vec, dtype=dtype)
        except Exception:
            continue

        # key format examples:
        #   "<layer_prefix>.attn_out"
        #   "<layer_prefix>.attn_block_out"
        #   "<layer_prefix>.mlp_out"
        #   "<layer_prefix>.mlp_block_out"
        if "." not in key:
            continue
        layer_prefix, kind = key.rsplit(".", 1)
        layer_mod = named_modules.get(layer_prefix)
        attn_mod, mlp_mod = _discover_attn_mlp_modules(model, layer_prefix)

        hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        if kind == "attn_out" and attn_mod is not None:
            hook_handle = attn_mod.register_forward_hook(
                _make_forward_hook(v_tensor, alpha, normalize, delta)
            )
        elif kind == "attn_block_out" and mlp_mod is not None:
            # Residual + attn output (MLP input)
            hook_handle = mlp_mod.register_forward_pre_hook(
                _make_forward_pre_hook(v_tensor, alpha, normalize, delta)
            )
        elif kind == "mlp_out" and mlp_mod is not None:
            hook_handle = mlp_mod.register_forward_hook(
                _make_forward_hook(v_tensor, alpha, normalize, delta)
            )
        elif kind == "mlp_block_out" and layer_mod is not None:
            hook_handle = layer_mod.register_forward_hook(
                _make_forward_hook(v_tensor, alpha, normalize, delta)
            )

        if hook_handle is not None:
            handles.append(hook_handle)

    if handles:
        _ACTIVE_HOOKS[model_key] = handles


def clear_simple_steering() -> None:
    """Public helper to clear all Simple Steering hooks."""
    _clear_all_hooks()


def apply_simple_steering_from_string(model_key: str, treatment_str: Optional[str]) -> None:
    """
    Parse treatment string as JSON and, if it is a Simple Steering config,
    apply hooks. If the string is empty or invalid, all hooks are cleared.
    """
    if not treatment_str:
        _clear_all_hooks()
        return

    try:
        cfg = json.loads(treatment_str)
    except Exception:
        # If it isn't JSON, treat as no structured treatment; clear hooks.
        _clear_all_hooks()
        return

    apply_simple_steering(model_key, cfg)

