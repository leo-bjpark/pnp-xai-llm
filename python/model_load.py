"""
Model loading and management for PnPXAI Tool.
Manages Loaded Model + Treatment combination for memory-efficient session handling.
Models are listed from config.yaml (llms:); loading logic follows backup/utils.py.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Try to use backup utils if available
try:
    from utils import get_config_models, get_config_models_grouped, load_llm, get_model_status, get_model_device
except ImportError:
    # Fallback when running from project root with backup
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from backup.utils import get_config_models, get_config_models_grouped, load_llm, get_model_status, get_model_device
    except ImportError:
        from functools import lru_cache
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        def get_model_device(model: Any) -> torch.device:
            """Safely get the device of a model (avoids meta device)."""
            try:
                device = next(model.parameters()).device
                if device.type == "meta":
                    return torch.device("cpu")
                return device
            except Exception:
                return torch.device("cpu")

        # Project root config.yaml
        _CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
        _MODEL_ALIASES: Dict[str, str] = {}

        def get_config_models() -> List[str]:
            """Read model names from config.yaml llms: list or grouped dict."""
            grouped = get_config_models_grouped()
            return [m for models in grouped.values() for m in models]

        def get_config_models_grouped() -> Dict[str, List[str]]:
            """Read llms from config.yaml as grouped dict (group -> list of model ids)."""
            if not _CONFIG_PATH.exists():
                return {"": []}
            try:
                import yaml
                with open(_CONFIG_PATH, encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                llms = data.get("llms")
                if isinstance(llms, dict):
                    return {k: (v if isinstance(v, list) else []) for k, v in llms.items()}
                if isinstance(llms, list):
                    flat = [m for m in llms if isinstance(m, str)]
                    return {"": flat}
            except Exception:
                pass
            return {"": []}

        def _resolve_model_name(model_key: str) -> str:
            return _MODEL_ALIASES.get(model_key, model_key)

        def _load_base_model(base_model_name: str):
            """
            Load model on CUDA only.

            - Requires torch.cuda.is_available() to be True.
            - Loads weights directly on GPU using device_map="cuda"
              to avoid meta-tensor -> .to("cuda") 문제.
            """
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. GPU-only loading is requested but no GPU is visible.")

            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token

            try:
                # Load directly on CUDA to avoid meta tensor copy issues.
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda",
                    output_attentions=True,
                )
                model.eval()

                # Sanity check: one forward pass with attentions
                with torch.inference_mode():
                    enc = tokenizer("hello", return_tensors="pt").to("cuda")
                    out = model(**enc, output_attentions=True)
                if not out.attentions or any(a is None for a in out.attentions):
                    raise RuntimeError("Model attentions are missing or None")
            except Exception as e:
                raise RuntimeError(f"Failed to load model on GPU (cuda): {e}") from e

            return model, tokenizer

        @lru_cache(maxsize=2)
        def load_llm(model_key: str) -> Tuple[Any, Any]:
            allowed = set(get_config_models())
            if model_key not in allowed:
                raise ValueError(f"Unknown model key: {model_key}. Allowed: {sorted(allowed)}")
            model_name = _resolve_model_name(model_key)
            model, tokenizer = _load_base_model(model_name)
            return tokenizer, model

        def get_model_status(model_key: str) -> Dict[str, Any]:
            tokenizer, model = load_llm(model_key)
            config = model.config
            num_layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None)
            num_heads = getattr(config, "num_attention_heads", None) or getattr(config, "n_head", None)
            name = getattr(config, "name_or_path", model_key) or model_key
            num_params = sum(p.numel() for p in model.parameters())
            device_stats: List[Dict[str, Any]] = []
            seen: Dict[str, Dict[str, Any]] = {}
            for p in model.parameters():
                if p.device.type == "meta":
                    continue
                dev_key = str(p.device)
                if dev_key not in seen:
                    seen[dev_key] = {"device": dev_key, "memory_bytes": 0, "memory_gb": 0.0}
                seen[dev_key]["memory_bytes"] += p.numel() * (p.element_size() or 4)
            for dev_key, info in seen.items():
                info["memory_gb"] = round(info["memory_bytes"] / (1024**3), 3)
                if info["device"].startswith("cuda"):
                    idx = int(info["device"].split(":")[-1]) if ":" in info["device"] else 0
                    try:
                        info["capacity_gb"] = round(torch.cuda.get_device_properties(idx).total_memory / (1024**3), 2)
                    except Exception:
                        info["capacity_gb"] = None
                else:
                    info["capacity_gb"] = None
                device_stats.append(info)
            device_stats.sort(key=lambda x: (0 if x["device"] == "cpu" else 1, x["device"]))
            config_dict = {}
            if hasattr(config, "to_dict"):
                try:
                    config_dict = config.to_dict()
                except Exception:
                    config_dict = {}
            try:
                modules_str = str(model)
                max_lines = 4000
                lines = modules_str.split("\n")
                if len(lines) > max_lines:
                    modules_str = "\n".join(lines[:max_lines]) + "\n... (truncated)"
            except Exception:
                modules_str = ""
            return {
                "model_key": model_key,
                "name": name,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "num_parameters": num_params,
                "device_status": device_stats,
                "config": config_dict,
                "modules": modules_str,
            }


def get_available_models() -> List[str]:
    """Return list of available model keys from config."""
    return get_config_models()


def load_model(model_key: str) -> None:
    """
    Load model into GPU memory for the current session.

    Policy:
    - Before loading a new model, clear all previously cached models and CUDA memory.
      This ensures that only one HF model is resident on the GPU at a time.
    """
    # Drop any previously loaded models from the LRU cache and free CUDA memory
    clear_model_cache()
    # Load requested model (this will place it on GPU via _load_base_model)
    load_llm(model_key)


def get_status(model_key: str) -> Dict[str, Any]:
    """Get model status (layers, heads, etc.)."""
    return get_model_status(model_key)


def clear_model_cache() -> None:
    """Clear model load cache (LRU) and CUDA cache. Use after Empty Cache / reset."""
    import torch
    if hasattr(load_llm, "cache_clear"):
        load_llm.cache_clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
