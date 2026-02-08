"""
PnP-XAI-LLM - VSCode-like XAI analysis tool.
"""

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Ensure project root is on path for backup.utils
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, redirect, render_template, request

# Model list and loading via python.model_load (reads config.yaml, uses backup.utils internally)
from python.model_load import clear_model_cache, get_config_models, get_config_models_grouped, load_model, get_model_status

from python.config_loader import get_xai_level_names, get_xai_level_names_grouped
from python.session_store import (
    add_task,
    get_task_by_id,
    get_tasks,
    update_task_title,
    update_task_result,
    delete_task,
    get_raw_memory,
    import_memory,
)
from python.dataset_pipeline_store import (
    get_pipelines,
    get_pipeline_by_id,
    add_pipeline,
    update_pipeline as update_pipeline_store,
    delete_pipeline,
)

app = Flask(__name__)

# In-memory session: current Loaded Model + Treatment
# Used for memory-efficient management and run confirmation
SESSION_STATE = {
    "loaded_model": None,
    "treatment": None,
}

# Conversation cache for 0.1.2: conversation_id -> {"model_key": str, "messages": [{"role", "content"}]}
CONVERSATION_CACHE = {}

# Global saved data variables: variable_name (dataName/split/randomN/seed/taskName) -> snapshot
SAVED_DATA_VARS = {}


@app.get("/panel")
def panel():
    """Standalone right panel window (opens in separate window, independent of main page)."""
    return render_template("panel.html")


@app.get("/")
def index():
    """Main IDE-like interface."""
    models = get_config_models()
    models_grouped = get_config_models_grouped()
    xai_level_names = get_xai_level_names()
    xai_level_grouped = get_xai_level_names_grouped()
    tasks = get_tasks(xai_level_names)
    dataset_pipelines = get_pipelines()
    return render_template(
        "index.html",
        models=models,
        models_grouped=models_grouped,
        tasks=tasks,
        xai_level_names=xai_level_names,
        xai_level_grouped=xai_level_grouped,
        dataset_pipelines=dataset_pipelines,
    )


@app.get("/task/<task_id>")
def task_view(task_id):
    """Generic task view - fetches task and renders by xai_level."""
    task = get_task_by_id(task_id)
    if not task:
        models = get_config_models()
        models_grouped = get_config_models_grouped()
        xai_level_names = get_xai_level_names()
        return render_template(
            "index.html",
            models=models,
            models_grouped=models_grouped,
            tasks=get_tasks(xai_level_names),
            xai_level_names=xai_level_names,
            xai_level_grouped=get_xai_level_names_grouped(),
            dataset_pipelines=get_pipelines(),
            error="Task not found",
        )
    level_key = task.get("xai_level", "0.1")
    return _render_task(task_id, level_key)




def _task_template(level_key: str) -> str:
    """Template name for task view by XAI level."""
    if level_key == "0.1.1":
        return "XAI_0_1_1_completion.html"
    if level_key == "0.1.2":
        return "XAI_0_1_2_conversation.html"
    if level_key == "1.0.1":
        return "XAI_1_1_response_attribution.html"
    return "XAI_not_implemented.html"


def _render_task(task_id: str, level_key: str):
    task = get_task_by_id(task_id)
    if not task:
        models = get_config_models()
        models_grouped = get_config_models_grouped()
        xai_level_names = get_xai_level_names()
        return render_template(
            "index.html",
            models=models,
            models_grouped=models_grouped,
            tasks=get_tasks(xai_level_names),
            xai_level_names=xai_level_names,
            xai_level_grouped=get_xai_level_names_grouped(),
            dataset_pipelines=get_pipelines(),
            error="Task not found",
        )
    models = get_config_models()
    models_grouped = get_config_models_grouped()
    xai_level_names = get_xai_level_names()
    tasks = get_tasks(xai_level_names)
    dataset_pipelines = get_pipelines()
    template = _task_template(level_key)
    return render_template(
        template,
        task=task,
        models=models,
        models_grouped=models_grouped,
        tasks=tasks,
        xai_level_names=xai_level_names,
        xai_level_grouped=get_xai_level_names_grouped(),
        dataset_pipelines=dataset_pipelines,
    )


# ----- Data Management: Dataset Pipeline -----

@app.get("/data")
def data_index():
    """Redirect to first pipeline or index if none."""
    pipelines = get_pipelines()
    if pipelines:
        return redirect(f"/data/{pipelines[0]['id']}", code=302)
    return redirect("/", code=302)


@app.get("/data/<pipeline_id>")
def data_pipeline_view(pipeline_id: str):
    """Dataset pipeline panel: load HF dataset, Data Processing, view data."""
    pipeline = get_pipeline_by_id(pipeline_id)
    if not pipeline:
        models = get_config_models()
        models_grouped = get_config_models_grouped()
        xai_level_names = get_xai_level_names()
        return render_template(
            "index.html",
            models=models,
            models_grouped=models_grouped,
            tasks=get_tasks(xai_level_names),
            xai_level_names=xai_level_names,
            xai_level_grouped=get_xai_level_names_grouped(),
            dataset_pipelines=get_pipelines(),
            error="Pipeline not found",
        )
    models = get_config_models()
    models_grouped = get_config_models_grouped()
    xai_level_names = get_xai_level_names()
    tasks = get_tasks(xai_level_names)
    dataset_pipelines = get_pipelines()
    processing_code = pipeline.get("processing_code") or (
        "def process(example):\n"
        "    \"\"\"Transform one example. Used as dataset.map(process).\"\"\"\n"
        "    return example\n"
    )
    return render_template(
        "data_pipeline_view.html",
        pipeline=pipeline,
        processing_code=processing_code,
        models=models,
        models_grouped=models_grouped,
        tasks=tasks,
        xai_level_names=xai_level_names,
        xai_level_grouped=get_xai_level_names_grouped(),
        dataset_pipelines=dataset_pipelines,
    )


# ----- API: Tasks -----

@app.get("/api/tasks")
def api_get_tasks():
    xai_level_names = get_xai_level_names()
    return jsonify({"tasks": get_tasks(xai_level_names), "xai_level_names": xai_level_names})


@app.get("/api/tasks/<task_id>")
def api_get_task(task_id):
    task = get_task_by_id(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(task)


@app.post("/api/tasks")
def api_create_task():
    data = request.get_json(force=True) or {}
    xai_level = data.get("xai_level", "0.1")
    title = data.get("title", "Untitled Task")
    model = data.get("model", "")
    treatment = data.get("treatment", "")
    result = data.get("result", {})

    task_id = add_task(xai_level, title, model, treatment, result)
    return jsonify({"task_id": task_id, "status": "ok"})


@app.patch("/api/tasks/<task_id>")
def api_update_task(task_id):
    data = request.get_json(force=True) or {}
    title = data.get("title")
    result = data.get("result")
    model = data.get("model")
    treatment = data.get("treatment")
    if title is not None:
        if update_task_title(task_id, title):
            return jsonify({"status": "ok"})
    if result is not None:
        if update_task_result(task_id, result, model=model, treatment=treatment):
            return jsonify({"status": "ok"})
    return jsonify({"error": "Update failed"}), 400


@app.delete("/api/tasks/<task_id>")
def api_delete_task(task_id):
    if delete_task(task_id):
        return jsonify({"status": "ok"})
    return jsonify({"error": "Task not found"}), 404


# ----- API: Session (Loaded Model + Treatment) -----

@app.get("/api/session")
def api_get_session():
    """Return current session state: loaded_model, treatment."""
    return jsonify({
        "loaded_model": SESSION_STATE["loaded_model"],
        "treatment": SESSION_STATE["treatment"],
    })


@app.post("/api/session")
def api_set_session():
    """Set session (after user confirms model load)."""
    data = request.get_json(force=True) or {}
    model = data.get("loaded_model")
    treatment = data.get("treatment", "")
    SESSION_STATE["loaded_model"] = model
    SESSION_STATE["treatment"] = treatment
    return jsonify({"status": "ok"})


# ----- API: Models -----

@app.get("/api/models")
def api_models():
    return jsonify({"models": get_config_models()})


@app.post("/api/load_model")
def api_load_model():
    """Load model and update session."""
    data = request.get_json(force=True) or {}
    model_key = data.get("model", "")
    treatment = data.get("treatment", SESSION_STATE.get("treatment") or "")
    if not model_key:
        return jsonify({"error": "model required"}), 400
    try:
        load_model(model_key)
        SESSION_STATE["loaded_model"] = model_key
        SESSION_STATE["treatment"] = treatment
        return jsonify({"status": "ok", "model": model_key})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/model_status")
def api_model_status():
    model_key = request.args.get("model")
    if not model_key:
        return jsonify({"error": "model query param required"}), 400
    try:
        status = get_model_status(model_key)
        return jsonify(status)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/cuda_env")
def api_get_cuda_env():
    """Return current CUDA_VISIBLE_DEVICES (echo result)."""
    import os
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return jsonify({"CUDA_VISIBLE_DEVICES": value, "echo": value})


@app.post("/api/cuda_env")
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


# ----- API: Run (placeholder - extend for actual XAI computation) -----

# ----- API: Memory Export/Import -----

@app.get("/api/memory/export")
def api_memory_export():
    """Export full memory as JSON (for download)."""
    from flask import Response
    data = get_raw_memory()
    data["session"] = SESSION_STATE.copy()
    # Filename: XAI_Level_Export_{timestamp}.json
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Response(
        json.dumps(data, indent=2, ensure_ascii=False),
        mimetype="application/json",
        headers={"Content-Disposition": f"attachment; filename=XAI_Level_Export_{ts}.json"},
    )


@app.post("/api/memory/import")
def api_memory_import():
    """Import memory from JSON. Expects JSON in request body."""
    data = request.get_json(force=True) or {}
    if not import_memory(data):
        return jsonify({"error": "Invalid format. Expected { tasks: {...} }"}), 400
    # Optionally restore session from imported data
    sess = data.get("session")
    if isinstance(sess, dict):
        SESSION_STATE["loaded_model"] = sess.get("loaded_model")
        SESSION_STATE["treatment"] = sess.get("treatment")
    return jsonify({"status": "ok"})


# ----- API: Run -----

@app.post("/api/run")
def api_run():
    """
    Execute XAI analysis.
    Client should ensure session matches input (model + treatment) before calling,
    or handle the confirm dialog when mismatch.
    For level 0.1.1 (Completion), runs text_completion with input_setting params.
    """
    data = request.get_json(force=True) or {}
    model = data.get("model", "")
    treatment = data.get("treatment", "")
    input_setting = data.get("input_setting", {})

    # Check session consistency
    current_model = SESSION_STATE.get("loaded_model")
    current_treatment = SESSION_STATE.get("treatment")

    if model != current_model or treatment != current_treatment:
        return jsonify({
            "error": "session_mismatch",
            "message": "Loaded Model + Treatment does not match the current session. Load the model with this setting?",
            "requested": {"model": model, "treatment": treatment},
            "current": {"model": current_model, "treatment": current_treatment},
        }), 400

    # Conversation (0.1.2): cache-backed multi-turn chat; return input_tokens, generated_tokens, conversation_id
    conversation_id = input_setting.get("conversation_id") or None
    content = (input_setting.get("content") or "").strip()
    messages_input = input_setting.get("messages")
    system_instruction = (input_setting.get("system_instruction") or "").strip()

    def _ensure_system_at_start(messages: list) -> list:
        """Put current system_instruction at the very start of messages (strip any existing leading system)."""
        if not system_instruction:
            return messages
        # Drop any existing leading system message so we use the current one
        rest = messages
        while rest and (rest[0].get("role") or "").lower() == "system":
            rest = rest[1:]
        return [{"role": "system", "content": system_instruction}] + rest

    # Enter conversation branch when there is new content or a full messages list (first message has content but no id)
    if current_model and (content or (messages_input and isinstance(messages_input, list))):
        try:
            from python.model_generation import chat_completion, get_cache_token_count, get_text_token_count
            temperature = float(input_setting.get("temperature", 0.7))
            max_new_tokens = int(input_setting.get("max_new_tokens", 256))
            top_p = float(input_setting.get("top_p", 1.0))
            top_k = int(input_setting.get("top_k", 50))
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid input_setting: temperature, max_new_tokens, top_p, top_k must be numbers"}), 400

        if conversation_id and content:
            cached = CONVERSATION_CACHE.get(conversation_id)
            if not cached or cached.get("model_key") != current_model:
                messages = _ensure_system_at_start([{"role": "user", "content": content}])
                conversation_id = None
            else:
                messages = _ensure_system_at_start(
                    list(cached["messages"]) + [{"role": "user", "content": content}]
                )
        elif content:
            messages = _ensure_system_at_start([{"role": "user", "content": content}])
            conversation_id = None
        elif messages_input:
            messages = [{"role": m.get("role", "user"), "content": (m.get("content") or "").strip()} for m in messages_input if (m.get("content") or "").strip()]
            if not messages:
                return jsonify({"error": "messages must contain at least one non-empty message"}), 400
            messages = _ensure_system_at_start(messages)
            conversation_id = None
        else:
            return jsonify({"error": "Provide conversation_id and content, or messages list"}), 400

        try:
            input_tokens = get_cache_token_count(current_model, messages)
            do_sample = temperature > 0
            generated_text = chat_completion(
                model_key=current_model,
                messages=messages,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            generated_tokens = get_text_token_count(current_model, generated_text)
            messages.append({"role": "assistant", "content": generated_text})
            if conversation_id is None:
                conversation_id = str(uuid.uuid4())
            CONVERSATION_CACHE[conversation_id] = {"model_key": current_model, "messages": messages}

            # conversation_list for saving/restore: instruction + messages as user/ai
            conversation_list = {
                "instruction": system_instruction,
                "messages": [
                    {"role": "user" if m.get("role") == "user" else "ai", "content": (m.get("content") or "").strip()}
                    for m in messages
                    if (m.get("role") or "").lower() in ("user", "assistant")
                ],
            }

            result = {
                "status": "ok",
                "model": model,
                "treatment": treatment,
                "generated_text": generated_text,
                "input_tokens": input_tokens,
                "generated_tokens": generated_tokens,
                "conversation_id": conversation_id,
                "cache_message_count": len(messages),
                "conversation_list": conversation_list,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "top_p": top_p,
                "top_k": top_k,
            }
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Completion (0.1.1) or Response Attribution (1.0.1): input_string + generation params
    input_string = (input_setting.get("input_string") or "").strip()
    system_instruction = (input_setting.get("system_instruction") or "").strip()
    attribution_method = (input_setting.get("attribution_method") or "").strip()
    if "input_string" in input_setting and current_model:
        try:
            temperature = float(input_setting.get("temperature", 0.7))
            max_new_tokens = int(input_setting.get("max_new_tokens", 256))
            top_p = float(input_setting.get("top_p", 1.0))
            top_k = int(input_setting.get("top_k", 50))
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid input_setting: temperature, max_new_tokens, top_p, top_k must be numbers"}), 400
        if attribution_method:
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
                return jsonify(result)
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        try:
            from python.model_generation import text_completion
            generated_text = text_completion(
                model_key=current_model,
                prompt=input_string,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
            )
            result = {
                "status": "ok",
                "model": model,
                "treatment": treatment,
                "input_string": input_string,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "generated_text": generated_text,
            }
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Fallback: placeholder for other levels
    result = {
        "status": "ok",
        "message": "Analysis complete. (Actual XAI computation to be implemented)",
        "model": model,
        "treatment": treatment,
        "input_setting": input_setting,
    }
    return jsonify(result)


@app.post("/api/conversation/clear")
def api_conversation_clear():
    """Clear server-side conversation cache for the given conversation_id (0.1.2)."""
    data = request.get_json(force=True) or {}
    conversation_id = data.get("conversation_id")
    if conversation_id and conversation_id in CONVERSATION_CACHE:
        del CONVERSATION_CACHE[conversation_id]
    return jsonify({"status": "ok"})


@app.post("/api/empty_cache")
def api_empty_cache():
    """
    Reset: clear loaded model, session state, conversation cache, and CUDA cache.
    Call after user confirms Empty Cache (warning shown in UI).
    """
    SESSION_STATE["loaded_model"] = None
    SESSION_STATE["treatment"] = None
    CONVERSATION_CACHE.clear()
    clear_model_cache()
    return jsonify({"status": "ok"})


# ----- API: Dataset Pipelines -----

@app.get("/api/dataset-pipelines")
def api_list_pipelines():
    return jsonify({"pipelines": get_pipelines()})


@app.post("/api/dataset-pipelines")
def api_create_pipeline():
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip() or "Unnamed"
    pipeline_id = add_pipeline(name=name, status="empty")
    pipeline = get_pipeline_by_id(pipeline_id)
    return jsonify({"status": "ok", "pipeline": pipeline, "id": pipeline_id})


@app.get("/api/dataset-pipelines/<pipeline_id>")
def api_get_pipeline(pipeline_id: str):
    pipeline = get_pipeline_by_id(pipeline_id)
    if not pipeline:
        return jsonify({"error": "Pipeline not found"}), 404
    return jsonify(pipeline)


@app.patch("/api/dataset-pipelines/<pipeline_id>")
def api_update_pipeline(pipeline_id: str):
    pipeline = get_pipeline_by_id(pipeline_id)
    if not pipeline:
        return jsonify({"error": "Pipeline not found"}), 404
    data = request.get_json(force=True) or {}
    if "name" in data:
        update_pipeline_store(pipeline_id, name=data.get("name"))
    if "status" in data:
        update_pipeline_store(pipeline_id, status=data.get("status"))
    if "hf_dataset_path" in data:
        update_pipeline_store(pipeline_id, hf_dataset_path=data.get("hf_dataset_path"))
    if "config" in data:
        update_pipeline_store(pipeline_id, config=data.get("config"))
    updated = get_pipeline_by_id(pipeline_id)
    return jsonify(updated)


@app.delete("/api/dataset-pipelines/<pipeline_id>")
def api_delete_pipeline(pipeline_id: str):
    if not get_pipeline_by_id(pipeline_id):
        return jsonify({"error": "Pipeline not found"}), 404
    delete_pipeline(pipeline_id)
    return jsonify({"status": "ok"})


def _random_select_dataset(ds, n: int, seed: int, requested_split: str = None):
    """Select N random rows using np.random with seed; restore RNG state after.
    Returns same type as input: DatasetDict or single Dataset.
    """
    import numpy as np
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        if hasattr(ds, "keys"):
            out = {}
            for split_name, d in ds.items():
                total = d.num_rows
                k = min(n, total)
                indices = np.random.choice(total, size=k, replace=False)
                out[split_name] = d.select(indices.tolist())
            import datasets
            return datasets.DatasetDict(out)
        else:
            split_name = (requested_split and str(requested_split).strip()) or "dataset"
            d = ds
            total = d.num_rows
            k = min(n, total)
            indices = np.random.choice(total, size=k, replace=False)
            return d.select(indices.tolist())
    finally:
        np.random.set_state(state)


def _dataset_to_info(ds, requested_split: str = None) -> dict:
    """Build serializable dataset_info: columns, num_rows per split, sample rows.
    Handles both DatasetDict (multiple splits) and single Dataset (when split= is used).
    """
    # When split= is passed, load_dataset returns a single Dataset, not DatasetDict
    if hasattr(ds, "keys"):
        # DatasetDict
        splits = list(ds.keys())
        datasets_by_split = {s: ds[s] for s in splits}
    else:
        # Single Dataset
        split_name = (requested_split and str(requested_split).strip()) or "dataset"
        splits = [split_name]
        datasets_by_split = {split_name: ds}

    num_rows = {s: d.num_rows for s, d in datasets_by_split.items()}
    columns = []
    sample_rows = {}
    sample_size = 50
    for split in splits:
        d = datasets_by_split[split]
        if not columns and hasattr(d, "column_names"):
            columns = list(d.column_names)
        try:
            n = min(sample_size, d.num_rows)
            if n > 0:
                batch = d.select(range(n))
                rows = []
                for i in range(n):
                    row = {col: _safe_value(batch[col][i]) for col in (batch.column_names or [])}
                    rows.append(row)
                sample_rows[split] = rows
        except Exception:
            sample_rows[split] = []
    return {
        "splits": splits,
        "num_rows": num_rows,
        "columns": columns,
        "sample_rows": sample_rows,
    }


def _safe_value(v):
    """Convert value to JSON-serializable form."""
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [_safe_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _safe_value(x) for k, x in v.items()}
    return str(v)


def _load_pipeline_dataset(pipeline: dict):
    """Load HF dataset from pipeline's path and options. Returns (ds, requested_split)."""
    import datasets
    path = (pipeline.get("hf_dataset_path") or "").strip()
    if not path:
        raise ValueError("Pipeline has no dataset path. Load a dataset first.")
    opts = pipeline.get("hf_load_options") or {}
    config_name = (opts.get("config") or "").strip() or None
    split = opts.get("split")
    random_n = opts.get("random_n")
    seed = opts.get("seed")
    load_kwargs = {"path": path}
    if config_name:
        load_kwargs["name"] = config_name
    if split:
        load_kwargs["split"] = split
    ds = datasets.load_dataset(**load_kwargs)
    if random_n is not None and random_n > 0 and seed is not None:
        ds = _random_select_dataset(ds, n=random_n, seed=seed, requested_split=split)
    return ds, split


def _get_process_function(code: str):
    """Execute user code and return a callable named 'process' (example -> dict)."""
    if not (code or code.strip()):
        return None
    namespace = {
        "json": __import__("json"),
        "re": __import__("re"),
    }
    exec(code.strip(), namespace)
    fn = namespace.get("process")
    if not callable(fn):
        raise ValueError("Code must define a function named 'process' that takes one argument (example dict) and returns a dict.")
    return fn


@app.post("/api/dataset-pipelines/<pipeline_id>/process")
def api_apply_processing(pipeline_id: str):
    """Apply Python map function to pipeline's dataset. Body: { "processing_code": "def process(example): ..." }."""
    pipeline = get_pipeline_by_id(pipeline_id)
    if not pipeline:
        return jsonify({"error": "Pipeline not found"}), 404
    data = request.get_json(force=True) or {}
    code = (data.get("processing_code") or data.get("code") or "").strip()
    if not code:
        return jsonify({"error": "processing_code is required"}), 400
    try:
        process_fn = _get_process_function(code)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    try:
        ds, requested_split = _load_pipeline_dataset(pipeline)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    try:
        if hasattr(ds, "keys"):
            ds = ds.map(process_fn, batched=False, remove_columns=None, desc="Processing")
        else:
            ds = ds.map(process_fn, batched=False, remove_columns=None, desc="Processing")
        processed_info = _dataset_to_info(ds, requested_split=requested_split)
        update_pipeline_store(
            pipeline_id,
            status="processed",
            processing_code=code,
            processed_dataset_info=processed_info,
        )
        return jsonify({
            "status": "ok",
            "processed_dataset_info": processed_info,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/dataset-pipelines/<pipeline_id>/load")
def api_load_hf_dataset(pipeline_id: str):
    """Load dataset from Hugging Face. Accepts path, config name, split (e.g. train, test)."""
    pipeline = get_pipeline_by_id(pipeline_id)
    if not pipeline:
        return jsonify({"error": "Pipeline not found"}), 404
    data = request.get_json(force=True) or {}
    path = (data.get("path") or data.get("hf_dataset_path") or "").strip()
    if not path:
        return jsonify({"error": "Dataset path required"}), 400
    config_name = (data.get("config") or data.get("config_name") or "").strip() or None
    split = data.get("split")
    if isinstance(split, str) and split.strip():
        split = split.strip()
    elif isinstance(split, list):
        split = "+".join(str(s).strip() for s in split if str(s).strip())
    else:
        split = None
    random_n = data.get("random_n") or data.get("n")
    if random_n is not None:
        try:
            random_n = int(random_n)
        except (TypeError, ValueError):
            random_n = None
    seed = data.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = None
    try:
        import datasets
    except ImportError:
        return jsonify({"error": "Install datasets: pip install datasets"}), 500
    try:
        load_kwargs = {"path": path}
        if config_name:
            load_kwargs["name"] = config_name
        if split:
            load_kwargs["split"] = split
        ds = datasets.load_dataset(**load_kwargs)
        if random_n is not None and random_n > 0 and seed is not None:
            ds = _random_select_dataset(ds, n=random_n, seed=seed, requested_split=split)
        dataset_info = _dataset_to_info(ds, requested_split=split)
        hf_load_options = {
            "config": config_name,
            "split": split,
            "random_n": random_n,
            "seed": seed,
        }
        update_pipeline_store(
            pipeline_id,
            hf_dataset_path=path,
            status="loaded",
            hf_load_options=hf_load_options,
            dataset_info=dataset_info,
        )
        return jsonify({
            "status": "ok",
            "path": path,
            "load_options": hf_load_options,
            "dataset_info": dataset_info,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _saved_var_name(path: str, split, random_n, seed, task_name: str, additional_naming: str = None) -> str:
    """Build variable name: dataName/split/randomN/seed/taskName[/additionalNaming]."""
    parts = [
        (path or "").strip() or "-",
        str(split).strip() if split is not None and str(split).strip() else "-",
        str(random_n) if random_n is not None else "-",
        str(seed) if seed is not None else "-",
        (task_name or "").strip() or "-",
    ]
    base = "/".join(parts)
    extra = (additional_naming or "").strip()
    if extra:
        return base + "/" + extra
    return base


def _estimate_data_var_memory_mb(data: dict) -> float:
    """Rough estimate of saved variable size in MB (dataset_info + processed)."""
    total_rows = 0
    for info in (data.get("dataset_info"), data.get("processed_dataset_info")):
        if not info:
            continue
        for n in (info.get("num_rows") or {}).values():
            total_rows += int(n) if n is not None else 0
    if total_rows == 0:
        return 0.0
    return round(total_rows * 0.0005, 2)  # ~500 bytes/row -> MB


@app.get("/api/data-vars")
def api_list_data_vars():
    """List loaded model (with GPU/RAM) and saved data variables. Name | Memory (GPU/RAM) | Delete."""
    loaded = SESSION_STATE.get("loaded_model")
    loaded_model = None
    if loaded:
        try:
            status = get_model_status(loaded)
            gpu_gb = 0.0
            ram_gb = 0.0
            for dev in (status.get("device_status") or []):
                d = dev.get("device", "")
                gb = float(dev.get("memory_gb") or 0)
                if d.startswith("cuda"):
                    gpu_gb += gb
                else:
                    ram_gb += gb
            loaded_model = {
                "name": loaded,
                "memory_gpu_gb": round(gpu_gb, 3),
                "memory_ram_gb": round(ram_gb, 3),
            }
        except Exception:
            loaded_model = {"name": loaded, "memory_gpu_gb": None, "memory_ram_gb": None}
    variables = []
    for name, data in SAVED_DATA_VARS.items():
        mem_mb = _estimate_data_var_memory_mb(data)
        variables.append({
            "name": name,
            "task_name": data.get("task_name"),
            "created_at": data.get("created_at"),
            "memory_ram_mb": mem_mb,
            "type": "data",
        })
    return jsonify({"loaded_model": loaded_model, "variables": variables})


@app.delete("/api/data-vars/<path:var_name>")
def api_delete_data_var(var_name: str):
    """Remove a saved data variable by name (URL-decoded)."""
    if var_name not in SAVED_DATA_VARS:
        return jsonify({"error": "Variable not found"}), 404
    del SAVED_DATA_VARS[var_name]
    return jsonify({"status": "ok"})


@app.post("/api/data-vars/save")
def api_save_data_var():
    """Save current pipeline state to global variable. Body: { pipeline_id, additional_naming? }. Variable name uses pipeline name; if additional_naming given, appends /additional_naming."""
    data = request.get_json(force=True) or {}
    pipeline_id = (data.get("pipeline_id") or "").strip()
    additional_naming = (data.get("additional_naming") or "").strip() or None
    if not pipeline_id:
        return jsonify({"error": "pipeline_id required"}), 400
    pipeline = get_pipeline_by_id(pipeline_id)
    if not pipeline:
        return jsonify({"error": "Pipeline not found"}), 404
    path = (pipeline.get("hf_dataset_path") or "").strip()
    if not path:
        return jsonify({"error": "Load a dataset first"}), 400
    task_name = (pipeline.get("name") or "").strip() or "-"
    opts = pipeline.get("hf_load_options") or {}
    split = opts.get("split")
    random_n = opts.get("random_n")
    seed = opts.get("seed")
    var_name = _saved_var_name(path, split, random_n, seed, task_name, additional_naming)
    from datetime import datetime
    SAVED_DATA_VARS[var_name] = {
        "variable_name": var_name,
        "pipeline_id": pipeline_id,
        "task_name": task_name,
        "data_name": path,
        "split": split,
        "random_n": random_n,
        "seed": seed,
        "dataset_info": pipeline.get("dataset_info"),
        "processed_dataset_info": pipeline.get("processed_dataset_info"),
        "hf_load_options": opts,
        "created_at": datetime.now().isoformat(),
    }
    return jsonify({"status": "ok", "variable_name": var_name})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
