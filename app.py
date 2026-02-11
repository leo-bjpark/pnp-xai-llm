"""
PnP-XAI-LLM - VSCode-like XAI analysis tool.

Thin Flask entrypoint. All routes are defined in python/web via blueprints.
"""

import sys
from pathlib import Path

from flask import Flask

from python.web import create_app


# Ensure project root is on path for other modules
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


app: Flask = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)

"""
PnP-XAI-LLM - VSCode-like XAI analysis tool.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on path for backup.utils
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask

from python.web import create_app


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
    level_key = task.get("xai_level", "")
    return _render_task(task_id, level_key)




def _task_template(level_key: str) -> str:
    """Template name for task view by task name."""
    _NAME_TO_TEMPLATE = {
        "Completion": "xai_0/completion.html",
        "Conversation": "xai_0/conversation.html",
        "Response Attribution": "xai_1/response_attribution.html",
        "Positive & Negative Attribution": "xai_1/response_attribution.html",
        "Residual Concept Detection": "xai_2/residual_concept_detection.html",
        "Layer Direction Similarity Analysis": "xai_2/layer_direction_similarity.html",
        "Brain Concept Visualization": "xai_2/brain_concept.html",
    }
    return _NAME_TO_TEMPLATE.get(level_key, "xai_2/not_implemented.html")


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
    xai_level = data.get("xai_level", "")
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
    sess = cache_store.get_session()
    return jsonify({
        "loaded_model": sess["loaded_model"],
        "treatment": sess["treatment"],
    })


@app.post("/api/session")
def api_set_session():
    """Set session (after user confirms model load)."""
    data = request.get_json(force=True) or {}
    model = data.get("loaded_model")
    treatment = data.get("treatment", "")
    cache_store.set_session(model, treatment)
    return jsonify({"status": "ok"})


@app.post("/api/session/leave")
def api_session_leave():
    """Leave current task: clear only task-associated caches (session namespace). Keeps loaded_model and treatment."""
    cache_store.clear_namespace("session")
    return jsonify({"status": "ok"})


@app.post("/api/transformer_cache/clear")
def api_transformer_cache_clear():
    """Clear transformer cache (model + CUDA) while preserving treatment."""
    clear_model_cache()
    sess = cache_store.get_session()
    cache_store.set_session(None, sess.get("treatment") or "")
    return jsonify({"status": "ok"})


# ----- API: Models -----

@app.get("/api/models")
def api_models():
    return jsonify({"models": get_config_models()})


@app.get("/api/brain-config")
def api_brain_config():
    """Return brain concept map nodes from brain.yaml for Brain Concept Visualization (2.1.0)."""
    from python.brain_config import load_brain_nodes
    nodes = load_brain_nodes()
    return jsonify({"nodes": nodes})


@app.post("/api/load_model")
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
        return jsonify({"status": "ok", "model": model_key})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


def _detect_layer_structure(model):
    """
    Inspect model to detect layers_base, attn_name, mlp_name, o_proj_name, down_proj_name.
    Returns dict with detected values; if a name doesn't match default, value is None.
    Defaults: self_attn, mlp, o_proj, down_proj.
    """
    DEFAULT_ATTN = "self_attn"
    DEFAULT_MLP = "mlp"
    DEFAULT_O_PROJ = "o_proj"
    DEFAULT_DOWN_PROJ = "down_proj"

    result = {
        "layers_base": None,
        "attn_name": None,
        "mlp_name": None,
        "o_proj_name": None,
        "down_proj_name": None,
    }
    layer_names = []
    for name, _ in model.named_modules():
        if not name or "layers." not in name and ".layers." not in name:
            continue
        parts = name.split(".")
        if parts[-1].isdigit():
            layer_names.append(name)
    layer_names = sorted(set(layer_names), key=lambda x: (x.count("."), x))
    if not layer_names:
        return result
    first = layer_names[0]
    if "." in first:
        base, idx = first.rsplit(".", 1)
        if idx.isdigit():
            result["layers_base"] = base

    prefix = result["layers_base"] + ".0."
    attn_candidates = {DEFAULT_ATTN, "attention", "self_attention"}
    mlp_candidates = {DEFAULT_MLP}

    first_layer_children = set()
    for name, _ in model.named_modules():
        if not name or not name.startswith(prefix):
            continue
        rest = name[len(prefix) :]
        if "." in rest:
            first_layer_children.add(rest.split(".")[0])
        else:
            first_layer_children.add(rest)

    for c in first_layer_children:
        if c in attn_candidates:
            result["attn_name"] = c
            break
    if result["attn_name"] is None and first_layer_children:
        for c in first_layer_children:
            if "attn" in c.lower():
                result["attn_name"] = c
                break

    for c in first_layer_children:
        if c in mlp_candidates:
            result["mlp_name"] = c
            break
    if result["mlp_name"] is None and first_layer_children:
        for c in first_layer_children:
            if "mlp" in c.lower():
                result["mlp_name"] = c
                break

    attn_prefix = prefix + (result["attn_name"] or DEFAULT_ATTN) + "."
    mlp_prefix = prefix + (result["mlp_name"] or DEFAULT_MLP) + "."
    for name, _ in model.named_modules():
        if not name:
            continue
        if name.startswith(attn_prefix) and name.endswith("." + DEFAULT_O_PROJ):
            result["o_proj_name"] = DEFAULT_O_PROJ
            break
        if name == attn_prefix + DEFAULT_O_PROJ:
            result["o_proj_name"] = DEFAULT_O_PROJ
            break
    if result["o_proj_name"] is None:
        for name, _ in model.named_modules():
            if attn_prefix in name and "o_proj" in name:
                result["o_proj_name"] = "o_proj"
                break

    for name, _ in model.named_modules():
        if not name:
            continue
        if name.startswith(mlp_prefix) and name.endswith("." + DEFAULT_DOWN_PROJ):
            result["down_proj_name"] = DEFAULT_DOWN_PROJ
            break
        if name == mlp_prefix + DEFAULT_DOWN_PROJ:
            result["down_proj_name"] = DEFAULT_DOWN_PROJ
            break

    return result


def _empty_layer_structure():
    """Empty layer structure when detection fails."""
    return {
        "layers_base": None,
        "attn_name": None,
        "mlp_name": None,
        "o_proj_name": None,
        "down_proj_name": None,
    }


@app.get("/api/model_layer_names")
def api_model_layer_names():
    """Return layer structure for the given model (layers_base, attn, mlp, o_proj, down_proj)."""
    model_key = request.args.get("model", "").strip()
    if not model_key:
        return jsonify({"error": "model query param required"}), 400
    try:
        from python.model_load import load_llm
        tokenizer, model = load_llm(model_key)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    try:
        names = []
        for name, _ in model.named_modules():
            if not name:
                continue
            if "layers." in name or ".layers." in name:
                parts = name.split(".")
                if parts[-1].isdigit():
                    names.append(name)
        names = sorted(set(names), key=lambda x: (x.count("."), x))
        layers_base = ""
        if names:
            first = names[0]
            if "." in first:
                base, idx = first.rsplit(".", 1)
                if idx.isdigit():
                    layers_base = base
        structure = _detect_layer_structure(model)
        if layers_base and not structure.get("layers_base"):
            structure["layers_base"] = layers_base
        return jsonify({
            "layer_names": names,
            "layers_base": layers_base,
            "layer_structure": structure,
        })
    except Exception:
        return jsonify({
            "layer_names": [],
            "layers_base": "",
            "layer_structure": _empty_layer_structure(),
        })


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
    data["session"] = cache_store.get_session().copy()
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
        cache_store.set_session(sess.get("loaded_model"), sess.get("treatment") or "")
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
    sess = cache_store.get_session()
    current_model = sess.get("loaded_model")
    current_treatment = sess.get("treatment")

    if model != current_model or treatment != current_treatment:
        return jsonify({
            "error": "session_mismatch",
            "message": "Loaded Model + Treatment does not match the current session. Load the model with this setting?",
            "requested": {"model": model, "treatment": treatment},
            "current": {"model": current_model, "treatment": current_treatment},
        }), 400

    # Conversation (0.1.2): messages from JS (client-side cache)
    messages_input = input_setting.get("messages")
    system_instruction = (input_setting.get("system_instruction") or "").strip()

    if current_model and messages_input and isinstance(messages_input, list):
        from python.xai_handlers import run_conversation

        result, status = run_conversation(
            model=model,
            treatment=treatment,
            current_model=current_model,
            messages_input=messages_input,
            system_instruction=system_instruction,
            input_setting=input_setting,
        )
        return jsonify(result), status

    # Completion (0.1.1) or Response Attribution (1.0.1): input_string + generation params
    input_string = (input_setting.get("input_string") or "").strip()
    system_instruction = (input_setting.get("system_instruction") or "").strip()
    attribution_method = (input_setting.get("attribution_method") or "").strip()

    if "input_string" in input_setting and current_model:
        if attribution_method:
            from python.xai_handlers import run_attribution

            result, status = run_attribution(
                model=model,
                treatment=treatment,
                current_model=current_model,
                input_string=input_string,
                system_instruction=system_instruction,
                attribution_method=attribution_method,
                input_setting=input_setting,
            )
            return jsonify(result), status
        from python.xai_handlers import run_completion

        result, status = run_completion(
            model=model,
            treatment=treatment,
            current_model=current_model,
            input_string=input_string,
            input_setting=input_setting,
        )
        return jsonify(result), status

    # Residual Concept Detection (2.0.1)
    if (input_setting.get("variable_name") or "").strip() and current_model:
        from python.xai_handlers import run_residual_concept

        result, status = run_residual_concept(
            model=model,
            input_setting=input_setting,
            load_dataset_fn=_load_pipeline_dataset,
            progress_callback=None,
        )
        return jsonify(result), status

    # Fallback: placeholder for other levels (xai_2+)
    from python.xai_handlers import run_placeholder

    result, status = run_placeholder(
        model=model,
        treatment=treatment,
        input_setting=input_setting,
    )
    return jsonify(result), status


@app.post("/api/run/residual-concept-stream")
def api_run_residual_concept_stream():
    """
    Residual Concept Detection with SSE progress stream.
    Same body as /api/run. Streams: data: {"type":"progress","batch":n,"total":m}\n\n
    Final: data: {"type":"done","result":{...}}\n\n
    """
    data = request.get_json(force=True) or {}
    model = data.get("model", "")
    treatment = data.get("treatment", "")
    input_setting = data.get("input_setting", {})

    sess = cache_store.get_session()
    current_model = sess.get("loaded_model")
    current_treatment = sess.get("treatment")

    if model != current_model or treatment != current_treatment:
        def err_gen():
            yield f"data: {json.dumps({'type': 'error', 'error': 'session_mismatch'})}\n\n"
        return Response(
            stream_with_context(err_gen()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if not (input_setting.get("variable_name") or "").strip() or not current_model:
        def err_gen():
            yield f"data: {json.dumps({'type': 'error', 'error': 'variable_name required'})}\n\n"
        return Response(
            stream_with_context(err_gen()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    queue = Queue()
    result_holder = {}
    error_holder = {}

    def progress_cb(batch, total):
        queue.put({"type": "progress", "batch": batch, "total": total, "message": f"Forward batch {batch}/{total}"})

    def run_thread():
        with app.app_context():
            try:
                from python.xai_handlers import run_residual_concept

                res, status = run_residual_concept(
                    model=model,
                    input_setting=input_setting,
                    load_dataset_fn=_load_pipeline_dataset,
                    progress_callback=progress_cb,
                )
                if status >= 400:
                    error_holder["error"] = res.get("error", "Unknown error")
                else:
                    result_holder["result"] = res
            except Exception as e:
                error_holder["error"] = str(e)
        queue.put({"type": "done"})

    t = Thread(target=run_thread)
    t.start()

    def gen():
        while True:
            try:
                msg = queue.get(timeout=300)
            except Empty:
                break
            if msg["type"] == "done":
                break
            yield f"data: {json.dumps(msg)}\n\n"

        if error_holder:
            yield f"data: {json.dumps({'type': 'error', 'error': error_holder.get('error', 'Unknown')})}\n\n"
        else:
            result = result_holder.get("result", {})
            try:
                yield f"data: {json.dumps({'type': 'done', 'result': result})}\n\n"
            except (TypeError, ValueError) as e:
                yield f"data: {json.dumps({'type': 'error', 'error': f'Result serialize error: {e}'})}\n\n"

    return Response(
        stream_with_context(gen()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/conversation/clear")
def api_conversation_clear():
    """No-op: conversation cache is managed by JS."""
    return jsonify({"status": "ok"})


SESSION_CACHE_NAMESPACE = "session"


@app.get("/api/memory/summary")
def api_memory_summary():
    """Return Session | Result | Variable memory usage in GB."""
    session_gb = 0.0
    result_gb = 0.0
    variable_gb = 0.0

    sess = cache_store.get_session()
    loaded_model = sess.get("loaded_model")
    session_caches = cache_store.list_session_caches()
    if loaded_model and session_caches:
        try:
            status = get_model_status(loaded_model)
            for dev in (status.get("device_status") or []):
                gb = float(dev.get("memory_gb") or 0)
                session_gb += gb
        except Exception:
            pass

    if TASKS_FILE.exists():
        result_gb = TASKS_FILE.stat().st_size / (1024**3)

    var_summary = variable_store.summarize_for_panel(loaded_model_key=loaded_model)
    if var_summary.get("loaded_model"):
        lm = var_summary["loaded_model"]
        g = lm.get("memory_gpu_gb")
        r = lm.get("memory_ram_gb")
        if g is not None:
            variable_gb += float(g)
        if r is not None:
            variable_gb += float(r)
    for v in var_summary.get("variables", []):
        mb = v.get("memory_ram_mb")
        if mb is not None:
            variable_gb += float(mb) / 1024

    return jsonify({
        "session_gb": round(session_gb, 3),
        "result_gb": round(result_gb, 3),
        "variable_gb": round(variable_gb, 3),
    })


@app.get("/api/memory/session/list")
def api_list_session_caches():
    """List Session caches. Key = Task | model | treatment | 이름."""
    items = cache_store.list_session_caches()
    return jsonify({"caches": items})


@app.post("/api/memory/session/register")
def api_register_session_cache():
    """Register a cache entry. Key = Task | model | treatment | 이름."""
    data = request.get_json(force=True) or {}
    task_id = (data.get("task_id") or "").strip()
    model = (data.get("model") or "").strip()
    treatment = (data.get("treatment") or "").strip()
    name = (data.get("name") or "").strip()
    if not task_id:
        return jsonify({"error": "task_id required"}), 400
    key = f"{task_id}|{model}|{treatment}|{name}"
    cache_store.put(SESSION_CACHE_NAMESPACE, key, {"key_parts": [task_id, model, treatment, name]})
    return jsonify({"status": "ok", "key": key})


@app.delete("/api/memory/session/unregister/<path:key>")
def api_unregister_session_cache(key: str):
    """Unregister one cache entry."""
    cache_store.delete(SESSION_CACHE_NAMESPACE, key)
    return jsonify({"status": "ok"})


@app.post("/api/memory/session/clear")
def api_clear_session():
    """Clear all Session caches (loaded model, treatment, all Python caches)."""
    cache_store.terminate_all()
    clear_model_cache()
    return jsonify({"status": "ok"})


@app.post("/api/memory/result/clear")
def api_clear_result():
    """Clear all Task Results (tasks.json)."""
    xai_level_names = get_xai_level_names()
    task_result_store.clear_all(xai_level_names)
    return jsonify({"status": "ok"})


@app.post("/api/memory/variable/clear")
def api_clear_variable():
    """Clear all Variables (working memory)."""
    variable_store.clear_all()
    return jsonify({"status": "ok"})


@app.post("/api/empty_cache")
def api_empty_cache():
    """
    Reset: clear loaded model, session state, conversation cache,
    CUDA cache, and working‑memory variables.

    Called after user confirms Empty Cache (warning shown in UI).
    """
    cache_store.terminate_all()
    clear_model_cache()
    wm_clear_all()
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


@app.get("/api/data-vars")
def api_list_data_vars():
    """List loaded model (with GPU/RAM) and saved working‑memory variables."""
    loaded = cache_store.get_session().get("loaded_model")
    payload = summarize_for_panel(loaded_model_key=loaded)
    loaded_map = cache_store.get_namespace("variable_loaded")
    loaded_names = set(loaded_map.keys()) if isinstance(loaded_map, dict) else set()

    variables = []
    seen_names = set()
    for v in payload.get("variables", []):
        name = (v.get("name") or "").strip()
        if not name:
            continue
        has_ram = name in loaded_names
        has_pickle = variable_store.has_pickle(name)
        has_disk = has_pickle
        if has_ram and has_disk:
            status = "ram_and_disk"
            status_label = "RAM + Disk"
        elif has_ram and not has_disk:
            status = "ram_only"
            status_label = "RAM only"
        elif not has_ram and has_disk:
            status = "disk_only"
            status_label = "Disk only"
        else:
            status = "missing"
            status_label = "Missing"
        v.update(
            {
                "has_ram": has_ram,
                "has_disk": has_disk,
                "has_pickle": has_pickle,
                "status": status,
                "status_label": status_label,
            }
        )
        variables.append(v)
        seen_names.add(name)

    if isinstance(loaded_map, dict):
        for name, entry in loaded_map.items():
            if name in seen_names:
                continue
            payload_type = (entry or {}).get("type") if isinstance(entry, dict) else None
            variables.append(
                {
                    "name": name,
                    "type": payload_type or "data",
                    "created_at": (entry or {}).get("loaded_at", "") if isinstance(entry, dict) else "",
                    "memory_ram_mb": None,
                    "has_ram": True,
                    "has_disk": False,
                    "status": "ram_only",
                    "status_label": "RAM only",
                }
            )

    payload["variables"] = variables
    return jsonify(payload)


@app.delete("/api/data-vars/<path:var_name>")
def api_delete_data_var(var_name: str):
    """Remove a working‑memory variable by name (URL‑decoded)."""
    if not wm_delete_variable(var_name):
        return jsonify({"error": "Variable not found"}), 404
    variable_store.delete_pickle(var_name)
    if cache_store.get("variable_loaded", var_name) is not None:
        cache_store.delete("variable_loaded", var_name)
    return jsonify({"status": "ok"})


@app.post("/api/data-vars/save")
def api_save_data_var():
    """
    Save current pipeline state to global working memory.

    Requires: Load → Apply Processing → Save. Only processed data is saved.
    Body: { pipeline_id, additional_naming? }.
    Variable name uses dataset path / split / random_n / seed / task name;
    if additional_naming is given, it is appended at the end.
    """
    data = request.get_json(force=True) or {}
    pipeline_id = (data.get("pipeline_id") or "").strip()
    additional_naming = (data.get("additional_naming") or "").strip() or None
    if not pipeline_id:
        return jsonify({"error": "pipeline_id required"}), 400
    pipeline = get_pipeline_by_id(pipeline_id)
    if not pipeline:
        return jsonify({"error": "Pipeline not found"}), 404
    if pipeline.get("status") != "processed":
        return jsonify({
            "error": "Apply Processing first before Save To Variable. Load → Apply Processing → Save.",
        }), 400
    try:
        var_name = save_pipeline_variable(
            pipeline_id=pipeline_id,
            pipeline=pipeline,
            additional_naming=additional_naming,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    pickle_saved = False
    loaded = False
    try:
        from python.web.dataset_utils import load_pipeline_dataset, get_process_function

        ds, requested_split = load_pipeline_dataset(pipeline)
        code = (pipeline.get("processing_code") or "").strip()
        if pipeline.get("status") == "processed" and code:
            process_fn = get_process_function(code)
            ds = ds.map(process_fn, batched=False, remove_columns=None, desc="Processing")
        payload = {"type": "data", "dataset": ds, "split": requested_split}
        pickle_saved = variable_store.save_pickle(var_name, payload)
        if pickle_saved:
            cache_store.put(
                "variable_loaded",
                var_name,
                {
                    "type": "data",
                    "dataset": ds,
                    "requested_split": requested_split,
                    "loaded_at": datetime.now().isoformat(),
                    "object_name": type(ds).__name__,
                },
            )
            loaded = True
    except Exception:
        pickle_saved = False
    return jsonify({"status": "ok", "variable_name": var_name, "pickle_saved": pickle_saved, "loaded": loaded})


@app.get("/api/residual-vars/<path:var_name>")
def api_get_residual_var(var_name: str):
    """Get residual variable by name (directions dict)."""
    rv = get_residual_variable(var_name)
    if not rv:
        return jsonify({"error": "Variable not found"}), 404
    return jsonify(rv)


@app.post("/api/residual-vars/save")
def api_save_residual_var():
    """
    Save residual direction vectors to variable.
    Body: { directions, task_name, model, num_keys, model_dim, additional_naming? }.
    """
    data = request.get_json(force=True) or {}
    directions = data.get("directions")
    if not directions or not isinstance(directions, dict):
        return jsonify({"error": "directions (dict) required"}), 400
    task_name = (data.get("task_name") or "").strip() or "Residual"
    model = (data.get("model") or "").strip() or "-"
    num_keys = data.get("num_keys")
    model_dim = data.get("model_dim")
    if num_keys is None:
        num_keys = len(directions)
    if model_dim is None and directions:
        first = next(iter(directions.values()), [])
        model_dim = len(first) if isinstance(first, (list, tuple)) else 0
    additional_naming = (data.get("additional_naming") or "").strip() or None
    try:
        var_name = save_residual_variable(
            directions=directions,
            task_name=task_name,
            model=model,
            num_keys=int(num_keys),
            model_dim=int(model_dim or 0),
            additional_naming=additional_naming,
        )
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    payload = {
        "directions": directions,
        "task_name": task_name,
        "model": model,
        "num_keys": int(num_keys),
        "model_dim": int(model_dim or 0),
        "created_at": datetime.now().isoformat(),
    }
    pickle_saved = variable_store.save_pickle(var_name, payload)
    return jsonify({"status": "ok", "variable_name": var_name, "pickle_saved": pickle_saved})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
