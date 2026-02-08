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

from flask import Flask, jsonify, render_template, request

# Model list and loading via python.model_load (reads config.yaml, uses backup.utils internally)
from python.model_load import get_config_models, get_config_models_grouped, load_model, get_model_status

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

app = Flask(__name__)

# In-memory session: current Loaded Model + Treatment
# Used for memory-efficient management and run confirmation
SESSION_STATE = {
    "loaded_model": None,
    "treatment": None,
}

# Conversation cache for 0.1.2: conversation_id -> {"model_key": str, "messages": [{"role", "content"}]}
CONVERSATION_CACHE = {}


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
    return render_template(
        "index.html",
        models=models,
        models_grouped=models_grouped,
        tasks=tasks,
        xai_level_names=xai_level_names,
        xai_level_grouped=xai_level_grouped,
    )


@app.get("/task/<task_id>")
def task_view(task_id):
    """Generic task view - fetches task and renders by xai_level."""
    task = get_task_by_id(task_id)
    if not task:
        models = get_config_models()
        models_grouped = get_config_models_grouped()
        xai_level_names = get_xai_level_names()
        return render_template("index.html", models=models, models_grouped=models_grouped, tasks=get_tasks(xai_level_names), xai_level_names=xai_level_names, xai_level_grouped=get_xai_level_names_grouped(), error="Task not found")
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
        return render_template("index.html", models=models, models_grouped=models_grouped, tasks=get_tasks(xai_level_names), xai_level_names=xai_level_names, xai_level_grouped=get_xai_level_names_grouped(), error="Task not found")
    models = get_config_models()
    models_grouped = get_config_models_grouped()
    xai_level_names = get_xai_level_names()
    tasks = get_tasks(xai_level_names)
    template = _task_template(level_key)
    return render_template(template, task=task, models=models, models_grouped=models_grouped, tasks=tasks, xai_level_names=xai_level_names, xai_level_grouped=get_xai_level_names_grouped())


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
