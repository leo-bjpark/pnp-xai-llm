import json
from queue import Empty, Queue
from threading import Thread

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context

from python.memory import cache_store
from python.xai_handlers import (
    run_completion,
    run_conversation,
    run_placeholder,
    run_residual_concept,
)
from python.dataset_pipeline_store import get_pipeline_by_id
from python.web.dataset_utils import load_pipeline_dataset

run_bp = Blueprint("run", __name__)

@run_bp.post("/api/run")
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
        return (
            jsonify(
                {
                    "error": "session_mismatch",
                    "message": "Loaded Model + Treatment does not match the current session. Load the model with this setting?",
                    "requested": {"model": model, "treatment": treatment},
                    "current": {"model": current_model, "treatment": current_treatment},
                }
            ),
            400,
        )

    # Conversation (0.1.2): messages from JS (client-side cache)
    messages_input = input_setting.get("messages")
    system_instruction = (input_setting.get("system_instruction") or "").strip()

    if current_model and messages_input and isinstance(messages_input, list):
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

    if current_model and input_setting.get("adversarial_rows") is not None:
        from python.xai_handlers import run_adversarial_text_generation

        result, status = run_adversarial_text_generation(
            model=model,
            treatment=treatment,
            current_model=current_model,
            input_setting=input_setting,
        )
        return jsonify(result), status

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
        result, status = run_residual_concept(
            model=model,
            input_setting=input_setting,
            load_dataset_fn=load_pipeline_dataset,
            progress_callback=None,
        )
        return jsonify(result), status

    # Fallback: placeholder for other levels (xai_2+)
    result, status = run_placeholder(
        model=model,
        treatment=treatment,
        input_setting=input_setting,
    )
    return jsonify(result), status


@run_bp.post("/api/run/residual-concept-stream")
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

    queue: "Queue[dict]" = Queue()
    result_holder: dict = {}
    error_holder: dict = {}

    def progress_cb(batch, total):
        queue.put({"type": "progress", "batch": batch, "total": total, "message": f"Forward batch {batch}/{total}"})

    # Capture the real app object while we are still in request context
    app_obj = current_app._get_current_object()

    def run_thread():
        # Use a dedicated application context inside the worker thread
        with app_obj.app_context():
            try:
                res, status = run_residual_concept(
                    model=model,
                    input_setting=input_setting,
                    load_dataset_fn=load_pipeline_dataset,
                    progress_callback=progress_cb,
                )
                if status >= 400:
                    error_holder["error"] = res.get("error", "Unknown error")
                else:
                    result_holder["result"] = res
            except Exception as e:  # noqa: BLE001
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


@run_bp.post("/api/conversation/clear")
def api_conversation_clear():
    """No-op: conversation cache is managed by JS."""
    return jsonify({"status": "ok"})
