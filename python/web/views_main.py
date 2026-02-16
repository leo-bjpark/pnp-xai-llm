from flask import Blueprint, redirect, render_template

from python.config_loader import (
    get_analyzer_list,
    get_dataset_groups,
    get_xai_level_names,
    get_xai_level_names_grouped,
)
from python.dataset_pipeline_store import get_pipeline_by_id, get_pipelines
from python.model_load import get_config_models, get_config_models_grouped
from python.session_store import get_task_by_id, get_tasks


main_bp = Blueprint("main", __name__)


def _dataset_category_map():
    groups = get_dataset_groups()
    mapping = {}
    for group, items in (groups or {}).items():
        for item in items:
            mapping[item] = group
    return mapping


@main_bp.get("/panel")
def panel():
    """Standalone right panel window (opens in separate window, independent of main page)."""
    return render_template("panel.html")


@main_bp.get("/")
def index():
    """Main IDE-like interface."""
    models = get_config_models()
    models_grouped = get_config_models_grouped()
    xai_level_names = get_xai_level_names()
    xai_level_grouped = get_xai_level_names_grouped()
    tasks = get_tasks(xai_level_names)
    dataset_pipelines = get_pipelines()
    analyzers = get_analyzer_list()
    return render_template(
        "index.html",
        models=models,
        models_grouped=models_grouped,
        tasks=tasks,
        xai_level_names=xai_level_names,
        xai_level_grouped=xai_level_grouped,
        dataset_pipelines=dataset_pipelines,
        dataset_categories=_dataset_category_map(),
        analyzers=analyzers,
    )


def _task_template(level_key: str) -> str:
    """Template name for task view by task name."""
    _NAME_TO_TEMPLATE = {
        "Completion": "xai_0/completion.html",
        "Conversation": "xai_0/conversation.html",
        "Response Attribution": "xai_1/response_attribution.html",
        "Positive & Negative Attribution": "xai_1/response_attribution.html",
        "Adversarial Text Generation": "xai_1/adversarial_text_generation.html",
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
    analyzers = get_analyzer_list()
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
        dataset_categories=_dataset_category_map(),
        analyzers=analyzers,
    )


@main_bp.get("/task/<task_id>")
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


@main_bp.get("/analyzer/layer_residual_pca")
def analyzer_layer_residual_pca():
    """Standalone analyzer view for Layer Residual PCA (no task object)."""
    models = get_config_models()
    models_grouped = get_config_models_grouped()
    xai_level_names = get_xai_level_names()
    xai_level_grouped = get_xai_level_names_grouped()
    tasks = get_tasks(xai_level_names)
    dataset_pipelines = get_pipelines()
    analyzers = get_analyzer_list()
    return render_template(
        "analyzer/layer_residual_pca.html",
        models=models,
        models_grouped=models_grouped,
        tasks=tasks,
        xai_level_names=xai_level_names,
        xai_level_grouped=xai_level_grouped,
        dataset_pipelines=dataset_pipelines,
        dataset_categories=_dataset_category_map(),
        analyzers=analyzers,
    )


@main_bp.get("/data")
def data_index():
    """Redirect to first pipeline or index if none."""
    pipelines = get_pipelines()
    if pipelines:
        return redirect(f"/data/{pipelines[0]['id']}", code=302)
    return redirect("/", code=302)


@main_bp.get("/data/<pipeline_id>")
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
        dataset_groups=get_dataset_groups(),
        dataset_categories=_dataset_category_map(),
    )
