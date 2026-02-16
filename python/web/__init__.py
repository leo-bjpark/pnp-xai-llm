from pathlib import Path

from flask import Flask


def create_app() -> Flask:
    """
    Application factory.

    - Creates Flask app
    - Registers all blueprints
    """
    # Project root = three levels up from this file: python/web/__init__.py -> web -> python -> project root
    project_root = Path(__file__).resolve().parents[2]
    templates_dir = project_root / "templates"
    static_dir = project_root / "static"

    app = Flask(
        __name__,
        template_folder=str(templates_dir),
        static_folder=str(static_dir),
    )

    # Import blueprints locally to avoid circular imports
    from .views_main import main_bp
    from .api_tasks import tasks_bp
    from .api_session import session_bp
    from .api_model import model_bp
    from .api_run import run_bp
    from .api_memory import memory_bp
    from .api_dataset import dataset_bp
    from .api_residual import residual_bp
    from .api_brain import brain_bp
    from .api_theme import theme_bp
    from .api_analyzer import analyzer_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(tasks_bp)
    app.register_blueprint(session_bp)
    app.register_blueprint(model_bp)
    app.register_blueprint(run_bp)
    app.register_blueprint(memory_bp)
    app.register_blueprint(dataset_bp)
    app.register_blueprint(residual_bp)
    app.register_blueprint(brain_bp)
    app.register_blueprint(theme_bp)
    app.register_blueprint(analyzer_bp)

    return app
