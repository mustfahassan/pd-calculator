from flask import Flask
from .routes import pupil_bp
from .resource_manager import resource_manager

def create_app():
    app = Flask(__name__)
    
    # Register blueprints
    app.register_blueprint(pupil_bp)
    
    # Register cleanup
    @app.teardown_appcontext
    def cleanup(exception=None):
        resource_manager.cleanup_resources()
    
    return app