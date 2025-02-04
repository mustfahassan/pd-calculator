from flask import Flask
from .routes import pupil_bp

def create_app():
    app = Flask(__name__)
    
    # Register blueprints
    app.register_blueprint(pupil_bp)
    
    return app