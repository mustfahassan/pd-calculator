from flask import Flask
import os

def create_app():
    # Create absolute paths
    app_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(app_dir, 'templates')
    static_dir = os.path.join(app_dir, 'static')
    
    # Initialize Flask with absolute paths
    app = Flask(__name__, 
                template_folder=template_dir,
                static_folder=static_dir)
    
    # Print paths for debugging
    print(f"Application directory: {app_dir}")
    print(f"Template directory: {template_dir}")
    print(f"Static directory: {static_dir}")
    
    # Register blueprint
    from .routes import pupil_bp
    app.register_blueprint(pupil_bp)
    
    return app