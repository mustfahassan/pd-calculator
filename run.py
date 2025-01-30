from flask import Flask
import os

def create_app():
    # Create Flask app with absolute path for template folder
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'app', 'templates'))
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'app', 'static'))
    
    app = Flask(__name__,
                template_folder=template_dir,
                static_folder=static_dir)
    
    # Enable debug mode based on environment
    app.config['DEBUG'] = os.environ.get('FLASK_ENV') == 'development'
    
    # Print debug info
    print(f"Current working directory: {os.getcwd()}")
    print(f"Template folder: {app.template_folder}")
    print(f"Static folder: {app.static_folder}")
    
    # Register blueprint
    from app.routes import pupil_bp
    app.register_blueprint(pupil_bp)
    
    return app

# Create the app instance for Gunicorn
app = create_app()

if __name__ == '__main__':
    # Run the app only if executed directly (not through Gunicorn)
    app.run(host='0.0.0.0', debug=True, port=8000)