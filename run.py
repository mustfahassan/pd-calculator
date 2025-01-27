from flask import Flask
import os
from app.routes import pupil_bp

def create_app():
    app = Flask(__name__,
                template_folder=os.path.join('app', 'templates'))
    
    # Enable debug mode
    app.config['DEBUG'] = True
    
    # Register blueprint
    app.register_blueprint(pupil_bp)
    
    # Print debug info
    print(f"Template folder: {app.template_folder}")
    print(f"Static folder: {app.static_folder}")
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)