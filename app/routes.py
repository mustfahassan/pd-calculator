from flask import Blueprint, render_template, jsonify, request
from .measurements import PupilDetector, PupilMeasurement

# Create blueprint
pupil_bp = Blueprint('pupil', __name__)

# Initialize detector
detector = PupilDetector()

@pupil_bp.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@pupil_bp.route('/calculate_pd', methods=['POST'])
def calculate_pd():
    """Calculate PD from landmarks sent by frontend"""
    try:
        data = request.get_json()
        landmarks = data.get('landmarks')
        
        if not landmarks:
            return jsonify({
                'status': 'error',
                'message': 'No landmarks provided'
            }), 400
            
        # Process landmarks and calculate measurement
        measurement = detector.process_landmarks(landmarks)
        
        if measurement:
            return jsonify({
                'status': 'success',
                'pd_mm': round(measurement.pd_mm, 1),
                'confidence': round(measurement.confidence * 100, 1),
                'message': 'Measurement complete'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to calculate PD'
            }), 400
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@pupil_bp.errorhandler(Exception)
def handle_error(error):
    """Handle any server errors"""
    print(f"Server error: {str(error)}")
    return jsonify({
        'status': 'error',
        'message': str(error)
    }), 500