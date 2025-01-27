from flask import Blueprint, render_template, Response, jsonify
import cv2
import time
from .measurements import PupilDetector
from .visualization import WebcamHandler
import threading
import queue

# Create blueprint
pupil_bp = Blueprint('pupil', __name__)

# Global objects
detector = PupilDetector()
webcam_handler = WebcamHandler()
is_running = False
is_measuring = False
last_measurement = None
alignment_status = {"aligned": False, "message": "Look at the camera"}

def process_frame(frame):
    """Process a single frame with pupil detection and visualization"""
    global last_measurement, alignment_status, is_measuring
    
    try:
        # Initialize face guide if needed
        webcam_handler.initialize_guide(frame)
        
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.face_mesh.process(frame_rgb)
        
        # Process frame if face is detected
        if results.multi_face_landmarks:
            # Check face alignment
            aligned, message = webcam_handler.guide.draw_guide(frame, results.multi_face_landmarks[0])
            alignment_status = {"aligned": aligned, "message": message}
            
            # Get measurement if face is aligned
            if aligned or is_measuring:
                measurement = detector.detect_pupils(frame, results)
                if measurement:
                    if is_measuring:
                        last_measurement = measurement
                        is_measuring = False
                    frame = webcam_handler.draw_measurement(frame, measurement)
        else:
            alignment_status = {"aligned": False, "message": "No face detected. Please look at the camera."}
        
        # Add instruction text
        frame = webcam_handler.add_instruction_text(
            frame, 
            alignment_status["message"],
            not alignment_status["aligned"]
        )
        
        return frame
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return frame

def generate_frames():
    """Generator function for streaming video frames"""
    global is_running
    
    is_running = True
    while is_running:
        success, frame = webcam_handler.cap.read()
        if not success:
            break
            
        # Process frame
        processed_frame = process_frame(frame)
            
        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue
            
        # Convert to bytes
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@pupil_bp.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@pupil_bp.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@pupil_bp.route('/start_measurement', methods=['POST'])
def start_measurement():
    """Start PD measurement process"""
    global is_measuring
    is_measuring = True
    return jsonify({'status': 'success'})

@pupil_bp.route('/get_measurement')
def get_measurement():
    """Get the latest measurement result"""
    if last_measurement:
        return jsonify({
            'pd_mm': round(last_measurement.pd_mm, 1),
            'confidence': last_measurement.confidence,
            'aligned': alignment_status["aligned"],
            'message': alignment_status["message"]
        })
    return jsonify({'status': 'no_measurement'})

@pupil_bp.route('/stop', methods=['POST'])
def stop():
    """Stop video streaming"""
    global is_running
    is_running = False
    webcam_handler.release()
    return jsonify({'status': 'success'})

# Error handlers
@pupil_bp.errorhandler(Exception)
def handle_error(error):
    """Handle any server errors"""
    return jsonify({
        'error': str(error),
        'status': 'error'
    }), 500