from flask import Blueprint, render_template, Response, jsonify
import cv2
import time
import numpy as np
from .measurements import PupilDetector, PupilMeasurement
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
camera_started = False
stable_frame_count = 0
last_measurement = None
alignment_status = {"aligned": False, "message": "Look at the camera"}
STABILITY_THRESHOLD = 45
MEASUREMENT_FRAMES = 10
measurement_buffer = []

def process_frame(frame):
    """Process a single frame with pupil detection and visualization"""
    global last_measurement, alignment_status, is_measuring, stable_frame_count, measurement_buffer
    
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
            
            if aligned:
                stable_frame_count += 1
                if stable_frame_count >= STABILITY_THRESHOLD and not is_measuring:
                    is_measuring = True
                    measurement_buffer = []
            else:
                stable_frame_count = 0
            
            # Get measurement if we're measuring
            if is_measuring:
                measurement = detector.detect_pupils(frame, results)
                if measurement:
                    # Calculate quality score using the existing function
                    quality = detector.calculate_measurement_quality(
                    results.multi_face_landmarks[0],
                    detector._calculate_iris_center(results.multi_face_landmarks[0], detector.LEFT_IRIS),
                    detector._calculate_iris_center(results.multi_face_landmarks[0], detector.RIGHT_IRIS)
                )
                    measurement.confidence = quality  # Update measurement quality
                    measurement_buffer.append(measurement)
                    
                    if len(measurement_buffer) >= MEASUREMENT_FRAMES:
                        # Calculate average PD
                        avg_pd_mm = sum(m.pd_mm for m in measurement_buffer) / len(measurement_buffer)
                        avg_quality = sum(m.confidence for m in measurement_buffer) / len(measurement_buffer)
                        
                        last_measurement = PupilMeasurement(
                            left_pupil=measurement.left_pupil,
                            right_pupil=measurement.right_pupil,
                            pd_pixels=measurement.pd_pixels,
                            pd_mm=avg_pd_mm,
                            confidence=avg_quality
                        )
                        is_measuring = False
                    
                    frame = webcam_handler.draw_measurement(frame, measurement)
        else:
            alignment_status = {"aligned": False, "message": "No face detected. Please look at the camera."}
            stable_frame_count = 0
        
        # Add instruction text
        frame = webcam_handler.add_instruction_text(
            frame, 
            f"Stability: {stable_frame_count}/{STABILITY_THRESHOLD}" if stable_frame_count > 0 else alignment_status["message"],
            not alignment_status["aligned"]
        )
        
        return frame
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return frame

def generate_frames():
    """Generator function for streaming video frames"""
    global is_running, camera_started
    
    print("Starting frame generation")
    is_running = True
    while is_running and camera_started:
        print("Attempting to read frame")
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
    global camera_started
    if not camera_started:
        # Return a blank frame if camera isn't started
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', blank_frame)
        frame_bytes = buffer.tobytes()
        return Response(b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n',
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@pupil_bp.route('/start_camera', methods=['POST'])
def start_camera():
    """Start the camera feed"""
    global camera_started, is_measuring, stable_frame_count
    try:
        webcam_handler.initialize_camera()
        camera_started = True
        is_measuring = False
        stable_frame_count = 0
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@pupil_bp.route('/get_status')
def get_status():
    """Get current measurement status"""
    return jsonify({
        'stable_frames': stable_frame_count,
        'threshold': STABILITY_THRESHOLD,
        'is_measuring': is_measuring,
        'aligned': alignment_status["aligned"],
        'message': alignment_status["message"]
    })

@pupil_bp.route('/get_measurement')
def get_measurement():
    """Get the latest measurement result"""
    if last_measurement:
        quality_percentage = int(last_measurement.confidence * 100)  # Convert to percentage
        print(f"Quality Score: {quality_percentage}%")  # Debug print
        
        return jsonify({
            'pd_mm': round(last_measurement.pd_mm, 1),
            'confidence': quality_percentage,
            'aligned': alignment_status["aligned"],
            'message': alignment_status["message"]
        })
    return jsonify({'status': 'no_measurement'})

@pupil_bp.route('/stop', methods=['POST'])
def stop():
    """Stop video streaming"""
    global is_running, camera_started
    is_running = False
    camera_started = False
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