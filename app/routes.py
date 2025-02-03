from flask import Blueprint, render_template, Response, jsonify
import cv2
import time
import numpy as np
from .measurements import PupilDetector, PupilMeasurement
from .visualization import WebcamHandler
from .resource_manager import ResourceManager

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
STABILITY_THRESHOLD = 30
MEASUREMENT_FRAMES = 10
measurement_buffer = []

def reset_state():
    """Reset all global state variables"""
    global is_running, is_measuring, camera_started, stable_frame_count, last_measurement
    is_running = False
    is_measuring = False
    camera_started = False
    stable_frame_count = 0
    last_measurement = None
    measurement_buffer.clear()

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
                    quality = detector.calculate_measurement_quality(
                        results.multi_face_landmarks[0],
                        detector._calculate_iris_center(results.multi_face_landmarks[0], detector.LEFT_IRIS),
                        detector._calculate_iris_center(results.multi_face_landmarks[0], detector.RIGHT_IRIS)
                    )
                    measurement.confidence = quality
                    measurement_buffer.append(measurement)
                    
                    if len(measurement_buffer) >= MEASUREMENT_FRAMES:
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
        try:
            success, frame = webcam_handler.read_frame()
            if not success:
                print("Failed to read frame")
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
        except Exception as e:
            print(f"Error in generate_frames: {str(e)}")
            break
    
    print("Frame generation stopped")

@pupil_bp.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@pupil_bp.route('/video_feed')
def video_feed():
    """Video streaming route"""
    global camera_started
    if not camera_started:
        blank_frame = webcam_handler.get_blank_frame("Click Start Measurement to begin")
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
        if webcam_handler.cap is not None:
            webcam_handler.release()
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
        quality_percentage = int(last_measurement.confidence * 100)
        print(f"Quality Score: {quality_percentage}%")
        
        return jsonify({
            'pd_mm': round(last_measurement.pd_mm, 1),
            'confidence': quality_percentage,
            'aligned': alignment_status["aligned"],
            'message': alignment_status["message"]
        })
    return jsonify({'status': 'no_measurement'})

@pupil_bp.route('/stop', methods=['POST'])
def stop():
    """Stop video streaming and cleanup resources"""
    global is_running, camera_started
    try:
        is_running = False
        camera_started = False
        if webcam_handler.cap is not None:
            webcam_handler.release()
        reset_state()
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error in stop route: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        # Reset state even if error occurs
        reset_state()

# Error handlers
@pupil_bp.errorhandler(Exception)
def handle_error(error):
    """Handle any server errors"""
    print(f"Server error: {str(error)}")
    return jsonify({
        'error': str(error),
        'status': 'error'
    }), 500