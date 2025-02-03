import platform
import time
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings"""
    text_color: Tuple[int, int, int] = (255, 255, 255)  # White text
    guide_color: Tuple[int, int, int] = (255, 255, 255)  # White guide
    error_color: Tuple[int, int, int] = (255, 255, 255)  # Keep all text white
    font_scale: float = 0.6
    thickness: int = 2




class FaceAlignmentGuide:
    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Define face guide dimensions - made smaller and more oval
        self.face_width_ratio = 0.25  # Much narrower
        self.face_height_ratio = 0.55  # Taller relative to width
        
        # Calculate dimensions
        self.oval_width = int(frame_width * self.face_width_ratio)
        self.oval_height = int(frame_height * self.face_height_ratio)
        self.center_x = frame_width // 2
        self.center_y = frame_height // 2
        
        # Tolerances for easier fitting
        self.position_tolerance = 0.05
        self.size_tolerance = 0.15
        
        # Position tracking
        self.position_history = []
        self.history_size = 5
        
        # Pure white color and subtle transparency
        self.guide_color = (255, 255, 255)  # White
        self.overlay_alpha = 0.6
        
        # MediaPipe facial landmarks for PD calculation
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]

    def create_overlay(self, frame) -> np.ndarray:
        """Create a transparent overlay with face oval guide"""
        overlay = np.zeros_like(frame, dtype=np.uint8)
        
        # Draw main oval guide - clean white line
        cv2.ellipse(overlay,
                   (self.center_x, self.center_y),
                   (self.oval_width//2, self.oval_height//2),
                   0, 0, 360,
                   self.guide_color,
                   2)
        
        return overlay

    def check_position_stability(self, face_center: Tuple[float, float]) -> bool:
        """Check if face position is stable"""
        if len(self.position_history) >= self.history_size:
            self.position_history.pop(0)
        self.position_history.append(face_center)
        
        if len(self.position_history) < self.history_size:
            return False
            
        x_coords = [pos[0] for pos in self.position_history]
        y_coords = [pos[1] for pos in self.position_history]
        x_variance = max(x_coords) - min(x_coords)
        y_variance = max(y_coords) - min(y_coords)
        
        return (x_variance < self.frame_width * 0.03 and 
                y_variance < self.frame_height * 0.03)

    def draw_guide(self, frame, face_landmarks) -> Tuple[bool, str]:
        """Draw face guide and check position"""
        try:
            if face_landmarks is None:
                self.position_history = []
                return False, "No face detected"

            # Create and add overlay
            overlay = self.create_overlay(frame)
            cv2.addWeighted(overlay, self.overlay_alpha, frame, 1, 0, frame)
            
            # Get face coordinates
            coords = np.array([[landmark.x * self.frame_width, landmark.y * self.frame_height] 
                            for landmark in face_landmarks.landmark])
            
            # Calculate face measurements
            face_left = np.min(coords[:, 0])
            face_right = np.max(coords[:, 0])
            face_top = np.min(coords[:, 1])
            face_bottom = np.max(coords[:, 1])
            
            face_width = face_right - face_left
            face_height = face_bottom - face_top
            face_center = [(face_left + face_right) / 2, (face_top + face_bottom) / 2]

            # Position checks
            x_offset = abs(face_center[0] - self.center_x) / self.frame_width
            y_offset = abs(face_center[1] - self.center_y) / self.frame_height
            width_diff = abs(face_width - self.oval_width) / self.oval_width

            # Simplified feedback
            if width_diff > self.size_tolerance:
                if face_width < self.oval_width:
                    return False, "Move closer"
                return False, "Move back"
            
            if x_offset > self.position_tolerance or y_offset > self.position_tolerance:
                return False, "Center your face"

            if not self.check_position_stability(face_center):
                return False, "Hold still"

            return True, "Perfect"
            
        except Exception as e:
            print(f"Error in draw_guide: {str(e)}")
            return False, "Error in alignment"




class WebcamHandler:
    """Class to handle webcam capture and visualization"""
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        """Initialize webcam handler without starting camera"""
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.config = VisualizationConfig()
        self.guide = None
        self.is_initialized = False
        self.is_macos = platform.system() == 'Darwin'
        self.init_retries = 3  # Number of retries for initialization
        
    def _try_camera_init(self, camera_index: int) -> bool:
        """Try to initialize camera with a specific index"""
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                return False
                
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                return False
                
            self.cap = cap
            return True
        except Exception as e:
            print(f"Failed to initialize camera with index {camera_index}: {e}")
            return False
            
    def initialize_camera(self):
        """Initialize camera with macOS/Docker support"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        try:
            # Try different capture methods for macOS
            capture_methods = [
                (0, cv2.CAP_AVFOUNDATION),  # macOS AVFoundation
                (1, cv2.CAP_AVFOUNDATION),  # Alternative camera index
                "avfoundation://",          # GStreamer AVFoundation
                (0, cv2.CAP_ANY),           # Any available method
                "autovideosrc ! video/x-raw,width=640,height=480 ! videoconvert ! appsink",  # GStreamer auto
            ]
            
            for method in capture_methods:
                try:
                    if isinstance(method, tuple):
                        self.cap = cv2.VideoCapture(method[0], method[1])
                    else:
                        self.cap = cv2.VideoCapture(method)
                        
                    if self.cap is not None and self.cap.isOpened():
                        # Configure camera
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        
                        # Test read
                        ret, frame = self.cap.read()
                        if ret and frame is not None:
                            print(f"Successfully initialized camera with method: {method}")
                            self.is_initialized = True
                            return
                        else:
                            self.cap.release()
                            
                except Exception as e:
                    print(f"Failed to initialize with method {method}: {e}")
                    if self.cap:
                        self.cap.release()
                        self.cap = None
                        
            raise RuntimeError("Could not initialize camera with any available method")
            
        except Exception as e:
            print(f"Camera initialization error: {str(e)}")
            self.is_initialized = False
            if self.cap:
                self.cap.release()
                self.cap = None
            raise RuntimeError(f"Camera initialization failed: {str(e)}")

    def ensure_camera_initialized(self):
        """Ensure camera is initialized when needed"""
        if not self.is_initialized or self.cap is None or not self.cap.isOpened():
            for attempt in range(self.init_retries):
                try:
                    print(f"Camera initialization attempt {attempt + 1}/{self.init_retries}")
                    self.initialize_camera()
                    if self.is_initialized:
                        return
                except Exception as e:
                    print(f"Initialization attempt {attempt + 1} failed: {e}")
                    time.sleep(0.5)  # Wait before retrying
            raise RuntimeError("Failed to initialize camera after multiple attempts")

    def initialize_guide(self, frame):
        """Initialize face alignment guide"""
        if self.guide is None:
            height, width = frame.shape[:2]
            self.guide = FaceAlignmentGuide(width, height)

    def get_blank_frame(self, message: str = "Initializing camera..."):
        """Generate a blank frame with message"""
        blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (self.width - text_size[0]) // 2
        text_y = (self.height + text_size[1]) // 2
        cv2.putText(blank, message, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return blank

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera with error handling"""
        try:
            self.ensure_camera_initialized()
            if not self.cap or not self.cap.isOpened():
                return False, self.get_blank_frame("Camera not available")
                
            for attempt in range(2):  # Try twice to read frame
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    return True, frame
                time.sleep(0.1)  # Short delay between attempts
                
            return False, self.get_blank_frame("Failed to read from camera")
            
        except Exception as e:
            print(f"Error reading frame: {str(e)}")
            return False, self.get_blank_frame(f"Camera error: {str(e)}")

    def draw_measurement(self, frame, measurement):
        """Draw PD measurement on frame"""
        if not measurement:
            return frame
            
        # Draw pupils
        cv2.circle(frame, measurement.left_pupil, 3, self.config.text_color, -1)
        cv2.circle(frame, measurement.right_pupil, 3, self.config.text_color, -1)
        
        # Draw line between pupils
        cv2.line(frame, measurement.left_pupil, measurement.right_pupil, 
                self.config.text_color, self.config.thickness)
        
        # Add measurement text
        pd_text = f"PD: {measurement.pd_mm:.1f}mm"
        text_y = min(measurement.left_pupil[1], measurement.right_pupil[1]) - 20
        text_x = (measurement.left_pupil[0] + measurement.right_pupil[0]) // 2
        
        cv2.putText(frame, pd_text, (text_x - 60, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, 
                   self.config.text_color, self.config.thickness)
        
        return frame

    def add_instruction_text(self, frame, text: str, is_error: bool = False):
        """Add instruction text to frame"""
        height = frame.shape[0]
        
        # Add semi-transparent black background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, height-60), (frame.shape[1], height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Center text
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        
        # Add white text with slight outline for better visibility
        cv2.putText(frame, text, (text_x, height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)  # Black outline
        cv2.putText(frame, text, (text_x, height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # White text
        
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Webcam resources released.")

    def __del__(self):
        self.release()