import platform
import time
import cv2
import numpy as np
from typing import Tuple, Optional, Dict
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
        
        # Define ideal face oval dimensions
        self.oval_width = int(frame_width * 0.4)
        self.oval_height = int(frame_height * 0.6)
        self.center_x = frame_width // 2
        self.center_y = frame_height // 2
        
        # Define stricter tolerances
        self.position_tolerance = 0.03  # 3% of frame dimension
        self.rotation_tolerance = 3.0   # degrees
        self.size_tolerance = 0.1       # 10% of ideal size
        
        # Store last valid position
        self.last_valid_position = None
        self.position_history = []
        self.history_size = 10

    def check_position_stability(self, face_center):
        """Check if face position is stable over time"""
        if len(self.position_history) >= self.history_size:
            self.position_history.pop(0)
        self.position_history.append(face_center)
        
        if len(self.position_history) < self.history_size:
            return False
            
        # Calculate position variance
        x_coords = [pos[0] for pos in self.position_history]
        y_coords = [pos[1] for pos in self.position_history]
        x_variance = max(x_coords) - min(x_coords)
        y_variance = max(y_coords) - min(y_coords)
        
        return (x_variance < self.frame_width * 0.02 and 
                y_variance < self.frame_height * 0.02)

    def draw_guide(self, frame, face_landmarks) -> Tuple[bool, str]:
        if face_landmarks is None:
            self.last_valid_position = None
            self.position_history = []
            return False, "No face detected. Please look at the camera."

        # Get face coordinates and measurements
        coords = np.array([[landmark.x * self.frame_width, landmark.y * self.frame_height] 
                          for landmark in face_landmarks.landmark])
        
        face_left = np.min(coords[:, 0])
        face_right = np.max(coords[:, 0])
        face_top = np.min(coords[:, 1])
        face_bottom = np.max(coords[:, 1])
        
        face_width = face_right - face_left
        face_height = face_bottom - face_top
        face_center = [(face_left + face_right) / 2, (face_top + face_bottom) / 2]

        # Draw ideal position oval
        cv2.ellipse(frame, 
                   (self.center_x, self.center_y),
                   (self.oval_width//2, self.oval_height//2),
                   0, 0, 360, (255, 255, 255), 2)

        # Check size
        ideal_width = self.oval_width
        width_diff = abs(face_width - ideal_width) / ideal_width
        if width_diff > self.size_tolerance:
            return False, "Move closer or further to fit the oval"

        # Check position
        x_offset = abs(face_center[0] - self.center_x) / self.frame_width
        y_offset = abs(face_center[1] - self.center_y) / self.frame_height
        
        if x_offset > self.position_tolerance:
            return False, "Move your face left/right to center"
        if y_offset > self.position_tolerance:
            return False, "Move your face up/down to center"

        # Check rotation using eye landmarks
        left_eye = np.mean(coords[468:472], axis=0)
        right_eye = np.mean(coords[473:477], axis=0)
        eye_angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1],
                                        right_eye[0] - left_eye[0]))
        
        if abs(eye_angle) > self.rotation_tolerance:
            return False, "Keep your head level"

        # Check position stability
        if not self.check_position_stability(face_center):
            return False, "Hold still..."

        self.last_valid_position = face_center
        return True, "Perfect! Maintaining position..."

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings"""
    text_color: Tuple[int, int, int] = (255, 255, 255)  # White text
    guide_color: Tuple[int, int, int] = (255, 255, 255)  # White guide
    error_color: Tuple[int, int, int] = (255, 255, 255)  # Keep all text white
    font_scale: float = 0.6
    thickness: int = 2

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
        """Initialize or reinitialize the camera with enhanced error handling"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        try:
            # List of camera indices to try
            camera_indices = [
                self.camera_id,      # User specified index
                0,                   # Default camera
                1                    # Alternative camera
            ]
            
            # Try each camera index
            for idx in camera_indices:
                if self._try_camera_init(idx):
                    # Configure camera settings
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    
                    # For macOS, add a small delay after initialization
                    if self.is_macos:
                        time.sleep(0.5)
                    
                    # Verify camera works
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.is_initialized = True
                        print(f"Successfully initialized camera with index {idx}")
                        return
                        
            # If we get here, no camera worked
            raise RuntimeError("Could not initialize camera with any available index")
            
        except Exception as e:
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