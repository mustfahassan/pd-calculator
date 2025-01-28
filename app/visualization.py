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
    """Class to handle face alignment visualization"""
    
    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Define ideal face oval dimensions
        self.oval_width = int(frame_width * 0.4)
        self.oval_height = int(frame_height * 0.6)
        self.center_x = frame_width // 2
        self.center_y = frame_height // 2

    def draw_guide(self, frame, face_landmarks) -> Tuple[bool, str]:
        """Draw face alignment guide and check alignment"""
        # Draw ideal position oval
        cv2.ellipse(frame, 
                   (self.center_x, self.center_y),
                   (self.oval_width//2, self.oval_height//2),
                   0, 0, 360, (255, 255, 255), 2)

        if face_landmarks is None:
            return False, "No face detected. Please look at the camera."

        # Get face coordinates
        coords = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * self.frame_width)
            y = int(landmark.y * self.frame_height)
            coords.append([x, y])
        face_coords = np.array(coords)
        
        # Get face bounds
        face_left = np.min(face_coords[:, 0])
        face_right = np.max(face_coords[:, 0])
        face_top = np.min(face_coords[:, 1])
        face_bottom = np.max(face_coords[:, 1])
        
        # Check alignment
        face_center_x = (face_left + face_right) / 2
        face_center_y = (face_top + face_bottom) / 2
        
        x_offset = abs(face_center_x - self.center_x)
        y_offset = abs(face_center_y - self.center_y)
        
        # Define tolerance
        x_tolerance = self.frame_width * 0.05
        y_tolerance = self.frame_height * 0.05
        
        if x_offset > x_tolerance:
            return False, "Move your face left/right to center"
        elif y_offset > y_tolerance:
            return False, "Move your face up/down to center"
        
        # Check face rotation using eye landmarks
        left_eye = np.mean([face_coords[468:472]], axis=1)
        right_eye = np.mean([face_coords[473:477]], axis=1)
        eye_angle = np.degrees(np.arctan2(right_eye[0][1] - left_eye[0][1],
                                        right_eye[0][0] - left_eye[0][0]))
        
        if abs(eye_angle) > 5:
            return False, "Keep your head level"
            
        return True, "Perfect! Hold still..."

class WebcamHandler:
    """Class to handle webcam capture and visualization"""
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        """Initialize webcam capture"""
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.initialize_camera()
        
    def initialize_camera(self):
        """Initialize or reinitialize the camera"""
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.cap.isOpened():
            raise RuntimeError("Could not initialize camera")
        self.config = VisualizationConfig()
        self.guide = None

    def initialize_guide(self, frame):
        """Initialize face alignment guide"""
        if self.guide is None:
            height, width = frame.shape[:2]
            self.guide = FaceAlignmentGuide(width, height)

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
        """Release webcam resources"""
        self.cap.release()
        cv2.destroyAllWindows()