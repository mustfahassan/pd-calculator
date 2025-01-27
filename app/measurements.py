import mediapipe as mp
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class PupilMeasurement:
    """Data class to store pupil measurement results"""
    left_pupil: Tuple[float, float]
    right_pupil: Tuple[float, float]
    pd_pixels: float
    pd_mm: float
    confidence: float = 0.0

class PupilDetector:
    """Class to handle pupil detection and PD measurement using MediaPipe Face Mesh"""
    
    def __init__(self, static_image_mode=False, max_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """Initialize MediaPipe Face Mesh"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # MediaPipe landmark indices for iris
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        # Face width landmarks (temple to temple)
        self.FACE_WIDTH_LANDMARKS = [127, 356]  
        
        # Average human face width (temple to temple) in mm
        self.AVERAGE_FACE_WIDTH_MM = 145.0
        self.PD_CALIBRATION_FACTOR = 0.943  # Adjustment factor to match 66mm baseline
        
    def _calculate_iris_center(self, landmarks, iris_indices) -> Tuple[float, float]:
        """Calculate the center point of an iris using MediaPipe landmarks"""
        iris_points = []
        for idx in iris_indices:
            point = landmarks.landmark[idx]
            iris_points.append((point.x, point.y))
            
        # Calculate centroid of iris points
        x_coords, y_coords = zip(*iris_points)
        return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

    def _calculate_face_width(self, landmarks, frame_width: int) -> float:
        """Calculate face width in pixels using temple-to-temple distance"""
        left_temple = landmarks.landmark[self.FACE_WIDTH_LANDMARKS[0]]
        right_temple = landmarks.landmark[self.FACE_WIDTH_LANDMARKS[1]]
        
        # Convert normalized coordinates to pixels
        left_x = int(left_temple.x * frame_width)
        right_x = int(right_temple.x * frame_width)
        
        return abs(right_x - left_x)

    def _pixel_to_mm(self, pixel_distance: float, face_width_pixels: float) -> float:
        """Convert pixel distance to millimeters using face width as reference"""
        mm_per_pixel = self.AVERAGE_FACE_WIDTH_MM / face_width_pixels
        return pixel_distance * mm_per_pixel * self.PD_CALIBRATION_FACTOR

    def detect_pupils(self, frame, results) -> Optional[PupilMeasurement]:
        """Detect pupils and calculate PD in a single frame"""
        frame_height, frame_width = frame.shape[:2]
        
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0]
        
        # Calculate iris centers
        left_center = self._calculate_iris_center(landmarks, self.LEFT_IRIS)
        right_center = self._calculate_iris_center(landmarks, self.RIGHT_IRIS)
        
        # Convert normalized coordinates to pixel coordinates
        left_pixel = (int(left_center[0] * frame_width), int(left_center[1] * frame_height))
        right_pixel = (int(right_center[0] * frame_width), int(right_center[1] * frame_height))
        
        # Calculate PD in pixels
        pd_pixels = np.sqrt((right_pixel[0] - left_pixel[0])**2 + (right_pixel[1] - left_pixel[1])**2)
        
        # Calculate face width in pixels
        face_width_pixels = self._calculate_face_width(landmarks, frame_width)
        
        # Convert PD to millimeters using face width as reference
        pd_mm = self._pixel_to_mm(pd_pixels, face_width_pixels)
        
        # Calculate confidence based on face mesh confidence
        confidence = sum(landmark.visibility for landmark in landmarks.landmark) / len(landmarks.landmark)
        
        # Create measurement result
        measurement = PupilMeasurement(
            left_pupil=left_pixel,
            right_pupil=right_pixel,
            pd_pixels=pd_pixels,
            pd_mm=pd_mm,
            confidence=confidence
        )
        
        return measurement

    def release(self):
        """Release MediaPipe resources"""
        self.face_mesh.close()