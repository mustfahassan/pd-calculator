from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np

@dataclass
class PupilMeasurement:
    """Data class to store pupil measurement results"""
    left_pupil: Tuple[float, float]
    right_pupil: Tuple[float, float]
    pd_pixels: float
    pd_mm: float
    confidence: float = 0.0

class PupilDetector:
    """Class to handle pupil detection and PD measurement"""
    
    def __init__(self):
        """Initialize measurement parameters"""
        # MediaPipe landmark indices for iris
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        # Face width landmarks (temple to temple)
        self.FACE_WIDTH_LANDMARKS = [127, 356]
        
        # Average human face width (temple to temple) in mm
        self.AVERAGE_FACE_WIDTH_MM = 145.0
        self.PD_CALIBRATION_FACTOR = 0.943  # Adjustment factor to match 66mm baseline

    def _calculate_iris_center(self, landmarks: Dict[str, Any], iris_indices: list) -> Tuple[float, float]:
        """Calculate the center point of an iris using landmarks"""
        iris_points = []
        for idx in iris_indices:
            point = landmarks[str(idx)]
            iris_points.append((point['x'], point['y']))
            
        # Calculate centroid of iris points
        x_coords, y_coords = zip(*iris_points)
        return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

    def _calculate_face_width(self, landmarks: Dict[str, Any]) -> float:
        """Calculate face width using temple-to-temple landmarks"""
        left_temple = landmarks[str(self.FACE_WIDTH_LANDMARKS[0])]
        right_temple = landmarks[str(self.FACE_WIDTH_LANDMARKS[1])]
        
        return abs(right_temple['x'] - left_temple['x'])

    def _pixel_to_mm(self, pixel_distance: float, face_width_pixels: float) -> float:
        """Convert pixel distance to millimeters using face width as reference"""
        mm_per_pixel = self.AVERAGE_FACE_WIDTH_MM / face_width_pixels
        return pixel_distance * mm_per_pixel * self.PD_CALIBRATION_FACTOR

    def calculate_measurement_quality(self, landmarks: Dict[str, Any], 
                                   left_center: Tuple[float, float], 
                                   right_center: Tuple[float, float]) -> float:
        """Calculate measurement quality based on multiple factors"""
        try:
            # Check face orientation
            nose_tip_x = landmarks['4']['x']  # Nose tip landmark
            face_orientation = abs(0.5 - nose_tip_x)
            
            # Check eye openness
            left_eye_top = float(landmarks['386']['y'])
            left_eye_bottom = float(landmarks['374']['y'])
            right_eye_top = float(landmarks['159']['y'])
            right_eye_bottom = float(landmarks['145']['y'])
            
            eye_openness = min(
                abs(left_eye_top - left_eye_bottom),
                abs(right_eye_top - right_eye_bottom)
            )
            
            # Check gaze direction
            gaze_center = abs(0.5 - (left_center[0] + right_center[0]) / 2)
            
            # Calculate component scores
            orientation_score = max(0, 1 - face_orientation * 4)
            eye_score = min(1, eye_openness * 10)
            gaze_score = max(0, 1 - gaze_center * 4)
            
            # Calculate weighted quality score
            quality = (orientation_score * 0.4 + 
                      eye_score * 0.3 + 
                      gaze_score * 0.3)
                      
            return round(min(1.0, quality), 2)
            
        except Exception as e:
            print(f"Error calculating quality: {str(e)}")
            return 0.0

    def process_landmarks(self, landmarks: Dict[str, Any]) -> Optional[PupilMeasurement]:
        """Process face landmarks and calculate PD measurement"""
        try:
            # Calculate iris centers
            left_center = self._calculate_iris_center(landmarks, self.LEFT_IRIS)
            right_center = self._calculate_iris_center(landmarks, self.RIGHT_IRIS)
            
            # Calculate PD in normalized coordinates
            pd_pixels = np.sqrt(
                (right_center[0] - left_center[0])**2 + 
                (right_center[1] - left_center[1])**2
            )
            
            # Calculate face width for scaling
            face_width_pixels = self._calculate_face_width(landmarks)
            
            # Convert to millimeters
            pd_mm = self._pixel_to_mm(pd_pixels, face_width_pixels)
            
            # Calculate quality score
            quality_score = self.calculate_measurement_quality(
                landmarks, left_center, right_center
            )
            
            # Create measurement result
            measurement = PupilMeasurement(
                left_pupil=left_center,
                right_pupil=right_center,
                pd_pixels=pd_pixels,
                pd_mm=pd_mm,
                confidence=quality_score
            )
            
            return measurement
            
        except Exception as e:
            print(f"Error processing landmarks: {str(e)}")
            return None