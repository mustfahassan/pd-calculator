class ResourceManager:
    def __init__(self):
        self.detector = None
        self.webcam_handler = None
        self.is_measuring = False
        self.stable_frame_count = 0
        self.last_measurement = None
        self.alignment_status = {"aligned": False, "message": "Look at the camera"}
        self.measurement_buffer = []

        # Constants
        self.STABILITY_THRESHOLD = 75
        self.MEASUREMENT_FRAMES = 20
        self.MIN_QUALITY_THRESHOLD = 0.85

    def init_resources(self):
        """Initialize detector and webcam handler if not already initialized"""
        from .measurements import PupilDetector
        from .visualization import WebcamHandler
        
        try:
            if self.detector is None:
                print("Initializing PupilDetector...")
                self.detector = PupilDetector()
            
            if self.webcam_handler is None:
                print("Initializing WebcamHandler...")
                self.webcam_handler = WebcamHandler()
            
            self.reset_state()
            print("Resources initialized successfully")
            
        except Exception as e:
            print(f"Error initializing resources: {str(e)}")
            self.cleanup_resources()
            raise

    def reset_state(self):
        """Reset measurement state"""
        self.is_measuring = False
        self.stable_frame_count = 0
        self.measurement_buffer = []
        self.last_measurement = None
        self.alignment_status = {"aligned": False, "message": "Look at the camera"}

    def cleanup_resources(self):
        """Clean up all resources"""
        try:
            print("Cleaning up resources...")
            
            if hasattr(self, 'detector') and self.detector:
                if hasattr(self.detector, 'face_mesh'):
                    self.detector.face_mesh.close()
                self.detector = None
            
            if hasattr(self, 'webcam_handler') and self.webcam_handler:
                self.webcam_handler = None
            
            self.reset_state()
            print("Resource cleanup completed")
            
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            raise

# Create global instance
resource_manager = ResourceManager()