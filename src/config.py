# Configuration file for YOLO Object Detection App - Requirement 1

import os

class Config:
    """Base configuration for YOLO Object Detection Web App"""
    
    # Application settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'yolo-webapp-secret-key'
    
    # Model configuration - Using YOLOv8 available models
    MODELS_DIR = "weights"
    DEFAULT_MODEL = "yolov8n"  # Default to fastest model
    
    # Available YOLOv8 models (Requirement 1: using available models)
    AVAILABLE_MODELS = {
        "yolov8n": {
            "name": "YOLOv8 Nano",
            "file": "yolov8n.pt",
            "description": "Fastest model - Good for real-time detection",
            "size": "6.2MB",
            "speed": "Very Fast"
        },
        "yolov8s": {
            "name": "YOLOv8 Small", 
            "file": "yolov8s.pt",
            "description": "Balanced speed and accuracy",
            "size": "21.5MB",
            "speed": "Fast"
        },
        "yolov8m": {
            "name": "YOLOv8 Medium",
            "file": "yolov8m.pt", 
            "description": "Higher accuracy - Better for detailed detection",
            "size": "49.7MB",
            "speed": "Medium"
        },
        "yolov8l": {
            "name": "YOLOv8 Large",
            "file": "yolov8l.pt",
            "description": "Highest accuracy - Best detection quality",
            "size": "83.7MB",
            "speed": "Slower"
        }
    }
    
    # File upload settings
    UPLOAD_FOLDER = "uploads"
    RESULTS_FOLDER = "results"
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    
    # Detection parameters
    DEFAULT_CONFIDENCE = 0.25
    DEFAULT_IOU = 0.45
    MIN_CONFIDENCE = 0.1
    MAX_CONFIDENCE = 1.0
    MIN_IOU = 0.1
    MAX_IOU = 1.0
    
    # Performance settings
    MAX_IMAGE_SIZE = (1280, 1280)  # Resize large images
    CLEANUP_INTERVAL = 3600  # Clean up old files every hour (seconds)
    MAX_FILES_AGE = 24 * 3600  # Delete files older than 24 hours

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # In production, you might want to use a database or cloud storage
    # UPLOAD_FOLDER = "/var/uploads"
    # RESULTS_FOLDER = "/var/results"

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    UPLOAD_FOLDER = "test_uploads"
    RESULTS_FOLDER = "test_results"

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
