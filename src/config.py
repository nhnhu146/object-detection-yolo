# Configuration file for YOLO Object Detection App - Requirement 1

import os

class Config:
    """Base configuration for YOLO Object Detection Web App"""
    
    # Application settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'yolo-webapp-secret-key'
    
    # Model configuration - YOLOv8s and custom trained model
    MODELS_DIR = "weights"
    DEFAULT_MODEL = "yolov8s"  # Default to YOLOv8s pre-trained model
    
    # Available models (Requirement 1: using available models)
    AVAILABLE_MODELS = {
        "yolov8s": {
            "name": "YOLOv8 Small", 
            "file": "yolov8s.pt",
            "description": "Pre-trained model - Balanced speed and accuracy",
            "size": "21.5MB",
            "speed": "Fast",
            "type": "pretrained"
        },
        "custom_trained": {
            "name": "Custom Trained Model",
            "file": "custom_model.pt",
            "description": "Custom trained model - Specialized for specific objects",
            "size": "TBD",
            "speed": "Fast",
            "type": "custom"
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
