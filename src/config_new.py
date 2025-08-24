# Configuration file for YOLO Object Detection App - Requirement 1

import os

class Config:
    """Base configuration for YOLO Object Detection Web App"""
    
    # Application settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'yolo-webapp-secret-key'
    
    # Model configuration - YOLOv8s and custom trained model
    MODELS_DIR = "weights"
    DEFAULT_MODEL = "yolov8s"  # Default to YOLOv8s pre-trained model
    
    # Available models (Requirement 1 & 2)
    AVAILABLE_MODELS = {
        "yolov8s": {
            "name": "General Object Detector", 
            "file": "yolov8s.pt",
            "description": "Phát hiện đa dạng các đối tượng - từ con người, phương tiện giao thông đến động vật và đồ vật",
            "size": "21.5MB",
            "speed": "Fast",
            "type": "general",
            "requirement": 1,
            "classes": 80,
            "class_names": [
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
                "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
                "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
                "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                "hair drier", "toothbrush"
            ],
            "icon": "fas fa-globe",
            "color": "#667eea"
        },
        "wild_animal": {
            "name": "Wildlife Specialist Detector",
            "file": "best.pt", 
            "description": "Nhận diện chuyên nghiệp các loài động vật hoang dã - được huấn luyện riêng cho Voi, Hươu cao cổ, Báo đốm, Sư tử và Ngựa vằn",
            "size": "21.5MB",
            "speed": "Fast", 
            "type": "specialized",
            "requirement": 2,
            "classes": 5,
            "class_names": ["elephant", "giraffe", "leopard", "lion", "zebra"],
            "class_names_vi": ["Voi", "Hươu cao cổ", "Báo đốm", "Sư tử", "Ngựa vằn"],
            "icon": "fas fa-paw",
            "color": "#f093fb"
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
