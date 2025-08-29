"""
YOLO Detection Module for Requirement 1
Handles YOLO model loading and inference with single model loading rule
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLODetector:
    """
    YOLO object detection class - Requirement 1 Implementation
    
    Features:
    - Uses YOLOv8s pre-trained model and custom trained model
    - Model loads only once and reused for all detections
    - Web interface compatible
    """
    
    def __init__(self):
        self.model = None
        self.current_model_name = ""
        self.class_names = []
        self.is_model_loaded = False
        self.last_fallback_info = None
    
    def load_model(self, model_name="yolov8s"):
        """
        Load YOLO model once and reuse for all detections
        This follows the requirement: "model load only one time"
        
        Args:
            model_name (str): Model name (yolov8s, wild_animal)
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Check if same model is already loaded
            if self.is_model_loaded and self.current_model_name == model_name:
                logger.info(f"Model {model_name} already loaded, reusing...")
                return True
            
            logger.info(f"Loading YOLO model: {model_name}")
            
            # Determine model file path
            if model_name == "wild_animal":
                # For custom trained model - search comprehensively
                model_file = self._find_best_model_file()
                
                if not model_file:
                    logger.warning("🔍 COMPREHENSIVE SEARCH FAILED: Custom wildlife model not found!")
                    logger.warning("🦎 Expected model: best.pt (trained on 5 wild animal classes)")
                    logger.warning("📊 Classes expected: Elephant, Giraffe, Leopard, Lion, Zebra")
                    logger.warning("⚠️ FALLBACK ACTIVATED: Using YOLOv8s general model (80 classes)")
                    logger.warning("🎯 IMPACT: Wildlife detection may be less accurate")
                    logger.warning("💡 SOLUTION: Ensure best.pt exists in project directory")
                    
                    # Store comprehensive fallback information for user notification
                    import datetime
                    self.last_fallback_info = {
                        'requested_model': 'wild_animal',
                        'requested_file': 'best.pt',
                        'fallback_model': 'yolov8s',
                        'fallback_file': 'yolov8s.pt',
                        'reason': 'Custom trained model file (best.pt) not found in project',
                        'searched_paths': self._get_search_paths(),
                        'impact': {
                            'en': 'Wildlife animals may be misclassified (e.g., leopard detected as dog)',
                            'vi': 'Động vật hoang dã có thể bị phân loại sai (VD: báo đốm nhận dạng thành chó)'
                        },
                        'solution': {
                            'en': 'Place the trained best.pt model file in the project root directory',
                            'vi': 'Đặt file mô hình best.pt đã được huấn luyện vào thư mục gốc của dự án'
                        },
                        'expected_classes': ['Elephant', 'Giraffe', 'Leopard', 'Lion', 'Zebra'],
                        'fallback_classes_count': 80,
                        'timestamp': datetime.datetime.now().isoformat(),
                        'severity': 'WARNING'
                    }
                    
                    model_file = "yolov8s.pt"
                    model_name = "yolov8s"  # Fallback to general model
                else:
                    # Clear any previous fallback info when model loads successfully
                    self.last_fallback_info = None
                    logger.info(f"✅ Custom wildlife model loaded from: {model_file}")
            else:
                # For pre-trained models
                model_file = f"{model_name}.pt"
            
            # Load YOLOv8 model (will auto-download if not exists)
            self.model = YOLO(model_file)
            
            # Store model info and validate class names
            self.current_model_name = model_name
            
            # Safe class names extraction with validation
            try:
                if hasattr(self.model, 'names') and self.model.names:
                    self.class_names = list(self.model.names.values())
                else:
                    logger.warning(f"Model {model_name} has no class names, using empty list")
                    self.class_names = []
            except Exception as e:
                logger.warning(f"Error extracting class names from {model_name}: {e}")
                self.class_names = []
            
            self.is_model_loaded = True
            
            logger.info(f"✅ Model loaded successfully: {model_name}")
            logger.info(f"✅ Classes available: {len(self.class_names)}")
            logger.info("🔄 Model will be reused for all future detections")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model {model_name}: {e}")
            self.model = None
            self.current_model_name = ""
            self.class_names = []
            self.is_model_loaded = False
            return False
    
    
    def predict(self, image_path, confidence=0.25, iou=0.45):
        """
        Run object detection on image using the loaded model
        Model is reused for all predictions (Requirement 1 compliance)
        
        Args:
            image_path (str): Path to input image
            confidence (float): Confidence threshold (0.0-1.0)
            iou (float): IoU threshold for NMS (0.0-1.0)
            
        Returns:
            dict: Detection results with predictions and annotated image
        """
        if not self.is_model_loaded or not self.model:
            logger.error("❌ No model loaded! Please load a model first.")
            return None
        
        try:
            logger.info(f"🔍 Running detection on: {os.path.basename(image_path)}")
            logger.info(f"📊 Using model: {self.current_model_name} (already loaded)")
            
            # Run inference using the loaded model
            results = self.model(
                image_path,
                conf=confidence,
                iou=iou,
                verbose=False
            )
            
            # Process results
            predictions = []
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    # Get box coordinates (xyxy format)
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    
                    # Get confidence and class
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Safe class name retrieval with fallback
                    try:
                        class_name = self.model.names[cls] if cls in self.model.names else f"class_{cls}"
                    except Exception:
                        class_name = f"unknown_class_{cls}"
                    
                    # Calculate additional info
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    prediction = {
                        'id': i,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'center': [float(center_x), float(center_y)],
                        'width': float(width),
                        'height': float(height),
                        'area': float(area),
                        'confidence': float(conf),
                        'class': class_name,
                        'class_id': int(cls)
                    }
                    predictions.append(prediction)
            
            # Create annotated image with bounding boxes
            annotated_img = result.plot(
                conf=True,
                labels=True,
                boxes=True,
                line_width=2
            )
            
            # Calculate detection statistics
            stats = self._calculate_stats(predictions)
            
            logger.info(f"✅ Detection completed: {len(predictions)} objects found")
            
            return {
                'success': True,
                'predictions': predictions,
                'annotated_image': annotated_img,
                'stats': stats,
                'model_info': {
                    'name': self.current_model_name,
                    'classes': len(self.class_names),
                    'reused': True  # Indicates model was reused
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _calculate_stats(self, predictions):
        """Calculate detection statistics"""
        if not predictions:
            return {
                'total_objects': 0,
                'classes_detected': [],
                'avg_confidence': 0.0,
                'max_confidence': 0.0,
                'min_confidence': 0.0,
                'class_counts': {}
            }
        
        confidences = [pred['confidence'] for pred in predictions]
        classes = [pred['class'] for pred in predictions]
        unique_classes = list(set(classes))
        
        # Count objects per class
        class_counts = {}
        for class_name in unique_classes:
            class_counts[class_name] = classes.count(class_name)
        
        return {
            'total_objects': len(predictions),
            'classes_detected': unique_classes,
            'avg_confidence': float(np.mean(confidences)),
            'max_confidence': float(np.max(confidences)),
            'min_confidence': float(np.min(confidences)),
            'class_counts': class_counts
        }
    
    def get_model_info(self):
        """Get information about the currently loaded model"""
        if not self.is_model_loaded:
            return {
                'loaded': False,
                'message': 'No model loaded'
            }
        
        return {
            'name': self.current_model_name,
            'classes': self.class_names,
            'num_classes': len(self.class_names),
            'loaded': True,
            'reused': True,  # Model is reused for all detections
            'file': f"{self.current_model_name}.pt"
        }
    
    def is_loaded(self):
        """Check if a model is currently loaded"""
        return self.is_model_loaded and self.model is not None
    
    def get_available_classes(self):
        """Get list of classes that the model can detect"""
        if not self.is_model_loaded:
            return []
        return self.class_names.copy()
    
    def switch_model(self, new_model_name):
        """
        Switch to a different model
        Use sparingly as per requirement: only when model needs to be updated
        
        Args:
            new_model_name (str): Name of the new model to load
        Returns:
            bool: True if successful, False otherwise
        """
        if new_model_name == self.current_model_name:
            logger.info(f"Already using model: {new_model_name}")
            return True
        
        logger.info(f"Switching from {self.current_model_name} to {new_model_name}")
        return self.load_model(new_model_name)
    
    def _find_best_model_file(self):
        """
        Comprehensive search for best.pt file in the entire project
        
        Returns:
            str or None: Path to best.pt if found, None otherwise
        """
        search_paths = self._get_search_paths()
        
        logger.info(f"Searching for custom wildlife model in {len(search_paths)} locations...")
        
        # Try each path and log the search process
        for i, path in enumerate(search_paths, 1):
            logger.info(f"[{i}/{len(search_paths)}] Checking: {path}")
            
            if os.path.exists(path):
                # Verify it's a valid model file
                try:
                    file_size = os.path.getsize(path)
                    if file_size > 1000:  # At least 1KB for a valid model
                        logger.info(f"✅ Found valid custom model: {path} ({file_size:,} bytes)")
                        return os.path.abspath(path)  # Return absolute path
                    else:
                        logger.warning(f"⚠️ Found {path} but file too small ({file_size} bytes)")
                except Exception as e:
                    logger.warning(f"⚠️ Found {path} but error reading: {e}")
        
        # Enhanced logging when model not found
        logger.warning("❌ Custom wildlife model 'best.pt' not found in any location!")
        logger.warning("📋 Searched locations:")
        for path in search_paths[:10]:  # Show first 10 paths
            logger.warning(f"   - {path}")
        if len(search_paths) > 10:
            logger.warning(f"   ... and {len(search_paths) - 10} more locations")
        
        return None
    
    def _get_search_paths(self):
        """
        Enhanced comprehensive search for best.pt file in entire project
        
        Returns:
            list: List of paths to search in priority order
        """
        # Get current working directory and project root
        current_dir = os.getcwd()
        project_root = current_dir
        
        # Try to find project root by looking for key files
        potential_roots = [current_dir]
        for parent in [os.path.dirname(current_dir), os.path.dirname(os.path.dirname(current_dir))]:
            if os.path.exists(os.path.join(parent, 'requirements.txt')) or os.path.exists(os.path.join(parent, 'src')):
                potential_roots.append(parent)
                project_root = parent
                break
        
        search_paths = []
        
        # Add all potential root directories first (highest priority)
        for root in potential_roots:
            search_paths.extend([
                os.path.join(root, "best.pt"),
                os.path.join(root, "weights", "best.pt"),
                os.path.join(root, "src", "weights", "best.pt"),
                os.path.join(root, "models", "best.pt"),
                os.path.join(root, "src", "models", "best.pt"),
                os.path.join(root, "src", "best.pt"),
            ])
        
        # Comprehensive recursive search in project directories
        search_directories = [
            project_root,
            os.path.join(project_root, "src"),
            os.path.join(project_root, "weights"),
            os.path.join(project_root, "models"),
            os.path.join(project_root, "archive"),
            current_dir
        ]
        
        for search_dir in search_directories:
            if os.path.exists(search_dir):
                # Recursive search for best.pt
                for root, dirs, files in os.walk(search_dir):
                    # Skip hidden directories and __pycache__
                    dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                    
                    if "best.pt" in files:
                        best_pt_path = os.path.join(root, "best.pt")
                        if best_pt_path not in search_paths:
                            search_paths.append(best_pt_path)
                    
                    # Also look for alternative names
                    for alt_name in ["wild_animal_detector.pt", "custom_model.pt", "trained_model.pt"]:
                        if alt_name in files:
                            alt_path = os.path.join(root, alt_name)
                            if alt_path not in search_paths:
                                search_paths.append(alt_path)
        
        # Remove duplicates while preserving order
        unique_paths = []
        for path in search_paths:
            if path not in unique_paths:
                unique_paths.append(path)
        
        return unique_paths
    
    def get_fallback_info(self):
        """Get information about current fallback status"""
        if self.last_fallback_info:
            return self.last_fallback_info
        else:
            return {
                'is_fallback': False,
                'current_model': self.current_model_name,
                'model_type': 'custom' if 'best' in self.current_model_name else 'pretrained'
            }
