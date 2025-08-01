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
    
    def load_model(self, model_name="yolov8s"):
        """
        Load YOLO model once and reuse for all detections
        This follows the requirement: "model load only one time"
        
        Args:
            model_name (str): Model name (yolov8s, custom_trained)
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Check if same model is already loaded
            if self.is_model_loaded and self.current_model_name == model_name:
                logger.info(f"Model {model_name} already loaded, reusing...")
                return True
            
            logger.info(f"Loading YOLO model: {model_name}")
            
            # Load YOLOv8 model (will auto-download if not exists)
            model_file = f"{model_name}.pt"
            self.model = YOLO(model_file)
            
            # Store model info
            self.current_model_name = model_name
            self.class_names = list(self.model.names.values())
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
                    class_name = self.model.names[cls]
                    
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
        Switch to a different model (this will reload)
        Use sparingly as per requirement: only when model needs to be updated
        """
        if new_model_name == self.current_model_name:
            logger.info(f"Already using {new_model_name}, no reload needed")
            return True
        
        logger.info(f"Switching from {self.current_model_name} to {new_model_name}")
        return self.load_model(new_model_name)
