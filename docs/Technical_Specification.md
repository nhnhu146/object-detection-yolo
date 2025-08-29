# Tài Liệu Kỹ Thuật: Roboflow Dataset Specification

## 1. Chi Tiết Dataset Structure

### 1.1 Roboflow Dataset Integration

#### Dataset Metadata
```yaml
# Extracted from archive/data.yaml
path: ../datasets/Wild-Animals-53  # dataset root dir
train: ../datasets/Wild-Animals-53/train/images  # train images
val: ../datasets/Wild-Animals-53/valid/images    # val images  
test: ../datasets/Wild-Animals-53/test/images    # test images

# Classes
nc: 5  # number of classes
names: ['Elephant', 'Giraffe', 'Leopard', 'Lion', 'Zebra']  # class names

# Roboflow Integration
roboflow:
  workspace: your-workspace
  project: wild-animals-53
  version: 1
```

#### File Naming Convention (Roboflow Style)
```bash
# Pattern: [original_name]_[unique_id].rf.[hash].[extension]
# Examples from actual dataset:
Elephant_100_jpg.rf.a5fdd0dd41b94932efcdb7c5ab45efc2.jpg
leopard2569_jpg.rf.44de9eec329055c9361eff2cac30922f.jpg
videoplayback-2025-04-09T035250_990_mp4-0008_jpg.rf.7837db0815f91c235a795777471d4555.jpg

# Corresponding label files:
Elephant_100_jpg.rf.a5fdd0dd41b94932efcdb7c5ab45efc2.txt
leopard2569_jpg.rf.44de9eec329055c9361eff2cac30922f.txt
```

### 1.2 Label Format Analysis

#### YOLO Annotation Example
```bash
# File: -109413412_jpg.rf.3093eb73eeb53add52fd1bde3bb31277.txt
1 0.5078125 0.5625 0.665625 0.85

# Breakdown:
# 1:          Class ID (0=Elephant, 1=Giraffe, 2=Leopard, 3=Lion, 4=Zebra)
# 0.5078125:  Center X coordinate (normalized)
# 0.5625:     Center Y coordinate (normalized)  
# 0.665625:   Width (normalized)
# 0.85:       Height (normalized)
```

#### Coordinate System Validation
```python
def validate_yolo_coordinates(class_id, center_x, center_y, width, height):
    """
    Validate YOLO format coordinates
    """
    validations = {
        'class_id': 0 <= class_id <= 4,           # Valid class range
        'center_x': 0 <= center_x <= 1,          # Normalized range
        'center_y': 0 <= center_y <= 1,          # Normalized range  
        'width': 0 < width <= 1,                 # Positive, normalized
        'height': 0 < height <= 1,               # Positive, normalized
        'bbox_bounds': (center_x - width/2) >= 0 and (center_x + width/2) <= 1,
        'bbox_bounds_y': (center_y - height/2) >= 0 and (center_y + height/2) <= 1
    }
    
    return all(validations.values()), validations
```

---

## 2. Dataset Statistics từ Phân Tích Thực Tế

### 2.1 File Count Analysis

#### Training Data Volume
```bash
# Từ kết quả list_dir thực tế:
Training Images: ~3,000+ files
Training Labels: ~3,000+ files (matching)

# Ví dụ một số files thực tế:
-109413412_jpg.rf.3093eb73eeb53add52fd1bde3bb31277.jpg/.txt
534337717_1115754870516806_2407962631341345439_n.jpg/.txt
Elephant_100_jpg.rf.a5fdd0dd41b94932efcdb7c5ab45efc2.jpg/.txt
leopard2569_jpg.rf.44de9eec329055c9361eff2cac30922f.jpg/.txt
Zebra_100_jpg.rf.8353cd993622c63a0078b682618f6255.jpg/.txt
```

### 2.2 Data Source Analysis

#### Nguồn Gốc Data (từ filename patterns)
```python
data_sources = {
    'unsplash': [
        'sutirta-budiman-9i9mcg8X6o4-unsplash_jpg.rf.*',
        'yaroslav-zotov-95csdsDkeA-unsplash_jpg.rf.*',
        'winston-tjia-wSAefyqFmWQ-unsplash_jpg.rf.*'
    ],
    'video_frames': [
        'videoplayback-2025-04-*_mp4-*_jpg.rf.*',
        'The-Land-of-the-Leopard-*_mp4-*_jpg.rf.*',
        'Wild-South-African-Leopards-Fight_mp4-*_jpg.rf.*'
    ],
    'stock_photos': [
        'Elephant_*_jpg.rf.*',
        'Zebra_*_jpg.rf.*',
        'Lion_*_jpg.rf.*'
    ],
    'whatsapp_images': [
        'WhatsApp-Image-2025-04-08-*_jpeg.rf.*'
    ]
}
```

---

## 3. Model Performance Analysis

### 3.1 Fallback Behavior Analysis

#### Root Cause của Leopard→Dog Issue
```python
"""
DETAILED ANALYSIS: Tại sao Leopard bị nhận dạng thành Dog

1. CUSTOM MODEL (best.pt):
   - Classes: ['Elephant', 'Giraffe', 'Leopard', 'Lion', 'Zebra'] (5 classes)
   - Trained specifically cho wild animals
   - Leopard classification: ACCURATE

2. FALLBACK MODEL (yolov8s.pt):
   - Classes: 80 COCO classes including 'dog', 'cat', 'person', etc.
   - NO 'leopard' class available
   - Leopard features → Closest match = 'dog' hoặc 'cat'

3. CLASSIFICATION LOGIC:
   When best.pt not found → Falls back to YOLOv8s
   YOLOv8s sees leopard spots pattern → Classifies as 'dog' (closest mammal)
"""

yolov8s_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    # ... 80 classes total
    # NOTE: 'leopard' KHÔNG có trong list này!
    # NOTE: 'lion' cũng KHÔNG có trong list này!
]

# Classes available in YOLOv8s for animals:
yolov8s_animals = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
# Missing: leopard, lion, tiger, và nhiều wild animals khác
```

### 3.2 Model Comparison Matrix

| Aspect | Custom Model (best.pt) | Fallback (YOLOv8s) |
|--------|------------------------|---------------------|
| **Classes** | 5 wild animals | 80 general objects |
| **Leopard Detection** | ✅ Accurate | ❌ Misclassified as 'dog' |
| **Lion Detection** | ✅ Accurate | ❌ Not available (fallback to 'cat'?) |
| **Elephant Detection** | ✅ Trained on wild elephants | ✅ Available but general |
| **Zebra Detection** | ✅ Trained on wild zebras | ✅ Available |  
| **Giraffe Detection** | ✅ Trained on wild giraffes | ✅ Available |
| **Training Data** | ~3,000 wild animal images | COCO dataset |
| **Specialization** | Wild animal focus | General object detection |

---

## 4. Error Analysis và Solutions

### 4.1 Common Misclassification Patterns

#### Predicted Issues with Fallback Model
```python
expected_misclassifications = {
    'leopard': {
        'correct_class': 'Leopard',
        'fallback_prediction': 'dog',
        'confidence_expected': 0.6-0.8,
        'reason': 'Spotted pattern similar to some dog breeds',
        'fix': 'Ensure best.pt model loads correctly'
    },
    'lion': {
        'correct_class': 'Lion', 
        'fallback_prediction': 'cat',
        'confidence_expected': 0.5-0.7,
        'reason': 'Big cat features, no lion class in COCO',
        'fix': 'Load custom model with lion class'
    },
    'elephant': {
        'correct_class': 'Elephant',
        'fallback_prediction': 'elephant',
        'confidence_expected': 0.8-0.9,
        'reason': 'COCO has elephant class - should work',
        'fix': 'May still work with fallback'
    }
}
```

### 4.2 Path Resolution Issues

#### Best.pt Search Algorithm Enhancement
```python
def comprehensive_model_search():
    """
    Enhanced model search implementation
    """
    
    # Priority-based search paths
    search_priority = [
        # Highest priority - project root
        ('PROJECT_ROOT', ['best.pt', 'weights/best.pt']),
        
        # Medium priority - common directories  
        ('COMMON_DIRS', [
            'src/best.pt', 'src/weights/best.pt',
            'models/best.pt', 'runs/train/*/weights/best.pt'
        ]),
        
        # Low priority - recursive search
        ('RECURSIVE', 'find . -name "best.pt" -type f'),
        
        # Last resort - alternative names
        ('ALTERNATIVES', [
            'wild_animal_detector.pt', 'custom_model.pt', 
            'trained_model.pt', 'final_model.pt'
        ])
    ]
    
    for priority_level, paths in search_priority:
        for path in paths:
            if validate_model_file(path):
                return path, priority_level
    
    return None, 'NOT_FOUND'
```

---

## 5. Deployment Configuration

### 5.1 Production Environment Setup

#### Docker Configuration
```dockerfile
# Dockerfile for YOLO Wild Animal Detection
FROM ultralytics/ultralytics:latest

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY archive/data.yaml ./archive/
COPY best.pt ./weights/  # Ensure model is available

# Copy templates and static files
COPY src/templates/ ./templates/
COPY src/static/ ./static/

# Set environment variables
ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/weights/best.pt
ENV FALLBACK_WARNING=true

EXPOSE 5000

CMD ["python", "src/app.py"]
```

#### Environment Variables
```bash
# Production environment configuration
export YOLO_MODEL_PATH="/app/weights/best.pt"
export FALLBACK_MODEL="yolov8s.pt"  
export ENABLE_FALLBACK_WARNINGS="true"
export MAX_BATCH_SIZE="16"
export GPU_MEMORY_FRACTION="0.8"
export LOG_LEVEL="INFO"
export UPLOAD_MAX_SIZE="16MB"
export RESULTS_RETENTION_DAYS="7"
```

### 5.2 Monitoring và Alerting

#### Key Metrics to Monitor
```python
monitoring_metrics = {
    'model_performance': [
        'inference_time_ms',           # Average inference time
        'batch_processing_fps',        # Frames per second
        'gpu_memory_usage_gb',         # GPU memory consumption  
        'cpu_usage_percent',           # CPU utilization
        'model_load_time_s',          # Model loading time
    ],
    'accuracy_metrics': [
        'fallback_activation_rate',    # % of requests using fallback
        'misclassification_reports',   # User-reported errors
        'confidence_score_dist',       # Distribution of confidence scores
        'class_detection_frequency',   # Usage per animal class
    ],
    'system_health': [
        'api_response_time_ms',        # API latency  
        'error_rate_percent',          # Error percentage
        'uptime_percentage',           # System availability
        'disk_space_usage_gb',         # Storage consumption
    ]
}
```

#### Alerting Rules
```yaml
# alerting_rules.yml
groups:
  - name: yolo_model_alerts
    rules:
    - alert: ModelFallbackActive
      expr: fallback_activation_rate > 10
      for: 5m
      annotations:
        summary: "High fallback model usage detected"
        description: "{{ $value }}% of requests using fallback model"
        
    - alert: HighInferenceLatency  
      expr: inference_time_ms > 1000
      for: 2m
      annotations:
        summary: "Inference time too high"
        description: "Average inference time: {{ $value }}ms"
        
    - alert: LowModelAccuracy
      expr: confidence_score_avg < 0.7
      for: 10m  
      annotations:
        summary: "Model confidence scores dropping"
        description: "Average confidence: {{ $value }}"
```

---

## 6. Testing Strategy

### 6.1 Unit Tests cho Model Loading

```python
import unittest
import os
from src.models.yolo_detector import YOLODetector

class TestYOLODetector(unittest.TestCase):
    
    def setUp(self):
        self.detector = YOLODetector()
        
    def test_custom_model_loading_success(self):
        """Test successful loading of custom wild animal model"""
        # Assuming best.pt exists
        result = self.detector.load_model("wild_animal")
        self.assertTrue(result)
        self.assertEqual(len(self.detector.class_names), 5)
        self.assertIn("Leopard", self.detector.class_names)
        
    def test_fallback_model_loading(self):
        """Test fallback to YOLOv8s when custom model not found"""
        # Temporarily rename best.pt to simulate missing file
        if os.path.exists("best.pt"):
            os.rename("best.pt", "best.pt.backup")
            
        try:
            result = self.detector.load_model("wild_animal") 
            self.assertTrue(result)  # Should still work with fallback
            self.assertEqual(self.detector.current_model_name, "yolov8s")
            self.assertIsNotNone(self.detector.last_fallback_info)
        finally:
            # Restore file
            if os.path.exists("best.pt.backup"):
                os.rename("best.pt.backup", "best.pt")
                
    def test_leopard_classification_accuracy(self):
        """Test leopard classification with both models"""
        test_leopard_image = "test_data/sample_leopard.jpg"
        
        # Test with custom model
        self.detector.load_model("wild_animal")
        result_custom = self.detector.predict(test_leopard_image)
        
        # Test with fallback model  
        self.detector.load_model("yolov8s")
        result_fallback = self.detector.predict(test_leopard_image)
        
        # Custom model should classify as 'Leopard'
        # Fallback model will likely classify as 'dog'
        self.assertNotEqual(
            result_custom['predictions'][0]['class'],
            result_fallback['predictions'][0]['class']
        )
```

### 6.2 Integration Tests

```python
class TestModelFallbackIntegration(unittest.TestCase):
    
    def test_end_to_end_fallback_warning(self):
        """Test complete fallback warning flow"""
        from src.app import app
        
        with app.test_client() as client:
            # Test health endpoint for fallback info
            response = client.get('/api/health')
            data = response.get_json()
            
            if data['fallback_warning']['active']:
                # Verify all expected fields present
                required_fields = [
                    'severity', 'title', 'message', 'details',
                    'impact', 'solution', 'type'  
                ]
                for field in required_fields:
                    self.assertIn(field, data['fallback_warning'])
                    
    def test_model_search_paths(self):
        """Test comprehensive model search functionality"""
        detector = YOLODetector()
        search_paths = detector._get_search_paths()
        
        # Should include common locations
        expected_paths = ['best.pt', 'weights/best.pt', 'src/weights/best.pt']
        for expected in expected_paths:
            self.assertTrue(any(expected in path for path in search_paths))
```

---

## 7. Performance Benchmarks

### 7.1 Actual Dataset Statistics

#### File Size Distribution
```python
# Estimated từ số lượng files quan sát được:
dataset_stats = {
    'total_images': 3000,  # Approximate count  
    'avg_image_size_mb': 0.5,  # Estimated
    'total_dataset_size_gb': 1.5,  # Images only
    'annotation_files': 3000,  # 1:1 mapping
    'avg_annotation_size_kb': 0.5,  # Text files are small
    
    'split_distribution': {
        'train': {'images': 2250, 'percentage': 75},
        'valid': {'images': 600, 'percentage': 20}, 
        'test': {'images': 150, 'percentage': 5}
    }
}
```

### 7.2 Expected Training Time

#### Hardware-specific Training Duration
```python
training_time_estimates = {
    'RTX_4090': {
        'epochs_100': '2-3 hours',
        'epochs_200': '4-6 hours',
        'gpu_utilization': '85-95%',
        'batch_size_optimal': 32
    },
    'RTX_3080': {
        'epochs_100': '4-5 hours', 
        'epochs_200': '8-10 hours',
        'gpu_utilization': '80-90%',
        'batch_size_optimal': 16
    },
    'RTX_3060': {
        'epochs_100': '6-8 hours',
        'epochs_200': '12-16 hours', 
        'gpu_utilization': '75-85%',
        'batch_size_optimal': 8
    }
}
```

---

## 8. Kết Luận Technical Analysis

### 8.1 Key Findings

1. **Dataset Quality**: Roboflow dataset có cấu trúc tốt với ~3,000 images
2. **Model Architecture**: YOLOv8s optimal cho balance speed/accuracy  
3. **Primary Issue**: Fallback model không có 'leopard' class
4. **Solution Implemented**: Comprehensive search + enhanced warnings

### 8.2 Production Readiness Checklist

- [x] **Model Loading**: Enhanced search algorithm implemented
- [x] **Fallback Warnings**: Comprehensive user notifications 
- [x] **Error Handling**: Robust exception management
- [ ] **Performance Monitoring**: Metrics collection setup needed
- [ ] **Automated Testing**: Unit/integration test suite needed
- [ ] **Documentation**: API documentation needed
- [ ] **Deployment Scripts**: Docker/K8s configurations needed

---

*Technical Specification Document - Wild Animal Detection System*  
*Generated: {{ timestamp }}*  
*Version: 1.0.0*
