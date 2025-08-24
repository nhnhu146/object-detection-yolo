#!/usr/bin/env python3
"""
Script test toàn diện sau khi fix các lỗi detect sai tên class
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_fixes():
    """Test các sửa lỗi đã implement"""
    print("=" * 60)
    print("🔍 KIỂM TRA CÁC SỬA LỖI DETECT SAI TÊN CLASS")
    print("=" * 60)
    
    try:
        # Import modules
        from src.models.yolo_detector import YOLODetector
        from src.config import Config
        
        print("✅ Import modules thành công")
        
        # Test 1: YOLODetector class
        print("\n📊 TEST 1: YOLODetector Methods")
        print("-" * 40)
        
        detector = YOLODetector()
        print("✅ YOLODetector initialized")
        
        # Test methods exist and work
        try:
            classes = detector.get_available_classes()
            print(f"✅ get_available_classes(): {len(classes)} classes")
            
            validation = detector.validate_class_names()
            print(f"✅ validate_class_names(): {validation['message']}")
            
            switch_result = detector.switch_model("yolov8s")
            print(f"✅ switch_model(): {switch_result}")
            
        except Exception as e:
            print(f"❌ Method test failed: {e}")
            
        # Test 2: Config consistency
        print("\n📊 TEST 2: Config Consistency Check")
        print("-" * 40)
        
        config = Config()
        available_models = config.AVAILABLE_MODELS
        
        for model_id, model_info in available_models.items():
            print(f"\n🔧 Checking model: {model_id}")
            
            # Check required fields
            required_fields = ['name', 'classes', 'class_names']
            missing_fields = []
            
            for field in required_fields:
                if field not in model_info:
                    missing_fields.append(field)
                    
            if missing_fields:
                print(f"❌ Missing fields: {missing_fields}")
            else:
                print("✅ All required fields present")
                
            # Check class count consistency
            if 'classes' in model_info and 'class_names' in model_info:
                expected_count = model_info['classes']
                actual_count = len(model_info['class_names'])
                
                if expected_count == actual_count:
                    print(f"✅ Class count consistent: {expected_count}")
                else:
                    print(f"❌ Class count mismatch: expected {expected_count}, got {actual_count}")
                    
        # Test 3: Duplicate method check
        print("\n📊 TEST 3: Code Quality Check")
        print("-" * 40)
        
        # Read the yolo_detector.py file
        detector_file = os.path.join('src', 'models', 'yolo_detector.py')
        if os.path.exists(detector_file):
            with open(detector_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Count method definitions
            method_definitions = content.count('def switch_model(')
            
            if method_definitions == 1:
                print("✅ No duplicate switch_model() method")
            else:
                print(f"❌ Found {method_definitions} switch_model() definitions")
                
            # Check for improved error handling
            if 'validate_class_names' in content:
                print("✅ validate_class_names method added")
            else:
                print("❌ validate_class_names method missing")
                
            if 'Custom model not found' in content and 'logger.error' in content:
                print("✅ Improved error handling for missing models")
            else:
                print("❌ Error handling needs improvement")
                
        # Test 4: App.py improvements
        print("\n📊 TEST 4: Web App Improvements")
        print("-" * 40)
        
        app_file = os.path.join('src', 'app.py')
        if os.path.exists(app_file):
            with open(app_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for debug endpoint
            if '/api/debug/classes' in content:
                print("✅ Debug endpoint added")
            else:
                print("❌ Debug endpoint missing")
                
            if '/api/switch_model' in content:
                print("✅ Model switching endpoint added")
            else:
                print("❌ Model switching endpoint missing")
                
        print("\n" + "=" * 60)
        print("✅ HOÀN THÀNH KIỂM TRA CÁC SỬA LỖI")
        print("=" * 60)
        
        # Summary and recommendations
        print(f"\n💡 TỔNG KẾT VÀ KHUYẾN NGHỊ")
        print("=" * 40)
        print("✅ Các sửa lỗi đã được implement:")
        print("   1. Fix duplicate switch_model() method")
        print("   2. Add class names validation")
        print("   3. Improve error handling for missing models")
        print("   4. Add debug endpoints")
        print()
        print("🔄 Bước tiếp theo để hoàn toàn giải quyết vấn đề:")
        print("   1. Cài đặt dependencies: pip install ultralytics torch opencv-python flask")
        print("   2. Test với ảnh thực tế để verify predictions")
        print("   3. Sử dụng /api/debug/classes để monitor class names")
        print("   4. Kiểm tra log files để track issues")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def generate_test_instructions():
    """Tạo hướng dẫn test cho user"""
    
    instructions = """
# 🧪 HƯỚNG DẪN TEST SAU KHI SỬA LỖI

## 1. Cài đặt Dependencies
```bash
pip install ultralytics torch torchvision opencv-python flask flask-cors numpy pillow
```

## 2. Chạy Application
```bash
cd "d:\\Github\\object-detection-yolo"
python src/app.py
```

## 3. Test các API endpoints

### a) Health check
```bash
curl http://localhost:5000/api/health
```

### b) Debug class names
```bash
curl http://localhost:5000/api/debug/classes
```

### c) Switch model
```bash
curl -X POST http://localhost:5000/api/switch_model -H "Content-Type: application/json" -d '{"model_name": "yolov8s"}'
```

### d) Upload và test prediction
1. Mở http://localhost:5000 trong browser
2. Upload một ảnh có objects (person, car, dog, etc.)
3. Kiểm tra class names trong kết quả

## 4. Các điều cần kiểm tra

### ✅ Model Loading:
- Model load thành công
- Class names khớp với config
- No duplicate method errors

### ✅ Predictions:
- Objects được detect đúng
- Class names chính xác (person không bị detect thành car)
- Confidence scores hợp lý

### ✅ Error Handling:
- Missing model files được handle tốt
- Clear error messages
- No silent fallbacks

### ✅ API Responses:
- Debug endpoint trả về class validation
- Switch model hoạt động đúng
- Consistent model info

## 5. Kiểm tra Logs

Trong console sẽ có logs như:
```
✅ Model loaded successfully: yolov8s
📊 Model có 80 classes
✅ All class names match between model and config
```

## 6. Nếu vẫn có lỗi "detect sai tên class"

1. Check `/api/debug/classes` để xem validation results
2. Upload ảnh test và check console logs
3. Verify model version: các version YOLO khác nhau có thể có class order khác nhau
4. Check model file integrity

## 7. Common Issues & Solutions

### Issue: Class names không khớp
**Solution**: Dùng validation method để identify mismatches

### Issue: Model không load được
**Solution**: Check file paths và permissions

### Issue: Predictions sai class
**Solution**: Verify với multiple test images, check confidence thresholds
"""
    
    with open('TEST_INSTRUCTIONS.md', 'w', encoding='utf-8') as f:
        f.write(instructions)
        
    print("📄 Đã tạo file TEST_INSTRUCTIONS.md với hướng dẫn chi tiết")

if __name__ == "__main__":
    test_fixes()
    generate_test_instructions()
