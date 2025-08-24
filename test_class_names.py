#!/usr/bin/env python3
"""
Test script để kiểm tra lỗi detect sai tên class
"""

import sys
import os
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_yolo_detector():
    """Test YOLO Detector class names"""
    try:
        print("=" * 60)
        print("🔍 KIỂM TRA LỖI DETECT SAI TÊN CLASS")
        print("=" * 60)
        
        # Import modules
        from src.models.yolo_detector import YOLODetector
        from src.config import Config
        
        print("✅ Import modules thành công")
        
        # Initialize detector
        detector = YOLODetector()
        print("✅ Khởi tạo YOLODetector thành công")
        
        # Test with YOLOv8s model
        print("\n📊 TEST 1: YOLOv8s Model")
        print("-" * 40)
        
        if detector.load_model("yolov8s"):
            print(f"✅ Load model 'yolov8s' thành công")
            print(f"📝 Model hiện tại: {detector.current_model_name}")
            print(f"📊 Số lượng classes: {len(detector.class_names)}")
            print("📋 Class names từ model:")
            
            # Get class names from model
            model_classes = detector.get_available_classes()
            for i, class_name in enumerate(model_classes[:10]):  # Show first 10
                print(f"   {i}: {class_name}")
            if len(model_classes) > 10:
                print(f"   ... và {len(model_classes) - 10} classes khác")
                
            # Get model info
            model_info = detector.get_model_info()
            print(f"📁 Model file: {model_info.get('file', 'N/A')}")
            
        else:
            print("❌ Không thể load model yolov8s")
            
        # Test with config class names
        print("\n📊 TEST 2: Config Class Names")
        print("-" * 40)
        
        config = Config()
        available_models = config.AVAILABLE_MODELS
        
        for model_id, model_info in available_models.items():
            print(f"\n🔧 Model: {model_id}")
            print(f"   📝 Name: {model_info['name']}")
            print(f"   📊 Classes: {model_info['classes']}")
            
            if 'class_names' in model_info:
                config_classes = model_info['class_names']
                print(f"   📋 Config class names ({len(config_classes)}):")
                for i, class_name in enumerate(config_classes[:5]):  # Show first 5
                    print(f"      {i}: {class_name}")
                if len(config_classes) > 5:
                    print(f"      ... và {len(config_classes) - 5} classes khác")
            else:
                print("   ⚠️  Không có class_names trong config")
                
            if 'class_names_vi' in model_info:
                vi_classes = model_info['class_names_vi']
                print(f"   🇻🇳 Vietnamese names ({len(vi_classes)}):")
                for i, class_name in enumerate(vi_classes):
                    print(f"      {i}: {class_name}")
                    
        # Test wild_animal model
        print("\n📊 TEST 3: Wild Animal Model")
        print("-" * 40)
        
        if detector.load_model("wild_animal"):
            print(f"✅ Load model 'wild_animal' thành công")
            print(f"📝 Model hiện tại: {detector.current_model_name}")
            print(f"📊 Số lượng classes: {len(detector.class_names)}")
            print("📋 Class names từ model:")
            
            # Get class names from model
            model_classes = detector.get_available_classes()
            for i, class_name in enumerate(model_classes):
                print(f"   {i}: {class_name}")
                
        else:
            print("⚠️  Không thể load model wild_animal (có thể chưa có file)")
            
        # Compare class names
        print("\n📊 TEST 4: So sánh Class Names")
        print("-" * 40)
        
        # Load yolov8s again
        detector.load_model("yolov8s")
        model_classes = detector.get_available_classes()
        config_classes = available_models["yolov8s"]["class_names"]
        
        print(f"📊 Model classes: {len(model_classes)}")
        print(f"📊 Config classes: {len(config_classes)}")
        
        if len(model_classes) == len(config_classes):
            print("✅ Số lượng classes khớp")
            
            # Check if class names match
            mismatched = []
            for i in range(len(model_classes)):
                if i < len(config_classes) and model_classes[i] != config_classes[i]:
                    mismatched.append({
                        'index': i,
                        'model': model_classes[i],
                        'config': config_classes[i]
                    })
                    
            if mismatched:
                print(f"❌ Phát hiện {len(mismatched)} class names không khớp:")
                for mismatch in mismatched[:10]:  # Show first 10 mismatches
                    print(f"   [{mismatch['index']}] Model: '{mismatch['model']}' != Config: '{mismatch['config']}'")
            else:
                print("✅ Tất cả class names đều khớp")
        else:
            print(f"❌ Số lượng classes không khớp: Model={len(model_classes)}, Config={len(config_classes)}")
            
        print("\n" + "=" * 60)
        print("✅ HOÀN THÀNH KIỂM TRA")
        print("=" * 60)
        
    except ImportError as e:
        print(f"❌ Lỗi import: {e}")
        print("💡 Hãy đảm bảo đã cài đặt các dependencies:")
        print("   pip install ultralytics torch torchvision opencv-python numpy")
    except Exception as e:
        print(f"❌ Lỗi không mong đợi: {e}")
        print("📊 Stack trace:")
        traceback.print_exc()

if __name__ == "__main__":
    test_yolo_detector()
