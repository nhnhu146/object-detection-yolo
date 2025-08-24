#!/usr/bin/env python3
"""
Script kiểm tra class names configuration
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_class_names():
    """Kiểm tra class names trong config"""
    try:
        print("=" * 60)
        print("🔍 KIỂM TRA CLASS NAMES CONFIGURATION")
        print("=" * 60)
        
        # Import config
        from src.config import Config
        
        print("✅ Import config thành công")
        
        config = Config()
        available_models = config.AVAILABLE_MODELS
        
        print(f"📊 Tổng số models: {len(available_models)}")
        
        # Standard COCO classes for YOLOv8
        COCO_CLASSES = [
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
        ]
        
        print(f"📋 COCO standard classes: {len(COCO_CLASSES)}")
        
        for model_id, model_info in available_models.items():
            print(f"\n🔧 MODEL: {model_id}")
            print("=" * 40)
            print(f"   📝 Name: {model_info['name']}")
            print(f"   📊 Expected classes: {model_info['classes']}")
            
            if 'class_names' in model_info:
                config_classes = model_info['class_names']
                print(f"   📋 Config class names: {len(config_classes)}")
                
                # Check if it matches expected count
                if len(config_classes) == model_info['classes']:
                    print("   ✅ Số lượng classes khớp với khai báo")
                else:
                    print(f"   ❌ Không khớp! Expected: {model_info['classes']}, Got: {len(config_classes)}")
                    
                # For YOLOv8s, compare with COCO classes
                if model_id == "yolov8s":
                    if len(config_classes) == len(COCO_CLASSES):
                        print("   ✅ Khớp với số lượng COCO classes")
                        
                        # Check exact matches
                        mismatches = []
                        for i in range(min(len(config_classes), len(COCO_CLASSES))):
                            if config_classes[i] != COCO_CLASSES[i]:
                                mismatches.append({
                                    'index': i,
                                    'config': config_classes[i],
                                    'coco': COCO_CLASSES[i]
                                })
                                
                        if mismatches:
                            print(f"   ❌ Phát hiện {len(mismatches)} class names sai:")
                            for mismatch in mismatches[:5]:  # Show first 5
                                print(f"      [{mismatch['index']}] Config: '{mismatch['config']}' != COCO: '{mismatch['coco']}'")
                            if len(mismatches) > 5:
                                print(f"      ... và {len(mismatches) - 5} lỗi khác")
                        else:
                            print("   ✅ Tất cả class names đều chính xác")
                    else:
                        print(f"   ❌ Số lượng không khớp COCO: Config={len(config_classes)}, COCO={len(COCO_CLASSES)}")
                        
                # Show first few classes
                print("   📋 Class names (5 đầu tiên):")
                for i, class_name in enumerate(config_classes[:5]):
                    print(f"      {i}: '{class_name}'")
                    
            else:
                print("   ⚠️  Không có 'class_names' trong config")
                
            # Check Vietnamese names
            if 'class_names_vi' in model_info:
                vi_classes = model_info['class_names_vi']
                print(f"   🇻🇳 Vietnamese names: {len(vi_classes)}")
                
                if 'class_names' in model_info:
                    if len(vi_classes) == len(model_info['class_names']):
                        print("   ✅ Số lượng Vietnamese names khớp")
                    else:
                        print(f"   ❌ Vietnamese names không khớp: EN={len(model_info['class_names'])}, VI={len(vi_classes)}")
                        
                # Show Vietnamese names
                print("   🇻🇳 Vietnamese names:")
                for i, class_name in enumerate(vi_classes):
                    print(f"      {i}: '{class_name}'")
                    
        # Additional checks
        print(f"\n🔍 PHÂN TÍCH VẤN ĐỀ")
        print("=" * 40)
        
        yolo_config = available_models.get("yolov8s", {})
        if 'class_names' in yolo_config:
            yolo_classes = yolo_config['class_names']
            print(f"✅ YOLOv8s config có {len(yolo_classes)} classes")
            
            # Check common detection issues
            common_issues = []
            
            # Check for missing common classes
            common_objects = ["person", "car", "dog", "cat", "bird"]
            for obj in common_objects:
                if obj not in yolo_classes:
                    common_issues.append(f"Thiếu class '{obj}'")
                    
            # Check for duplicate classes
            if len(yolo_classes) != len(set(yolo_classes)):
                duplicates = []
                seen = set()
                for cls in yolo_classes:
                    if cls in seen:
                        duplicates.append(cls)
                    seen.add(cls)
                common_issues.append(f"Classes trùng lặp: {duplicates}")
                
            # Check for empty or None classes
            empty_classes = [i for i, cls in enumerate(yolo_classes) if not cls or cls.strip() == ""]
            if empty_classes:
                common_issues.append(f"Classes rỗng tại index: {empty_classes}")
                
            if common_issues:
                print("❌ Các vấn đề phát hiện:")
                for issue in common_issues:
                    print(f"   - {issue}")
            else:
                print("✅ Không phát hiện vấn đề rõ ràng trong config")
                
        print(f"\n💡 KHUYẾN NGHỊ")
        print("=" * 40)
        print("1. Kiểm tra model thực tế được load có khớp với config không")
        print("2. Test prediction với 1 ảnh cụ thể để xem class nào được detect")
        print("3. So sánh kết quả với expected classes")
        print("4. Kiểm tra version của ultralytics YOLO")
        
        print("\n" + "=" * 60)
        print("✅ HOÀN THÀNH KIỂM TRA CONFIG")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_class_names()
