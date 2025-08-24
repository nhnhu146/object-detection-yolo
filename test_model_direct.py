#!/usr/bin/env python3
"""
Script test đơn giản để kiểm tra YOLO model loading và class names
Chỉ cần ultralytics library
"""

def test_model_direct():
    """Test trực tiếp với ultralytics YOLO"""
    try:
        print("=" * 60)
        print("🔍 TEST TRỰC TIẾP YOLO MODEL")
        print("=" * 60)
        
        # Try to import ultralytics
        try:
            from ultralytics import YOLO
            print("✅ ultralytics imported thành công")
        except ImportError:
            print("❌ Chưa cài ultralytics. Cần chạy: pip install ultralytics")
            return
            
        # Test YOLOv8s model
        print("\n📊 Loading YOLOv8s model...")
        try:
            model = YOLO('yolov8s.pt')
            print("✅ YOLOv8s model loaded thành công")
            
            # Get class names from actual model
            model_classes = list(model.names.values())
            print(f"📋 Model có {len(model_classes)} classes")
            
            # Show first 10 classes
            print("📋 Classes từ model (10 đầu tiên):")
            for i, class_name in enumerate(model_classes[:10]):
                print(f"   {i}: '{class_name}'")
                
            print(f"   ... và {len(model_classes) - 10} classes khác")
            
            # Check specific classes that might cause issues
            problem_classes = []
            common_objects = ["person", "car", "dog", "cat", "bird", "horse", "cow", "elephant"]
            
            print("\n🔍 Kiểm tra các classes phổ biến:")
            for obj in common_objects:
                if obj in model_classes:
                    index = model_classes.index(obj)
                    print(f"   ✅ '{obj}' -> index {index}")
                else:
                    print(f"   ❌ '{obj}' -> KHÔNG TÌM THẤY")
                    problem_classes.append(obj)
                    
            if problem_classes:
                print(f"\n❌ Không tìm thấy {len(problem_classes)} classes: {problem_classes}")
            else:
                print("\n✅ Tất cả classes phổ biến đều có trong model")
                
            # Check class name types and format
            print("\n🔍 Kiểm tra format class names:")
            for i, class_name in enumerate(model_classes[:5]):
                print(f"   {i}: '{class_name}' (type: {type(class_name)}, len: {len(class_name)})")
                
            # Compare với config
            print("\n📊 So sánh với config...")
            try:
                import sys
                import os
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
                from src.config import Config
                
                config = Config()
                config_classes = config.AVAILABLE_MODELS["yolov8s"]["class_names"]
                
                print(f"📊 Config có {len(config_classes)} classes")
                print(f"📊 Model có {len(model_classes)} classes")
                
                if len(config_classes) == len(model_classes):
                    print("✅ Số lượng classes khớp")
                    
                    # Check exact matches
                    mismatches = []
                    for i in range(len(model_classes)):
                        if model_classes[i] != config_classes[i]:
                            mismatches.append({
                                'index': i,
                                'model': model_classes[i],
                                'config': config_classes[i]
                            })
                            
                    if mismatches:
                        print(f"❌ Phát hiện {len(mismatches)} class names không khớp:")
                        for mismatch in mismatches[:10]:
                            print(f"   [{mismatch['index']}] Model: '{mismatch['model']}' != Config: '{mismatch['config']}'")
                        
                        print("\n💡 NGUYÊN NHÂN CÓ THỂ:")
                        print("   1. Config class names không đúng với YOLOv8s")
                        print("   2. Version YOLO khác nhau có class names khác nhau") 
                        print("   3. Model file bị corrupted")
                        
                    else:
                        print("✅ Tất cả class names đều khớp giữa model và config")
                        
                else:
                    print(f"❌ Số lượng classes không khớp!")
                    print(f"   Config: {len(config_classes)} classes")
                    print(f"   Model:  {len(model_classes)} classes")
                    
            except Exception as e:
                print(f"⚠️  Không thể so sánh với config: {e}")
                
        except Exception as e:
            print(f"❌ Lỗi loading model: {e}")
            
        print("\n" + "=" * 60)
        print("✅ HOÀN THÀNH TEST MODEL")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Lỗi tổng quát: {e}")
        import traceback
        traceback.print_exc()

def test_with_image():
    """Test prediction với một ảnh sample nếu có"""
    try:
        print("\n" + "=" * 60)
        print("🔍 TEST PREDICTION (NẾU CÓ ẢNH)")
        print("=" * 60)
        
        from ultralytics import YOLO
        import os
        
        # Look for sample images
        sample_dirs = ["uploads", "test_images", "samples", "."]
        sample_files = []
        
        for directory in sample_dirs:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        sample_files.append(os.path.join(directory, file))
                        
        if sample_files:
            print(f"📁 Tìm thấy {len(sample_files)} ảnh sample")
            test_image = sample_files[0]
            print(f"🖼️  Test với ảnh: {test_image}")
            
            model = YOLO('yolov8s.pt')
            results = model(test_image, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    print(f"✅ Phát hiện {len(result.boxes)} objects:")
                    
                    for i, box in enumerate(result.boxes):
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model.names[cls]
                        
                        print(f"   {i+1}. Class: '{class_name}' (ID: {cls}, Confidence: {conf:.2f})")
                        
                else:
                    print("⚠️  Không phát hiện object nào trong ảnh")
            else:
                print("❌ Không có kết quả prediction")
                
        else:
            print("⚠️  Không tìm thấy ảnh sample để test")
            print("💡 Có thể upload ảnh vào thư mục uploads/ để test")
            
    except Exception as e:
        print(f"⚠️  Không thể test prediction: {e}")

if __name__ == "__main__":
    test_model_direct()
    test_with_image()
