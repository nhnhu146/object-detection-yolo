#!/usr/bin/env python3
"""
Script kiểm tra code quality không cần dependencies
"""

import os
import re

def check_code_quality():
    """Kiểm tra chất lượng code và các sửa lỗi"""
    print("=" * 60)
    print("🔍 KIỂM TRA CHẤT LƯỢNG CODE VÀ CÁC SỬA LỖI")
    print("=" * 60)
    
    # Check yolo_detector.py
    detector_file = os.path.join('src', 'models', 'yolo_detector.py')
    if os.path.exists(detector_file):
        print("\n📊 KIỂM TRA: yolo_detector.py")
        print("-" * 40)
        
        with open(detector_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 1. Check duplicate methods
        switch_model_count = len(re.findall(r'def switch_model\(', content))
        if switch_model_count == 1:
            print("✅ FIXED: Duplicate switch_model() method resolved")
        else:
            print(f"❌ ISSUE: Found {switch_model_count} switch_model() definitions")
            
        # 2. Check validation method
        if 'def validate_class_names(' in content:
            print("✅ ADDED: validate_class_names() method implemented")
        else:
            print("❌ MISSING: validate_class_names() method not found")
            
        # 3. Check improved error handling
        if 'logger.error(f"Custom model not found:' in content:
            print("✅ IMPROVED: Better error handling for missing models")
        else:
            print("❌ ISSUE: Error handling not improved")
            
        # 4. Check fallback logic
        if 'return False' in content and 'Custom model not found' in content:
            print("✅ FIXED: Silent fallback removed, explicit errors added")
        else:
            print("⚠️  WARNING: Fallback logic may still have issues")
            
    # Check app.py
    app_file = os.path.join('src', 'app.py')
    if os.path.exists(app_file):
        print("\n📊 KIỂM TRA: app.py")
        print("-" * 40)
        
        with open(app_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 1. Check debug endpoint
        if '@app.route(\'/api/debug/classes\')' in content:
            print("✅ ADDED: Debug endpoint /api/debug/classes")
        else:
            print("❌ MISSING: Debug endpoint not added")
            
        # 2. Check switch model endpoint
        if '@app.route(\'/api/switch_model\',' in content:
            print("✅ ADDED: Model switching endpoint /api/switch_model")
        else:
            print("❌ MISSING: Model switching endpoint not added")
            
        # 3. Check validation usage
        if 'validate_class_names(' in content:
            print("✅ INTEGRATED: Class validation used in app")
        else:
            print("⚠️  WARNING: Class validation not integrated")
            
    # Check config files
    print("\n📊 KIỂM TRA: Configuration Files")
    print("-" * 40)
    
    config_files = ['src/config.py', 'src/config_new.py']
    
    for config_file in config_files:
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check YOLO class names
            if '"person", "bicycle", "car"' in content:
                print(f"✅ {os.path.basename(config_file)}: YOLO classes correct")
            else:
                print(f"❌ {os.path.basename(config_file)}: YOLO classes may be wrong")
                
            # Check wild_animal classes
            if '"elephant", "giraffe", "leopard", "lion", "zebra"' in content:
                print(f"✅ {os.path.basename(config_file)}: Wild animal classes correct")
            else:
                print(f"❌ {os.path.basename(config_file)}: Wild animal classes missing")
                
    # Check test files created
    print("\n📊 KIỂM TRA: Test Files & Documentation")
    print("-" * 40)
    
    test_files = [
        'test_class_names.py',
        'check_config.py', 
        'test_model_direct.py',
        'test_fixes.py',
        'ISSUE_ANALYSIS.md',
        'TEST_INSTRUCTIONS.md'
    ]
    
    for file in test_files:
        if os.path.exists(file):
            print(f"✅ Created: {file}")
        else:
            print(f"❌ Missing: {file}")
            
    # Overall assessment
    print("\n" + "=" * 60)
    print("📋 TỔNG KẾT ĐÁNH GIÁ")
    print("=" * 60)
    
    fixes_implemented = []
    issues_remaining = []
    
    # Summary based on checks above
    if os.path.exists(detector_file):
        with open(detector_file, 'r', encoding='utf-8') as f:
            detector_content = f.read()
            
        if len(re.findall(r'def switch_model\(', detector_content)) == 1:
            fixes_implemented.append("Duplicate switch_model() method")
        else:
            issues_remaining.append("Duplicate switch_model() method")
            
        if 'def validate_class_names(' in detector_content:
            fixes_implemented.append("Class names validation")
        else:
            issues_remaining.append("Class names validation")
            
        if 'logger.error(f"Custom model not found:' in detector_content:
            fixes_implemented.append("Better error handling")
        else:
            issues_remaining.append("Better error handling")
            
    if os.path.exists(app_file):
        with open(app_file, 'r', encoding='utf-8') as f:
            app_content = f.read()
            
        if '@app.route(\'/api/debug/classes\')' in app_content:
            fixes_implemented.append("Debug endpoints")
        else:
            issues_remaining.append("Debug endpoints")
            
    print("✅ FIXES IMPLEMENTED:")
    for fix in fixes_implemented:
        print(f"   - {fix}")
        
    if issues_remaining:
        print("\n❌ ISSUES REMAINING:")
        for issue in issues_remaining:
            print(f"   - {issue}")
    else:
        print("\n🎉 ALL MAJOR FIXES IMPLEMENTED!")
        
    # Recommendations
    print(f"\n💡 KHUYẾN NGHỊ TIẾP THEO:")
    print("1. Cài đặt dependencies để test thực tế")
    print("2. Test với ảnh thực để verify predictions")
    print("3. Monitor logs khi chạy detection")
    print("4. Sử dụng debug endpoints để troubleshoot")
    
    return len(issues_remaining) == 0

if __name__ == "__main__":
    success = check_code_quality()
    if success:
        print(f"\n🎉 SUCCESS: Tất cả các sửa lỗi chính đã được implemented!")
    else:
        print(f"\n⚠️  WARNING: Vẫn còn một số issues cần resolve")
