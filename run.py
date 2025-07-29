#!/usr/bin/env python3
"""
Run script for YOLO Object Detection Web App - Requirement 1
Cross-platform launcher for using available YOLO models
"""

import os
import sys
import subprocess

def check_python():
    """Check if Python is available"""
    try:
        result = subprocess.run([sys.executable, '--version'], 
                              capture_output=True, text=True)
        print(f"✅ Python found: {result.stdout.strip()}")
        return True
    except:
        print("❌ Python not found!")
        return False

def check_conda_env():
    """Check if conda yolo environment is activated"""
    if 'CONDA_DEFAULT_ENV' in os.environ:
        env_name = os.environ['CONDA_DEFAULT_ENV']
        if env_name == 'yolo':
            print(f"✅ Conda environment: {env_name}")
            return True
        else:
            print(f"⚠️  Current conda env: {env_name}")
            print("💡 Recommendation: conda activate yolo")
    else:
        print("⚠️  No conda environment detected")
        print("💡 Recommendation: conda activate yolo")
    return True  # Don't fail, just warn

def check_requirements():
    """Check if requirements are installed for Requirement 1"""
    required_packages = ['flask', 'cv2', 'numpy', 'ultralytics']
    missing_packages = []
    
    try:
        import flask
        print("✅ Flask installed")
    except ImportError:
        missing_packages.append('flask')
        
    try:
        import cv2
        print("✅ OpenCV installed") 
    except ImportError:
        missing_packages.append('opencv-python')
        
    try:
        import numpy
        print("✅ NumPy installed")
    except ImportError:
        missing_packages.append('numpy')
        
    try:
        import ultralytics
        print("✅ YOLOv8 (ultralytics) installed")
    except ImportError:
        missing_packages.append('ultralytics')
    
    if missing_packages:
        print(f"⚠️  Missing packages: {', '.join(missing_packages)}")
        return install_requirements()
    else:
        print("✅ All requirements installed")
        return True

def install_requirements():
    """Install requirements for Requirement 1"""
    print("\n📦 Installing requirements for YOLO Object Detection...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        print("💡 Try: conda activate yolo && pip install -r requirements.txt")
        return False

def run_app():
    """Run the YOLO Object Detection Flask application"""
    print("\n🚀 Starting YOLO Object Detection Web App - Requirement 1...")
    print("📋 Features:")
    print("   ✅ Available YOLO models (YOLOv8n/s/m/l)")
    print("   ✅ Model loads once and reused for all detections")
    print("   ✅ Web interface for image upload and detection")
    print("🌐 Access the app at: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server\n")
    
    # Change to src directory
    src_dir = os.path.join(os.path.dirname(__file__), 'src')
    if os.path.exists(src_dir):
        os.chdir(src_dir)
        print(f"📁 Working directory: {src_dir}")
    else:
        print("❌ src directory not found!")
        return False
    
    try:
        subprocess.run([sys.executable, 'app.py'])
        return True
    except KeyboardInterrupt:
        print("\n\n👋 App stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Error running app: {e}")
        return False

def main():
    """Main function for Requirement 1 launcher"""
    print("🔍 YOLO Object Detection Web App - Requirement 1")
    print("=" * 55)
    print("📋 Project: Using available YOLO models with web interface")
    print("🎯 Goal: Model loads once, reused for all detections")
    print("=" * 55)
    
    # Check Python
    if not check_python():
        sys.exit(1)
    
    # Check conda environment
    check_conda_env()
    
    # Check and install requirements
    if not check_requirements():
        print("\n❌ Failed to setup requirements")
        print("💡 Manual fix: conda activate yolo && pip install -r requirements.txt")
        sys.exit(1)
    
    # Run the app
    print("\n🎉 Setup completed! Starting web application...")
    if not run_app():
        sys.exit(1)

if __name__ == '__main__':
    main()
