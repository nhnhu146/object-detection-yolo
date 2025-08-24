# YOLO Object Detection Web Application - Requirement 1
# Web interface for object detection using available YOLO models

import os
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import uuid

# Import custom modules
from models.yolo_detector import YOLODetector
from utils.file_handler import FileHandler
from config import Config

# Flask application setup
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config.from_object(Config)
CORS(app)

# Create necessary directories
os.makedirs(app.config.get('UPLOAD_FOLDER', 'uploads'), exist_ok=True)
os.makedirs(app.config.get('RESULTS_FOLDER', 'results'), exist_ok=True)
os.makedirs(app.config.get('MODELS_DIR', 'weights'), exist_ok=True)

# Initialize components
print("🚀 Initializing YOLO Object Detection Web App...")

# Global detector instance - Model will be loaded once and reused (Requirement 1)
detector = YOLODetector()

# Load default model on startup (Requirement 1: model load only once)
print(f"📥 Loading default model: {app.config.get('DEFAULT_MODEL', 'yolov8n')}")
default_model = app.config.get('DEFAULT_MODEL', 'yolov8n')
if detector.load_model(default_model):
    print(f"✅ Default model {default_model} loaded successfully!")
    print("🔄 This model will be reused for all detections (Requirement 1)")
else:
    print(f"❌ Failed to load default model: {default_model}")
    print("⚠️  Model will be loaded on first request")

print("✅ App initialized successfully!")

# Initialize file handler
file_handler = FileHandler(
    upload_folder=app.config.get('UPLOAD_FOLDER', 'uploads'),
    results_folder=app.config.get('RESULTS_FOLDER', 'results'),
    allowed_extensions=app.config.get('ALLOWED_EXTENSIONS', {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}),
    max_file_size=app.config.get('MAX_FILE_SIZE', 16*1024*1024)
)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector.is_loaded(),
        'current_model': detector.current_model_name,
                        'app_name': 'LokWild - Look to Lock the Wild',
        'model_reused': True
    })

@app.route('/api/models')
def get_models():
    """Get available YOLO models (Requirement 1 & 2: show both general and specialized models)"""
    available_models = []
    current_config = app.config.get('AVAILABLE_MODELS', {})
    
    for model_id, model_info in current_config.items():
        model_data = {
            'id': model_id,
            'name': model_info['name'],
            'description': model_info['description'],
            'size': model_info['size'],
            'speed': model_info['speed'],
            'file': model_info['file'],
            'type': model_info['type'],
            'requirement': model_info['requirement'],
            'classes': model_info['classes'],
            'is_current': model_id == detector.current_model_name,
            'icon': model_info.get('icon', 'fas fa-brain'),
            'color': model_info.get('color', '#667eea')
        }
        
        # Add class names if available
        if 'class_names' in model_info:
            model_data['class_names'] = model_info['class_names']
        if 'class_names_vi' in model_info:
            model_data['class_names_vi'] = model_info['class_names_vi']
            
        available_models.append(model_data)
    
    return jsonify({
        'success': True,
        'models': available_models,
        'current_model': detector.current_model_name,
        'model_loaded': detector.is_loaded(),
        'message': f'Tìm thấy {len(available_models)} models khả dụng'
    })

@app.route('/api/switch_model', methods=['POST'])
def switch_model():
    """
    Switch to a different YOLO model
    Note: Use sparingly as per requirement - only when model needs to be updated
    """
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({'error': 'Tên model không được để trống'}), 400
        
        available_models = app.config.get('AVAILABLE_MODELS', {})
        if model_name not in available_models:
            return jsonify({'error': f'Model {model_name} không tồn tại'}), 400
        
        model_info = available_models[model_name]
        
        # Check if already using this model (avoid unnecessary reload)
        if detector.current_model_name == model_name:
            return jsonify({
                'success': True,
                'message': f'Đã đổi sang {model_info["name"]}',
                'model_name': model_name,
                'reused': True
            })
        
        # Switch model (will reload)
        print(f"🔄 Switching model from {detector.current_model_name} to {model_name}")
        if detector.switch_model(model_name):
            return jsonify({
                'success': True,
                'message': f'Đã đổi sang {model_info["name"]}',
                'model_name': model_name,
                'previous_model': detector.current_model_name,
                'reloaded': True
            })
        else:
            return jsonify({'error': f'Không thể tải {model_info["name"]}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Run object detection on uploaded image using the loaded model
    Model is reused for all predictions (Requirement 1 compliance)
    """
    try:
        # Check if file uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        if not file_handler.is_allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP, WebP'}), 400
        
        # Check if model is loaded - auto load default if not loaded
        if not detector.is_loaded():
            print(f"📥 No model loaded. Loading default model: {app.config.get('DEFAULT_MODEL', 'yolov8n')}")
            default_model = app.config.get('DEFAULT_MODEL', 'yolov8n')
            if not detector.load_model(default_model):
                return jsonify({'error': f'Failed to load default model: {default_model}'}), 500
            print(f"✅ Default model {default_model} loaded successfully!")
        else:
            print(f"🔄 Using already loaded model: {detector.current_model_name}")
        
        # Save uploaded file
        print(f"💾 Saving uploaded file: {file.filename}")
        save_result = file_handler.save_upload(file)
        if not save_result['success']:
            return jsonify({'error': save_result['error']}), 400
        
        upload_path = save_result['file_path']
        print(f"📁 File saved to: {upload_path}")
        
        # Get detection parameters
        confidence = float(request.form.get('confidence', app.config.get('DEFAULT_CONFIDENCE', 0.25)))
        iou = float(request.form.get('iou', app.config.get('DEFAULT_IOU', 0.45)))
        
        print(f"🎯 Running detection with model: {detector.current_model_name}")
        print(f"📊 Parameters: confidence={confidence}, iou={iou}")
        
        # Run prediction using the loaded model (model is reused)
        result = detector.predict(upload_path, confidence, iou)
        
        if not result or not result.get('success'):
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Save result image with bounding boxes
        print("💾 Saving result image...")
        result_save = file_handler.save_result_image(
            result['annotated_image'], 
            save_result['original_filename']
        )
        
        if not result_save['success']:
            return jsonify({'error': 'Failed to save result image'}), 500
        
        # Convert annotated image to base64 for frontend display
        _, buffer = cv2.imencode('.jpg', result['annotated_image'])
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        print(f"✅ Detection completed: {len(result['predictions'])} objects found")
        print(f"🔄 Model {detector.current_model_name} was reused (not reloaded)")
        
        return jsonify({
            'success': True,
            'predictions': result['predictions'],
            'result_image': f"data:image/jpeg;base64,{result_base64}",
            'stats': result['stats'],
            'model_info': result['model_info'],
            'upload_info': {
                'filename': save_result['original_filename'],
                'size': save_result['size']
            }
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/results/<filename>')
def serve_result(filename):
    """Serve result images"""
    return send_from_directory(app.config.get('RESULTS_FOLDER', 'results'), filename)

@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve uploaded images"""
    return send_from_directory(app.config.get('UPLOAD_FOLDER', 'uploads'), filename)

# Cleanup old files periodically
def cleanup_files():
    """Clean up old uploaded and result files"""
    try:
        file_handler.cleanup_old_files(max_age_hours=24)
        print("🧹 Cleaned up old files")
    except Exception as e:
        print(f"⚠️ Cleanup error: {e}")

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 LokWild - Look to Lock the Wild")
    print("="*50)
    print(f"📊 Model loaded: {detector.current_model_name}")
    print(f"🔄 Model reuse: Enabled (as required)")
    print(f"🌐 Server: http://localhost:5000")
    print("="*50 + "\n")
    
    # Start Flask development server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Disable reloader to prevent model from being loaded multiple times
    )
