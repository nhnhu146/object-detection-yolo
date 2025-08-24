# 🤖 AI Object Detection Studio - Requirement 1

Web application phát hiện đối tượng AI sử dụng YOLO models với giao diện hiện đại.

## 📋 Yêu cầu Requirement 1
- ✅ Cài đặt YOLO và sử dụng available models  
- ✅ Xây dựng chương trình với web interface hiện đại
- ✅ Cho phép upload ảnh và trả về kết quả object detection
- ✅ **Model chỉ load 1 lần và dùng cho tất cả detection** (tuân thủ quy định)
- ✅ Giao diện đẹp với thiết kế AI Object Detection Studio

## 🏗️ Kiến trúc dự án

```
src/
├── app.py              # Flask application chính
├── config.py           # Cấu hình models và settings
├── static/            
│   ├── favicon.svg     # Logo AI Studio
│   └── favicon.ico     # Favicon backup
├── models/
│   └── yolo_detector.py # Logic YOLO detection
├── utils/
│   └── file_handler.py  # Xử lý file upload/result
├── templates/
│   └── index.html       # Giao diện web hiện đại
├── uploads/            # Thư mục ảnh upload (tự tạo)
├── results/            # Thư mục kết quả (tự tạo)
└── weights/            # Thư mục models (tự tạo)
```

## 🎯 Models được hỗ trợ

- **YOLOv8s**: Pre-trained model (21.5MB) - Cân bằng tốc độ và độ chính xác
- **Custom Trained Model**: Dành cho model tự train (placeholder)

## 🚀 Cài đặt và chạy

### Cách 1: Chạy tự động (Khuyên dùng)
```bash
python run.py
```
Script này sẽ tự động:
- Kiểm tra và cài đặt dependencies
- Tạo virtual environment nếu cần
- Khởi động ứng dụng

### Cách 2: Cài đặt thủ công
```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy ứng dụng
python src/app.py
```

## 📱 Cách sử dụng

1. **Khởi động**: Chạy `python run.py` hoặc `python src/app.py`
2. **Truy cập**: Mở http://localhost:5000 
3. **Chọn model**: YOLOv8s (pre-trained) hoặc Custom Trained Model (tự train)
4. **Upload ảnh**: Drag-drop hoặc click chọn file
5. **Xem kết quả**: Bounding boxes + confidence scores + thống kê

## ⚠️ Quan trọng - Tuân thủ Requirement 1
- ✅ Model YOLOv8s được load sẵn lúc khởi động
- ✅ Model được **reuse** cho tất cả detection (không reload)
- ✅ Chỉ reload khi user chọn model khác (theo quy định)
- ✅ Web interface hiện đại với thiết kế AI Object Detection Studio

## 🎨 Giao diện Features
- 🤖 **Modern AI Studio Design**: Thiết kế hiện đại với gradient và glass morphism
- 🎯 **Model Selection**: Chọn model đơn giản với 2 options
- 📤 **Drag & Drop Upload**: Kéo thả file dễ dàng
- 📊 **Real-time Results**: Hiển thị kết quả với metrics và predictions
- 📱 **Responsive**: Tối ưu cho mọi thiết bị
- 🚀 **Smooth Animations**: Hiệu ứng mượt mà không quá phức tạp

## 💾 Dependencies
```bash
ultralytics>=8.0.0
torch>=1.13.0
flask>=2.3.0
flask-cors>=4.0.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=9.5.0
```

## 🔧 Cấu hình chính
- **Default Model**: YOLOv8s (cân bằng tốc độ/chính xác)
- **Upload Limit**: 16MB
- **Supported Formats**: JPG, PNG, GIF, BMP, WebP
- **Auto Cleanup**: Xóa file cũ sau 24h

## 📁 Thư mục tự động tạo
Các thư mục này sẽ được tạo tự động khi chạy:
- `src/uploads/` - Lưu ảnh upload
- `src/results/` - Lưu ảnh kết quả  
- `src/weights/` - Lưu model files