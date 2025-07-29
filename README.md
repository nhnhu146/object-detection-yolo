# YOLO Object Detection Web App - Requirement 1

Web interface cho object detection sử dụng available YOLO models (YOLOv8).

## 📋 Yêu cầu Requirement 1
- ✅ Cài đặt YOLO và sử dụng available models
- ✅ Xây dựng chương trình với web interface  
- ✅ Cho phép insert ảnh và trả về kết quả object detection
- ✅ **Model chỉ load 1 lần và dùng cho tất cả detection** (tuân thủ quy định)

## 🏗️ Kiến trúc đơn giản

```
src/
├── app.py              # Ứng dụng Flask chính
├── config.py           # Cấu hình model và thiết lập
├── models/
│   └── yolo_detector.py # Logic nhận diện YOLO
├── utils/
│   └── file_handler.py  # Xử lý file upload/download
└── templates/
    └── index.html       # Giao diện web (HTML + CSS + JS)
```

## Cài đặt và chạy

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
```## 📱 Cách sử dụng

1. **Khởi động**: Chạy `conda activate yolo && python src/app.py`
2. **Truy cập**: Mở http://localhost:5000 
3. **Chọn model**: YOLOv8n (nhanh) → YOLOv8l (chính xác)
4. **Upload ảnh**: Drag-drop hoặc click chọn file
5. **Xem kết quả**: Bounding boxes + confidence scores

## ⚠️ Quan trọng - Tuân thủ Requirement 1
- ✅ Model YOLOv8n được load sẵn lúc khởi động
- ✅ Model được **reuse** cho tất cả detection (không reload)
- ✅ Chỉ reload khi user chọn model khác (theo quy định)
- ✅ Web interface hoàn chỉnh cho upload ảnh và hiển thị kết quả

## 💾 Dependencies
```bash
pip install ultralytics torch flask opencv-python
```

## 🔧 File cấu hình chính
- `src/config.py`: Danh sách available models
- `src/models/yolo_detector.py`: Logic model reuse  
- `src/app.py`: Web interface và API endpoints