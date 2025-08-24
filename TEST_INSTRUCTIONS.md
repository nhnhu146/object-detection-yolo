
# 🧪 HƯỚNG DẪN TEST SAU KHI SỬA LỖI

## 1. Cài đặt Dependencies
```bash
pip install ultralytics torch torchvision opencv-python flask flask-cors numpy pillow
```

## 2. Chạy Application
```bash
cd "d:\Github\object-detection-yolo"
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
