# Quick Start

Hướng dẫn ngắn để chạy app kết hợp.

## Điều kiện

Bạn cần có đủ cả hai phần:

### SVM

- `SVM/model/inference_pipeline.pkl`

### PhoBERT

- `PhoBERT/model/model.safetensors`
- `PhoBERT/model/config.json`
- `PhoBERT/model/label_config.json`
- `PhoBERT/model/tokenizer_config.json`

Nếu có thêm `PhoBERT/model/thresholds.json`, app sẽ bật hiệu chỉnh ngưỡng cho nhánh PhoBERT.

## Cài thư viện

```bash
pip install streamlit requests beautifulsoup4 pyvi numpy pandas scikit-learn torch transformers scipy
```

## Chạy app

Từ thư mục `Combined_Model_App/`, chạy:

```bash
streamlit run app_combined.py
```

Sau đó mở:

```text
http://localhost:8501
```

## App làm được gì

- phân loại bằng URL
- phân loại bằng text thủ công
- batch nhiều URL
- hiển thị kết quả kết hợp, SVM riêng và PhoBERT riêng
- xem nội dung bài báo trong khung có thể cuộn và copy

## Nếu app không chạy

Kiểm tra:

1. đã có `SVM/model/inference_pipeline.pkl` chưa
2. đã có đủ file trong `PhoBERT/model/` chưa
3. đã cài đúng thư viện chưa
4. terminal hoặc notebook có đang dùng đúng Python environment không
