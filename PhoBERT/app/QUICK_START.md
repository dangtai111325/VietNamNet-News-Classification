# Quick Start

Hướng dẫn ngắn để chạy app PhoBERT.

## Điều kiện

Bạn cần có model export trong `PhoBERT/model/`, tối thiểu gồm:

- `model.safetensors`
- `config.json`
- `label_config.json`
- `tokenizer_config.json`

Nếu có thêm `thresholds.json`, app sẽ bật hiệu chỉnh ngưỡng.

## Cài thư viện

```bash
pip install streamlit requests beautifulsoup4 pyvi torch transformers numpy pandas scipy
```

## Chạy app

Từ thư mục `PhoBERT/`, chạy:

```bash
streamlit run app/app_PhoBERT.py
```

Sau đó mở:

```text
http://localhost:8501
```

## App làm được gì

- phân loại bằng URL
- phân loại bằng text thủ công
- batch nhiều URL
- lưu lịch sử trong phiên
- xem nội dung bài báo trong khung có thể cuộn và copy

## Nếu app không chạy

Kiểm tra:

1. đã có đủ file trong `PhoBERT/model/` chưa
2. đã cài đúng thư viện chưa
3. terminal hoặc notebook có đang dùng đúng Python environment không
