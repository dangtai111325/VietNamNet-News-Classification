# Quick Start

Hướng dẫn ngắn để chạy app SVM.

## Điều kiện

Bạn cần có:

- `SVM/model/inference_pipeline.pkl`
- các thư viện Streamlit cần thiết

## Cài thư viện

```bash
pip install streamlit requests beautifulsoup4 pyvi numpy pandas scikit-learn
```

## Chạy app

Từ thư mục `SVM/`, chạy:

```bash
streamlit run app/app_SVM.py
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

1. đã có `SVM/model/inference_pipeline.pkl` chưa
2. đã cài đúng thư viện chưa
3. terminal hoặc notebook có đang dùng đúng Python environment không
