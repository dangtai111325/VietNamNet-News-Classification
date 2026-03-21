# Quick Start

Hướng dẫn ngắn để chạy app PhoBERT trong:

- [app_PhoBERT.py](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/PhoBERT/app/app_PhoBERT.py)
- [app_PhoBERT.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/PhoBERT/app/app_PhoBERT.ipynb)

Nếu cần giải thích đầy đủ, đọc thêm [README.md](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/PhoBERT/app/README.md).

## Yêu cầu trước khi chạy

Bạn cần có:

- Python 3
- Streamlit
- model đã được train và export vào `PhoBERT/model/`

App cần các thư viện:

- `streamlit`
- `requests`
- `beautifulsoup4`
- `pyvi`
- `torch`
- `transformers`
- `numpy`
- `pandas`
- `scipy`

## Cài thư viện

```bash
pip install streamlit requests beautifulsoup4 pyvi torch transformers numpy pandas scipy
```

Sau khi cài xong:

1. chọn đúng Python environment
2. restart kernel nếu chạy bằng notebook

## Model cần có

Trước khi chạy app, thư mục `PhoBERT/model/` nên có ít nhất:

- `model.safetensors`
- `config.json`
- `label_config.json`
- `tokenizer_config.json`

Nếu có thêm:

- `thresholds.json`

thì app sẽ bật threshold calibration.

Nếu chưa có model, hãy chạy:

- [main_PhoBERT.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/PhoBERT/main_PhoBERT.ipynb)

## Cách chạy nhanh nhất

### Cách 1: dùng notebook launcher

Mở:

- [app_PhoBERT.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/PhoBERT/app/app_PhoBERT.ipynb)

Sau đó:

1. chạy cell kiểm tra môi trường
2. chạy cell kiểm tra file app
3. chạy cell khởi động

App sẽ mở ở:

```text
http://localhost:8501
```

### Cách 2: dùng terminal

Từ thư mục `PhoBERT/`, chạy:

```bash
streamlit run app/app_PhoBERT.py
```

## App làm được gì

App có 4 tab chính:

- `🔗 Nhập URL`
  Dán URL bài báo VietNamNet để app tự tải và phân loại.
- `📝 Nhập text`
  Dán tiêu đề và nội dung thủ công.
- `📋 Batch URL`
  Dán nhiều URL để xử lý hàng loạt và tải CSV.
- `📜 Lịch sử`
  Xem và tải lịch sử phân loại trong session.

## Cách dùng rất nhanh

### Phân loại bằng URL

1. mở tab `🔗 Nhập URL`
2. dán URL bài báo
3. bấm `Phân loại`

### Phân loại bằng text thủ công

1. mở tab `📝 Nhập text`
2. nhập tiêu đề hoặc nội dung
3. bấm `Phân loại`

### Phân loại nhiều URL

1. mở tab `📋 Batch URL`
2. dán mỗi URL trên một dòng
3. bấm `Phân loại tất cả`
4. tải CSV nếu cần

## Nếu app không chạy

Kiểm tra lần lượt:

1. đã cài đủ thư viện chưa
2. đã có model trong `PhoBERT/model/` chưa
3. notebook / terminal có đang dùng đúng Python environment không

## Khi nào cần restart app

Bạn nên restart app khi:

- vừa train model mới
- vừa sửa `app_PhoBERT.py`
- vừa thêm hoặc thay `thresholds.json`

## Tóm tắt 30 giây

- cài thư viện
- đảm bảo có model trong `PhoBERT/model/`
- chạy `streamlit run app/app_PhoBERT.py`
- mở `http://localhost:8501`
- dán URL hoặc nội dung bài báo để phân loại
