# Quick Start

Hướng dẫn ngắn để chạy app SVM trong:

- [app_SVM.py](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/SVM/app/app_SVM.py)
- [app_SVM.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/SVM/app/app_SVM.ipynb)

Nếu cần giải thích đầy đủ, đọc thêm [README.md](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/SVM/app/README.md).

## Yêu cầu trước khi chạy

Bạn cần có:

- Python 3
- Streamlit
- pipeline SVM đã được export vào `SVM/model/`

App cần các thư viện:

- `streamlit`
- `requests`
- `beautifulsoup4`
- `pyvi`
- `numpy`
- `pandas`
- `scikit-learn`

## Cài thư viện

```bash
pip install streamlit requests beautifulsoup4 pyvi numpy pandas scikit-learn
```

Sau khi cài xong:

1. chọn đúng Python environment
2. restart kernel nếu chạy bằng notebook

## Pipeline cần có

Trước khi chạy app, thư mục `SVM/model/` nên có:

- `inference_pipeline.pkl`

Nếu chưa có, hãy chạy:

- [main_SVM.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/SVM/main_SVM.ipynb)

## Cách chạy nhanh nhất

### Cách 1: dùng notebook launcher

Mở:

- [app_SVM.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/SVM/app/app_SVM.ipynb)

Sau đó:

1. chạy cell kiểm tra môi trường
2. chạy cell kiểm tra file app
3. chạy cell khởi động

App thường mở ở:

```text
http://localhost:8501
```

### Cách 2: dùng terminal

Từ thư mục `SVM/`, chạy:

```bash
streamlit run app/app_SVM.py
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

Lưu ý:

- app SVM hiện hiển thị điểm tin cậy suy ra từ `decision_function`
- đây không phải xác suất calibrated thật của model

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
2. đã có `SVM/model/inference_pipeline.pkl` chưa
3. notebook hoặc terminal có đang dùng đúng Python environment không

## Khi nào cần restart app

Bạn nên restart app khi:

- vừa train lại model
- vừa export pipeline mới
- vừa sửa `app_SVM.py`

## Tóm tắt 30 giây

- cài thư viện
- đảm bảo có `SVM/model/inference_pipeline.pkl`
- chạy `streamlit run app/app_SVM.py`
- mở `http://localhost:8501`
- dán URL hoặc nội dung bài báo để phân loại
