# Quick Start

Hướng dẫn ngắn để chạy app combined trong:

- [app_combined.py](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/Combined_Model_App/app_combined.py)
- [app_combined.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/Combined_Model_App/app_combined.ipynb)

Nếu cần giải thích đầy đủ, đọc thêm [README.md](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/Combined_Model_App/README.md).

## Yêu cầu trước khi chạy

Bạn cần có:

- Python 3
- Streamlit
- pipeline SVM đã export
- model PhoBERT đã export

App cần các thư viện:

- `streamlit`
- `requests`
- `beautifulsoup4`
- `pyvi`
- `numpy`
- `pandas`
- `scikit-learn`
- `torch`
- `transformers`
- `scipy`

## Cài thư viện

```bash
pip install streamlit requests beautifulsoup4 pyvi numpy pandas scikit-learn torch transformers scipy
```

Sau khi cài xong:

1. chọn đúng Python environment
2. restart kernel nếu chạy bằng notebook

## Model cần có

Trước khi chạy app, bạn cần đủ cả hai phần:

### Phần SVM

Trong `SVM/model/` nên có:

- `inference_pipeline.pkl`

### Phần PhoBERT

Trong `PhoBERT/model/` nên có ít nhất:

- `model.safetensors`
- `config.json`
- `label_config.json`
- `tokenizer_config.json`

Nếu có thêm:

- `thresholds.json`

thì app sẽ bật threshold calibration cho PhoBERT.

Nếu thiếu model, hãy chạy:

- [main_SVM.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/SVM/main_SVM.ipynb)
- [main_PhoBERT.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/PhoBERT/main_PhoBERT.ipynb)

## Cách chạy nhanh nhất

### Cách 1: dùng notebook launcher

Mở:

- [app_combined.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/Combined_Model_App/app_combined.ipynb)

Sau đó:

1. chạy cell kiểm tra môi trường
2. chạy cell kiểm tra file app
3. chạy cell khởi động

App thường mở ở:

```text
http://localhost:8501
```

### Cách 2: dùng terminal

Từ thư mục `Combined_Model_App/`, chạy:

```bash
streamlit run app_combined.py
```

## App làm được gì

App có 4 tab chính:

- `🔗 Nhập URL`
  Dán URL bài báo VietNamNet để app tự tải, chạy cả SVM lẫn PhoBERT, rồi kết hợp kết quả.
- `📝 Nhập text`
  Dán tiêu đề và nội dung thủ công.
- `📋 Batch URL`
  Dán nhiều URL để xử lý hàng loạt và tải CSV.
- `📜 Lịch sử`
  Xem và tải lịch sử phân loại trong session.

Lưu ý:

- nhánh SVM trong app combined dùng điểm tin cậy suy ra từ `decision_function`
- nhánh PhoBERT dùng score sau `temperature scaling` và có thể có thêm threshold calibration

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
3. đã có đủ file trong `PhoBERT/model/` chưa
4. notebook hoặc terminal có đang dùng đúng Python environment không

## Khi nào cần restart app

Bạn nên restart app khi:

- vừa train lại SVM
- vừa train lại PhoBERT
- vừa export model mới
- vừa sửa `app_combined.py`

## Tóm tắt 30 giây

- cài thư viện
- đảm bảo có model ở cả `SVM/model/` và `PhoBERT/model/`
- chạy `streamlit run app_combined.py`
- mở `http://localhost:8501`
- dán URL hoặc nội dung bài báo để phân loại bằng mô hình kết hợp
