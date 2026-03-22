# Combined Model App

Tài liệu này mô tả ứng dụng kết hợp SVM và PhoBERT.

## File chính

- [app_combined.py](/c:/Users/DELL/Downloads/HTTM/VietNamNet News Classification/Combined_Model_App/app_combined.py)
- [app_combined.ipynb](/c:/Users/DELL/Downloads/HTTM/VietNamNet News Classification/Combined_Model_App/app_combined.ipynb)

Trong đó:

- `app_combined.py` là app Streamlit chính
- `app_combined.ipynb` chỉ là notebook launcher / checker

## App dùng những model nào

### Nhánh SVM

App load:

- `SVM/model/inference_pipeline.pkl`

### Nhánh PhoBERT

App load các file trong `PhoBERT/model/`:

- `model.safetensors`
- `config.json`
- `label_config.json`
- `tokenizer_config.json`
- nếu có: `thresholds.json`

## Cách app kết hợp hai nhánh

Luồng chính:

1. load pipeline SVM
2. load model và tokenizer PhoBERT
3. nhận đầu vào từ URL hoặc text
4. tiền xử lý riêng cho từng nhánh
5. lấy score từ từng model
6. kết hợp score để ra dự đoán cuối cùng

### Score phía SVM

SVM dùng:

- `decision_function()`

Sau đó app áp softmax để đổi score sang phân phối tương đối.

### Score phía PhoBERT

PhoBERT dùng:

- logits
- temperature scaling
- softmax
- hiệu chỉnh ngưỡng nếu có `thresholds.json`

### Phần kết hợp

Sau khi có score từ 2 nhánh, app kết hợp chúng theo từng class rồi chuẩn hóa lại để chọn class cuối cùng.

## Tiền xử lý trong app

### Nhánh SVM

1. ghép `title + title + content`
2. lowercase
3. bỏ dấu câu
4. bỏ số
5. `ViTokenizer`
6. loại stopwords

### Nhánh PhoBERT

1. ghép `title + content`
2. lowercase
3. bỏ dấu câu
4. bỏ số
5. `ViTokenizer`

## Tính năng chính

App có 4 tab:

- `Nhập URL`
- `Nhập text`
- `Batch URL`
- `Lịch sử`

Người dùng có thể:

- xem kết quả kết hợp
- xem riêng kết quả của SVM
- xem riêng kết quả của PhoBERT
- xử lý nhiều URL cùng lúc
- tải lịch sử hoặc kết quả batch ra CSV
- xem khung nội dung bài báo có thể cuộn và copy

## Giao diện hiện tại

App đã được đồng bộ theo giao diện sáng cùng phong cách với app PhoBERT và app SVM.

## Điều kiện để app chạy được

Bạn cần có:

- Python 3
- các thư viện Streamlit cần thiết
- `SVM/model/inference_pipeline.pkl`
- model export trong `PhoBERT/model/`

Nếu thiếu một trong hai nhánh, app sẽ không chạy đầy đủ.

## Cài thư viện

```bash
pip install streamlit requests beautifulsoup4 pyvi numpy pandas scikit-learn torch transformers scipy
```

## Cách chạy

Từ thư mục `Combined_Model_App/`, chạy:

```bash
streamlit run app_combined.py
```

## Khi nào cần restart app

- vừa train lại SVM
- vừa train lại PhoBERT
- vừa export artifact mới
- vừa sửa `app_combined.py`

Lý do là cả hai model đều được cache bằng `st.cache_resource`.
