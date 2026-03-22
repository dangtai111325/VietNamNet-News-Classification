# SVM App

Tài liệu này mô tả app giao diện của nhánh SVM.

## File chính

- [app_SVM.py](/c:/Users/DELL/Downloads/HTTM/VietNamNet News Classification/SVM/app/app_SVM.py)
- [app_SVM.ipynb](/c:/Users/DELL/Downloads/HTTM/VietNamNet News Classification/SVM/app/app_SVM.ipynb)

Trong đó:

- `app_SVM.py` là app Streamlit chính
- `app_SVM.ipynb` chỉ là notebook launcher / checker

## App dùng model nào

App load:

- `SVM/model/inference_pipeline.pkl`

Pipeline này được export từ [main_SVM.ipynb](/c:/Users/DELL/Downloads/HTTM/VietNamNet News Classification/SVM/main_SVM.ipynb).

## Cách app suy luận

Luồng chính:

1. tiền xử lý văn bản giống notebook train
2. vector hóa bằng TF-IDF
3. lấy `decision_function()` từ `LinearSVC`
4. đổi score sang phân phối tương đối bằng softmax
5. hiển thị chủ đề dự đoán và top 5 chủ đề

Lưu ý:

- phần trăm trong app là điểm tin cậy suy ra từ `decision_function()`
- đây không phải `predict_proba()` calibrated thật

## Tiền xử lý trong app

App dùng cùng logic với notebook SVM:

1. ghép `title + title + content`
2. lowercase
3. bỏ dấu câu
4. bỏ số
5. `ViTokenizer`
6. loại stopwords
7. chuẩn hóa khoảng trắng

## Tính năng chính

App có 4 tab:

- `Nhập URL`
- `Nhập text`
- `Batch URL`
- `Lịch sử`

Người dùng có thể:

- dán URL bài báo VietNamNet để app tự scrape rồi phân loại
- nhập tiêu đề và nội dung thủ công
- xử lý nhiều URL cùng lúc
- tải lịch sử hoặc kết quả batch ra CSV
- xem khung nội dung bài báo có thể cuộn và copy

## Điều kiện để app chạy được

Bạn cần có:

- Python 3
- các thư viện Streamlit cần thiết
- file `SVM/model/inference_pipeline.pkl`

Nếu chưa có pipeline, hãy chạy notebook train trước.

## Cài thư viện

```bash
pip install streamlit requests beautifulsoup4 pyvi numpy pandas scikit-learn
```

## Cách chạy

Từ thư mục `SVM/`, chạy:

```bash
streamlit run app/app_SVM.py
```

## Khi nào cần restart app

- vừa train lại model SVM
- vừa export pipeline mới
- vừa sửa `app_SVM.py`

Lý do là model được cache bằng `st.cache_resource`.
