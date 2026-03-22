# PhoBERT App

Tài liệu này mô tả app giao diện của nhánh PhoBERT.

## File chính

- [app_PhoBERT.py](/c:/Users/DELL/Downloads/HTTM/VietNamNet News Classification/PhoBERT/app/app_PhoBERT.py)
- [app_PhoBERT.ipynb](/c:/Users/DELL/Downloads/HTTM/VietNamNet News Classification/PhoBERT/app/app_PhoBERT.ipynb)

Trong đó:

- `app_PhoBERT.py` là app Streamlit chính
- `app_PhoBERT.ipynb` chỉ là notebook launcher / checker

## App dùng model nào

App load các file trong `PhoBERT/model/`:

- `model.safetensors`
- `config.json`
- `label_config.json`
- `tokenizer_config.json`
- nếu có: `thresholds.json`

## Cách app suy luận

Luồng chính:

1. tiền xử lý văn bản giống notebook train
2. encode theo chiến lược head-tail
3. model sinh `logits`
4. áp dụng temperature scaling
5. softmax để ra xác suất
6. nếu có `thresholds.json`, áp dụng thêm hiệu chỉnh ngưỡng
7. hiển thị top class và top 5 chủ đề

## Ý nghĩa các score hiển thị

App có thể hiển thị hai loại tín hiệu:

- xác suất softmax sau temperature scaling
- điểm quyết định sau hiệu chỉnh ngưỡng

Điều này giúp phân biệt:

- xác suất nền của model
- quyết định cuối cùng sau khi điều chỉnh threshold theo class

## Tiền xử lý trong app

App dùng cùng logic với notebook PhoBERT:

1. ghép `title + content`
2. lowercase
3. bỏ dấu câu
4. bỏ số
5. `ViTokenizer`
6. chuẩn hóa khoảng trắng

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

## Giao diện hiện tại

App đã được đồng bộ theo giao diện sáng:

- nền sáng
- tab sáng
- form nhập sáng
- badge màu theo mức tin cậy

## Điều kiện để app chạy được

Bạn cần có:

- Python 3
- các thư viện Streamlit cần thiết
- model export trong `PhoBERT/model/`

## Cài thư viện

```bash
pip install streamlit requests beautifulsoup4 pyvi torch transformers numpy pandas scipy
```

## Cách chạy

Từ thư mục `PhoBERT/`, chạy:

```bash
streamlit run app/app_PhoBERT.py
```

## Khi nào cần restart app

- vừa train lại model
- vừa cập nhật `label_config.json`
- vừa cập nhật `thresholds.json`
- vừa sửa `app_PhoBERT.py`

Lý do là model được cache bằng `st.cache_resource`.
