# SVM App

Tài liệu này mô tả chi tiết ứng dụng giao diện của nhánh SVM.

Thư mục này chứa:

- [app_SVM.py](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/SVM/app/app_SVM.py)  
  Ứng dụng Streamlit để phân loại bài báo.
- [app_SVM.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/SVM/app/app_SVM.ipynb)  
  Notebook launcher để kiểm tra môi trường và khởi động app.

Nếu bạn chỉ muốn chạy nhanh app, đọc [QUICK_START.md](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/SVM/app/QUICK_START.md).

## Mục tiêu của app

App này cho phép người dùng:

1. dán URL bài báo VietNamNet để hệ thống tự scrape và phân loại
2. dán tiêu đề + nội dung thủ công nếu URL không lấy được
3. xử lý nhiều URL cùng lúc
4. xem lịch sử phân loại trong phiên làm việc
5. tải kết quả ra CSV

## App dùng mô hình nào

App SVM dùng pipeline đã export từ notebook train:

- `SVM/model/inference_pipeline.pkl`

Pipeline này chứa:

- `vectorizer`
- `model`
- `stopwords`
- `classes`
- `config`

Model hiện tại của nhánh SVM là:

- `SVC`
- kết hợp với `TF-IDF`

Điều này có nghĩa:

- dữ liệu được vector hóa bằng TF-IDF
- app lấy `decision_function()` từ SVM rồi suy ra phân phối điểm tin cậy bằng softmax
- phần trăm hiển thị trong app là điểm tin cậy tương đối, không phải xác suất calibrated thật

## Điều kiện để app chạy được

Trước khi chạy app, bạn cần có:

- Python 3
- các thư viện cần thiết cho Streamlit app
- file `SVM/model/inference_pipeline.pkl`

Nếu chưa có pipeline, hãy chạy notebook train:

- [main_SVM.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/SVM/main_SVM.ipynb)

## Thư viện cần cài

App cần các thư viện sau:

- `streamlit`
- `requests`
- `beautifulsoup4`
- `pyvi`
- `numpy`
- `pandas`
- `scikit-learn`

Cài nhanh bằng:

```bash
pip install streamlit requests beautifulsoup4 pyvi numpy pandas scikit-learn
```

## App kiểm tra gì khi khởi động bằng notebook

Trong [app_SVM.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/SVM/app/app_SVM.ipynb), notebook launcher thường kiểm tra:

1. thư viện đã có đủ chưa
2. file app có tồn tại không
3. file pipeline có tồn tại không
4. Streamlit có khởi động được không

Nếu notebook báo thiếu pipeline, nghĩa là bạn chưa export model từ notebook train.

## Luồng hoạt động của app

Luồng chính của app như sau:

1. load `inference_pipeline.pkl` bằng `st.cache_resource`
2. người dùng chọn một trong ba cách nhập:
   - URL
   - text thủ công
   - batch URL
3. nếu là URL, app sẽ scrape tiêu đề và nội dung bài báo
4. app tiền xử lý văn bản đúng theo pipeline train SVM
5. app biến đổi văn bản sang vector TF-IDF
6. model SVM sinh `decision score` cho toàn bộ class
7. app đổi `decision score` sang phân phối điểm tin cậy bằng softmax
8. app hiển thị:
   - chủ đề dự đoán
   - điểm tin cậy top 1
   - top 5 chủ đề
   - các từ khóa quyết định nếu model có `coef_`
   - preview nội dung
9. app lưu lịch sử vào session hiện tại

## Tiền xử lý trong app

App SVM phải khớp với notebook train. Logic hiện tại là:

1. ghép `title + title + content` để tăng trọng số tiêu đề
2. chuyển về lowercase
3. bỏ dấu câu
4. bỏ số
5. tách từ bằng `ViTokenizer`
6. loại stopwords
7. chuẩn hóa khoảng trắng

Điểm này rất quan trọng:

- title được tăng trọng số trước content để mô hình chú ý nhiều hơn tới tiêu đề
- app phải giữ đúng logic này để khớp với dữ liệu huấn luyện

## App tính điểm tin cậy như thế nào

App SVM dùng:

- `pipeline["model"].decision_function(...)`

Sau đó:

1. lấy vector score của tất cả class
2. áp softmax để đổi thành phân phối điểm tin cậy tương đối
3. chọn class có score cao nhất làm dự đoán cuối cùng

Nghĩa là:

- model vẫn phân loại bình thường dù `probability=False`
- phần trăm hiển thị trong app là score suy ra từ `decision_function`
- không nên hiểu đây là xác suất calibrated thật như `predict_proba()`

Ngoài ra app còn hiển thị:

- `Top 5 chủ đề có thể`
- khoảng cách giữa top 1 và top 2 để cảnh báo trường hợp dễ nhầm

Nếu chênh lệch top 1 và top 2 thấp, app sẽ hiển thị cảnh báo để người dùng biết bài đó chưa tách lớp rõ.

## Top từ khóa là gì

App có thêm một phần giải thích dự đoán bằng các từ khóa mạnh nhất của class.

Phần này chỉ có khi:

- model SVM có thuộc tính `coef_`
- và có thể truy xuất trọng số tuyến tính theo từng class

Với cấu hình `SVC` hiện tại:

- một số model sẽ không có `coef_`
- khi đó app sẽ ẩn phần từ khóa và chỉ hiển thị thông báo tương ứng

## Các tab trong giao diện

App có 4 tab chính.

### 1. `🔗 Nhập URL`

Người dùng dán URL bài báo VietNamNet.

App sẽ:

1. tải HTML
2. lấy tiêu đề
3. lấy phần nội dung chính
4. chạy mô hình SVM
5. hiển thị kết quả

### 2. `📝 Nhập text`

Dùng khi:

- URL không scrape được
- bài báo nằm sau paywall
- cần kiểm tra một đoạn text bất kỳ

Người dùng có thể nhập:

- chỉ tiêu đề
- chỉ nội dung
- hoặc cả hai

### 3. `📋 Batch URL`

Cho phép dán nhiều URL, mỗi dòng một URL.

App sẽ:

- xử lý tuần tự từng URL
- hiển thị tiến độ
- gom kết quả thành bảng
- cho tải CSV

### 4. `📜 Lịch sử`

Lưu lại các lần phân loại trong session hiện tại.

Người dùng có thể:

- xem bảng lịch sử
- tải CSV
- xóa lịch sử

## Cách app scrape nội dung

App dùng `requests` + `BeautifulSoup`.

Quy trình:

1. gửi request với header trình duyệt giả lập
2. bỏ các thẻ không cần thiết như `script`, `style`, `nav`, `footer`
3. thử nhiều selector để lấy `title`
4. thử nhiều selector để lấy `content`
5. nếu không tìm được khối chính, fallback sang gom tất cả thẻ `p`

Vì vậy:

- app hoạt động tốt với đa số bài VietNamNet chuẩn
- nhưng vẫn có thể lỗi nếu cấu trúc trang thay đổi

## File đầu ra và dữ liệu phiên

App không ghi lại database cố định.

Những gì app tạo trong lúc chạy:

- lịch sử trong `st.session_state`
- file CSV khi người dùng bấm tải

Lịch sử sẽ mất khi:

- restart app
- refresh session
- bấm xóa lịch sử

## Cách chạy app

### Cách 1: chạy bằng notebook launcher

Mở:

- [app_SVM.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/SVM/app/app_SVM.ipynb)

Sau đó:

1. chạy cell kiểm tra môi trường
2. chạy cell kiểm tra file app
3. chạy cell khởi động Streamlit

### Cách 2: chạy bằng terminal

Từ thư mục `SVM/`, chạy:

```bash
streamlit run app/app_SVM.py
```

Sau khi chạy thành công, app thường mở ở:

```text
http://localhost:8501
```

## Khi nào cần restart app

Bạn nên restart app khi:

- vừa train lại model SVM
- vừa export pipeline mới
- vừa sửa `app_SVM.py`

Lý do là `st.cache_resource` sẽ giữ model đã load trong session hiện tại.

## Lỗi thường gặp

### 1. Không tìm thấy `inference_pipeline.pkl`

Nguyên nhân:

- chưa train hoặc chưa export model

Cách xử lý:

- chạy [main_SVM.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/SVM/main_SVM.ipynb)
- đảm bảo đã sinh `SVM/model/inference_pipeline.pkl`

### 2. Thiếu thư viện

Nguyên nhân:

- chưa cài đủ package
- notebook đang dùng nhầm Python environment

Cách xử lý:

- cài lại các thư viện trong đúng môi trường
- restart kernel rồi chạy lại

### 3. URL scrape thất bại

Nguyên nhân:

- bài báo không đúng cấu trúc
- mạng lỗi
- website chặn request

Cách xử lý:

- chuyển sang tab `📝 Nhập text`
- dán tiêu đề và nội dung thủ công

## Tóm tắt nhanh

Nhánh app này dành cho trường hợp bạn muốn:

- dùng mô hình SVM nhẹ hơn PhoBERT
- có top 5 class và điểm tin cậy tương đối
- xem từ khóa quan trọng nếu model hỗ trợ `coef_`
- chạy app nhanh trên CPU
