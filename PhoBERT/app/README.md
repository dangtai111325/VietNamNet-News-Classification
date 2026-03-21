# PhoBERT App

Tài liệu này giải thích chi tiết phần ứng dụng giao diện của nhánh PhoBERT.

Thư mục này chứa:

- [app_PhoBERT.py](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/PhoBERT/app/app_PhoBERT.py)
  Ứng dụng Streamlit dùng để phân loại bài báo.
- [app_PhoBERT.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/PhoBERT/app/app_PhoBERT.ipynb)
  Notebook launcher để kiểm tra môi trường và khởi động app.

Nếu bạn chỉ muốn chạy nhanh app, đọc [QUICK_START.md](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/PhoBERT/app/QUICK_START.md).

## Mục tiêu của app

App này cho phép người dùng:

1. dán URL bài báo VietNamNet để hệ thống tự scrape và phân loại
2. dán tiêu đề + nội dung thủ công nếu URL không scrape được
3. xử lý hàng loạt nhiều URL cùng lúc
4. xem lịch sử phân loại trong phiên làm việc
5. tải kết quả ra file CSV

## App dùng mô hình nào

App không hardcode tên mô hình cố định.

Khi khởi động, app sẽ đọc:

- `PhoBERT/model/label_config.json`
- `PhoBERT/model/config.json`
- `PhoBERT/model/model.safetensors`
- nếu có: `PhoBERT/model/thresholds.json`

Nói cách khác:

- model nào đã được export trong notebook train thì app sẽ load model đó
- app hiện đã hỗ trợ cả temperature scaling và threshold calibration nếu file `thresholds.json` tồn tại

## Điều kiện để app chạy được

Trước khi chạy app, bạn cần có:

- Python 3
- các thư viện cần thiết cho Streamlit app
- model đã được train và lưu trong thư mục `PhoBERT/model/`

Nếu chưa có model, app sẽ không chạy được.

Model phải được tạo từ notebook:

- [main_PhoBERT.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/PhoBERT/main_PhoBERT.ipynb)

## Thư viện cần cài

App cần các thư viện sau:

- `streamlit`
- `requests`
- `beautifulsoup4`
- `pyvi`
- `torch`
- `transformers`
- `numpy`
- `pandas`
- `scipy`

Bạn có thể cài nhanh bằng lệnh:

```bash
pip install streamlit requests beautifulsoup4 pyvi torch transformers numpy pandas scipy
```

Lưu ý:

- `torch` nên được cài đúng bản phù hợp với máy của bạn
- app có thể chạy bằng CPU, không bắt buộc GPU
- nhưng nếu dùng GPU thì inference sẽ nhanh hơn

## App kiểm tra những gì khi khởi động bằng notebook

Trong [app_PhoBERT.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/PhoBERT/app/app_PhoBERT.ipynb), cell kiểm tra môi trường sẽ xác nhận:

### 1. Thư viện

- `streamlit`
- `requests`
- `bs4`
- `pyvi`
- `torch`
- `transformers`
- `numpy`
- `pandas`
- `scipy`

### 2. Model files

- `model.safetensors`
- `label_config.json`
- `config.json`
- `tokenizer_config.json`

### 3. Threshold calibration

Nếu có `thresholds.json`, app sẽ bật threshold calibration.

Nếu không có, app vẫn chạy được, nhưng sẽ ra quyết định theo softmax thông thường.

## Luồng hoạt động của app

Luồng chính của app như sau:

1. load model và tokenizer một lần bằng `st.cache_resource`
2. load `label_config.json`
3. nếu có `thresholds.json`, load cả:
   - `temperature`
   - vector `thresholds`
4. người dùng chọn cách nhập dữ liệu:
   - URL
   - text thủ công
   - batch URL
5. app tiền xử lý văn bản
6. app encode theo chiến lược head-tail
7. model sinh `logits`
8. app áp dụng:
   - temperature scaling
   - softmax
   - threshold calibration nếu có
9. app hiển thị:
   - chủ đề dự đoán
   - xác suất softmax
   - top 5 chủ đề
   - preview nội dung
10. app lưu lịch sử vào session

## Tiền xử lý trong app

App cố gắng khớp với pipeline training.

Hiện tại preprocessing là:

1. ghép `title + content`
2. chuyển về lowercase
3. bỏ dấu câu
4. bỏ số
5. `ViTokenizer.tokenize`
6. chuẩn hóa khoảng trắng

Điểm quan trọng:

- app hiện đã được sửa để dùng `title + content`
- không còn nhân đôi title
- điều này cần khớp với notebook train hiện tại

## Head-Tail tokenization là gì

PhoBERT chỉ nhận một số lượng token giới hạn cho mỗi lần suy luận.

App dùng chiến lược:

- lấy 127 token đầu
- lấy 127 token cuối
- thêm token đặc biệt `[CLS]` và `[SEP]`

Tổng cộng:

- `256 tokens`

Nếu bài ngắn hơn giới hạn, app tokenize bình thường với padding.

Nếu bài dài hơn, app giữ cả đầu và cuối thay vì chỉ lấy phần đầu.

## Cách app tính xác suất và ra quyết định

Đây là phần dễ gây hiểu nhầm, nên cần nói rõ.

### Bước 1: model sinh logits

Model PhoBERT trả về `logits` cho 19 class.

### Bước 2: temperature scaling

App chia logits cho `temperature`:

- nếu `temperature > 1`, phân phối xác suất sẽ mềm hơn
- mục tiêu là giảm overconfidence

### Bước 3: softmax

App áp dụng softmax để ra:

- `probs`

Đây là xác suất softmax đã qua temperature scaling.

### Bước 4: threshold calibration

Nếu có `thresholds.json`, app tính:

- `adj_probs = probs / thresholds`

Rồi chọn class có `adj_probs` lớn nhất.

Điều này giúp cân bằng precision/recall giữa các class, đặc biệt với class yếu.

### Hai khái niệm app đang hiển thị

App hiện hiển thị 2 tín hiệu:

- `Xác suất softmax`
  Đây là xác suất calibrated sau temperature scaling.
- `Điểm quyết định sau calibration`
  Đây là tỉ lệ tương đối sau khi đã chia threshold và so sánh giữa các class.

Điều này tốt hơn việc gọi đơn giản là “độ tin cậy”, vì quyết định cuối cùng có thể đã bị threshold calibration làm thay đổi thứ hạng class.

## Các tab trong giao diện

App có 4 tab chính.

## 1. Tab `🔗 Nhập URL`

Người dùng dán URL bài báo VietNamNet.

Luồng xử lý:

1. app tải HTML bài báo
2. parse title
3. parse content
4. nếu lấy được dữ liệu, app phân loại
5. hiển thị kết quả

Nếu URL lỗi hoặc bị chặn:

- app hiển thị lỗi
- gợi ý dùng tab `📝 Nhập text`

## 2. Tab `📝 Nhập text`

Dùng khi:

- URL bị chặn
- trang web không scrape được
- người dùng chỉ có nội dung copy tay

Người dùng nhập:

- tiêu đề
- nội dung

Sau đó app phân loại trực tiếp, không cần đi qua bước scrape URL.

## 3. Tab `📋 Batch URL`

Dùng để xử lý hàng loạt.

Người dùng dán nhiều URL, mỗi dòng một URL.

App sẽ:

1. lần lượt scrape từng URL
2. phân loại từng bài
3. gom kết quả thành bảng
4. cho tải xuống CSV

CSV batch hiện chứa:

- URL
- tiêu đề
- chủ đề dự đoán
- tin cậy
- trạng thái

## 4. Tab `📜 Lịch sử`

Tab này hiển thị lịch sử phân loại trong session hiện tại.

Người dùng có thể:

- xem bảng lịch sử
- tải lịch sử xuống CSV
- xóa lịch sử

Lưu ý:

- lịch sử chỉ sống trong phiên Streamlit hiện tại
- reload app hoặc restart app có thể làm mất lịch sử

## Cách scrape URL trong app

App có một scraper đơn giản bằng:

- `requests`
- `BeautifulSoup`

Nó thử các selector cho:

- `title`
- `content`

So với notebook crawl dữ liệu:

- app scraper nhẹ hơn
- không có 5 lớp fallback phức tạp như notebook crawler
- mục tiêu là đủ tốt để dùng interactive, không phải crawl quy mô lớn

Vì vậy:

- có URL sẽ scrape tốt
- nhưng cũng có bài bị lỗi hoặc bị chặn
- lúc đó tab nhập text là phương án dự phòng

## Giao diện hiện tại

App hiện dùng Streamlit với light theme tùy biến bằng CSS.

Các thay đổi giao diện chính:

- nền sáng
- tab sáng
- input sáng
- nút màu xanh lá
- badge kết quả có màu theo mức tin cậy

Mức tin cậy đang chia 3 mức:

- cao
- trung bình
- thấp

Nếu khoảng cách giữa top 1 và top 2 nhỏ hơn 5%, app hiển thị cảnh báo:

- bài có thể thuộc hai chủ đề gần nhau

## Cách chạy app

Có 2 cách.

## Cách 1: chạy bằng notebook launcher

Mở:

- [app_PhoBERT.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/PhoBERT/app/app_PhoBERT.ipynb)

Rồi:

1. chạy cell kiểm tra
2. chạy cell kiểm tra file app
3. chạy cell khởi động Streamlit

Notebook sẽ mở:

- `http://localhost:8501`

## Cách 2: chạy bằng terminal

Từ thư mục `PhoBERT/`, chạy:

```bash
streamlit run app/app_PhoBERT.py
```

Sau đó mở:

```text
http://localhost:8501
```

## Khi nào app không chạy được

Các lỗi thường gặp:

### 1. Thiếu thư viện

Notebook launcher sẽ báo thiếu package.

Giải pháp:

- cài đúng thư viện
- restart kernel

### 2. Chưa có model

Nếu chưa có:

- `model.safetensors`
- `config.json`
- `label_config.json`

thì app không thể load model.

Giải pháp:

- chạy notebook train và export model trước

### 3. Sai Python environment

Bạn có thể đã cài package ở một môi trường Python khác môi trường notebook hoặc terminal đang dùng.

Giải pháp:

- kiểm tra `sys.executable`
- chọn đúng kernel
- cài lại package vào đúng môi trường

## Cách cập nhật app sau khi train model mới

Nếu bạn train model mới:

1. chạy lại notebook train để export model vào `PhoBERT/model/`
2. restart Streamlit app
3. app sẽ load model mới từ thư mục model

Nếu bạn chỉ sửa code app:

1. lưu lại `app_PhoBERT.py`
2. refresh hoặc restart Streamlit

## Quan hệ giữa app và notebook train

App phụ thuộc vào:

- [main_PhoBERT.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/PhoBERT/main_PhoBERT.ipynb)

Notebook train chịu trách nhiệm tạo:

- model weights
- tokenizer files
- label config
- threshold config

App chỉ là lớp giao diện và suy luận.

## Những file app quan trọng nhất

- [app_PhoBERT.py](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/PhoBERT/app/app_PhoBERT.py)
  logic Streamlit và inference
- [app_PhoBERT.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/PhoBERT/app/app_PhoBERT.ipynb)
  notebook giúp chạy app thuận tiện
- `PhoBERT/model/label_config.json`
  cấu hình class và metadata model
- `PhoBERT/model/thresholds.json`
  thông tin threshold calibration

## Tóm tắt ngắn

- `app_PhoBERT.py` là app Streamlit thật
- `app_PhoBERT.ipynb` chỉ là launcher/checker
- app nhận URL, text thủ công, hoặc batch URL
- app suy luận bằng PhoBERT + head-tail tokenization
- app có hỗ trợ temperature scaling và threshold calibration
- muốn app chạy được thì phải có model trong `PhoBERT/model/`
