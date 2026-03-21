# Quick Start

Hướng dẫn ngắn để chạy [crawl_data.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/Crawling%20Data/crawl_data.ipynb) và tạo dataset.

Nếu bạn cần giải thích đầy đủ, đọc thêm [README.md](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/Crawling%20Data/README.md).

## Yêu cầu trước khi chạy

Trước khi mở notebook, bạn cần có:

- Python 3
- Jupyter Notebook hoặc VS Code hỗ trợ chạy `.ipynb`
- kết nối Internet ổn định

Notebook này crawl web và xử lý HTML, nên cần cài đủ các thư viện sau:

- `requests`
- `tqdm`
- `beautifulsoup4`
- `lxml`
- `pyarrow`
- `aiohttp`
- `newspaper3k`
- `trafilatura`
- `readability-lxml`
- `goose3`

## Cài thư viện

Bạn có thể cài tất cả một lần bằng lệnh:

```bash
pip install requests tqdm beautifulsoup4 lxml pyarrow aiohttp newspaper3k trafilatura readability-lxml goose3
```

Nếu dùng môi trường ảo, hãy kích hoạt môi trường đó trước khi cài.

Sau khi cài xong:

1. mở notebook
2. chạy cell kiểm tra thư viện đầu tiên
3. nếu không còn báo thiếu thư viện thì tiếp tục chạy các cell còn lại

## Nếu notebook vẫn báo thiếu thư viện

Hãy kiểm tra:

- bạn đã cài đúng Python environment mà notebook đang dùng chưa
- kernel của notebook có đúng môi trường vừa cài thư viện không
- bạn đã restart kernel sau khi cài chưa

Nếu cần, cài lại đúng ngay trong môi trường notebook rồi restart kernel.

## Mục tiêu

Notebook này sẽ:

1. Crawl danh sách URL bài viết từ VietNamNet cho 19 chuyên mục.
2. Crawl `title` và `content` của từng bài.
3. Lưu dữ liệu vào thư mục [`Dataset`](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/Dataset).

## File đầu ra

Sau khi chạy xong, bạn sẽ có:

- `Dataset/data_URLs.json`
- `Dataset/<category>.parquet`
- `Dataset/data_URLs_empty_title.json`
- `Dataset/data_URLs_empty_content.json`

## Cách chạy nhanh

1. Mở notebook [crawl_data.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/Crawling%20Data/crawl_data.ipynb).
2. Chạy toàn bộ notebook từ trên xuống dưới.
3. Chờ notebook hoàn tất crawl URL và crawl content.
4. Xem bảng thống kê ở cell cuối để kiểm tra chất lượng dữ liệu.

## Các bước trong notebook

### Section 1

Chuẩn bị:

- kiểm tra thư viện
- tạo thư mục `Dataset/` nếu chưa có
- tạo hoặc sửa `data_URLs.json`

### Section 2

Crawl URL bài viết:

- chỉ crawl category chưa có URL trong `data_URLs.json`
- category đã có URL sẽ được giữ nguyên

### Section 3

Crawl nội dung bài viết:

- chỉ crawl category chưa có file `.parquet`
- nếu chưa có URL thì tự crawl URL trước
- lưu dữ liệu theo batch xuống parquet

### Section 4

Kiểm tra chất lượng dữ liệu:

- tổng số bài mỗi category
- số bài thiếu `title`
- số bài thiếu `content`
- số bài thiếu cả hai

## Cấu hình quan trọng

Trong notebook có 3 biến đáng chú ý:

```python
MAX_PAGES   = 500
BATCH_SIZE  = 100
NUM_WORKERS = 32
```

Ý nghĩa:

- `MAX_PAGES`: số trang tối đa dò URL cho mỗi chuyên mục
- `BATCH_SIZE`: số bài ghi parquet mỗi lần
- `NUM_WORKERS`: số request crawl content chạy đồng thời

Nếu máy hoặc mạng yếu, có thể giảm:

```python
NUM_WORKERS = 8
```

hoặc:

```python
NUM_WORKERS = 16
```

## Khi nào notebook bỏ qua bước đã làm

Notebook không crawl lại mọi thứ nếu dữ liệu đã tồn tại.

### URL

Nếu `data_URLs.json` đã có URL cho category đó, notebook sẽ bỏ qua bước crawl URL.

### Content

Nếu `Dataset/<category>.parquet` đã tồn tại, notebook sẽ bỏ qua bước crawl content của category đó.

## Khi nào cần xóa file cũ

### Muốn crawl lại URL của 1 category

Sửa hoặc xóa key đó trong:

- `Dataset/data_URLs.json`

### Muốn crawl lại content của 1 category

Xóa file:

- `Dataset/<category>.parquet`

### Muốn crawl lại toàn bộ

Xóa:

- `Dataset/data_URLs.json`
- toàn bộ `Dataset/*.parquet`

Rồi chạy lại notebook.

## Cách kiểm tra đã crawl thành công chưa

Bạn nên kiểm tra 3 thứ:

1. Trong `Dataset/` đã có đủ các file `.parquet`.
2. Có file `data_URLs.json`.
3. Cell cuối in bảng thống kê mà số bài thiếu `title/content` không quá bất thường.

## Dữ liệu này được dùng ở đâu

Dataset được tạo ở đây sẽ được dùng tiếp bởi:

- [SVM/main_SVM.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/SVM/main_SVM.ipynb)
- [PhoBERT/main_PhoBERT.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/PhoBERT/main_PhoBERT.ipynb)

## Khuyến nghị thực tế

- Lần đầu: chạy toàn bộ notebook.
- Lần sau: cứ chạy lại toàn bộ, notebook sẽ tự bỏ qua phần đã có.
- Nếu thấy nhiều bài thiếu `content`, kiểm tra lại các file debug JSON.
- Không nên tăng `NUM_WORKERS` quá cao nếu mạng không ổn định.

## Tóm tắt 30 giây

- Mở notebook.
- Chạy all.
- Chờ tạo `Dataset/data_URLs.json` và các file `Dataset/*.parquet`.
- Xem cell cuối để kiểm tra chất lượng.
- Nếu cần crawl lại, xóa đúng file tương ứng rồi chạy lại.
