# Crawling Data

Tài liệu này mô tả notebook crawl dữ liệu của project. Mục tiêu là tạo dataset bài báo VietNamNet theo từng chủ đề để dùng cho SVM và PhoBERT.

Nếu chỉ cần bắt đầu nhanh, đọc [QUICK_START.md](./QUICK_START.md).

## Mục tiêu của notebook

Notebook `crawl_data.ipynb` dùng để:

- crawl danh sách URL bài báo từ các chuyên mục
- lấy nội dung chính của từng bài
- chuẩn hóa dữ liệu đầu ra
- lưu dữ liệu vào thư mục `Dataset/`

## Luồng hoạt động tổng quát

1. Khai báo danh sách chuyên mục cần crawl.
2. Tạo URL trang danh sách bài viết của từng chuyên mục.
3. Lấy danh sách link bài viết.
4. Truy cập từng bài viết để lấy tiêu đề và nội dung.
5. Làm sạch dữ liệu đầu ra.
6. Loại bản ghi lỗi hoặc trùng.
7. Lưu thành file trong `Dataset/`.

## Những gì notebook thường xử lý

- loại URL trùng
- bỏ bài viết không có nội dung hợp lệ
- chuẩn hóa khoảng trắng
- cắt bớt lỗi HTML, script và thẻ không cần thiết
- gắn nhãn chủ đề theo nguồn crawl

## Khi nào nên chạy lại notebook crawl

- khi muốn mở rộng số lượng dữ liệu
- khi muốn bổ sung các chủ đề còn ít mẫu
- khi cấu trúc trang nguồn thay đổi
- khi muốn làm lại dataset từ đầu

## Thư viện cần cài

```bash
pip install requests beautifulsoup4 pandas pyarrow tqdm lxml
```

## Cách dùng

1. Mở [crawl_data.ipynb](./crawl_data.ipynb).
2. Chạy các cell cấu hình.
3. Chạy toàn bộ notebook hoặc chạy theo từng section.
4. Kiểm tra thư mục `Dataset/` sau khi hoàn tất.

## Sau bước crawl

Bạn có thể chuyển sang:

- [SVM/README.md](../SVM/README.md)
- [PhoBERT/README.md](../PhoBERT/README.md)
