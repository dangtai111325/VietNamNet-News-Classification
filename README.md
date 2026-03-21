# VietNamNet News Classification

Project này phân loại bài báo tiếng Việt của Vietnamnet vào 19 chủ đề.

Repo hiện có 4 phần chính:

- `Crawling Data/`: crawl dữ liệu và tạo dataset `.parquet`
- `SVM/`: pipeline TF-IDF + SVC và app riêng
- `PhoBERT/`: pipeline PhoBERT và app riêng
- `Combined_Model_App/`: app kết hợp SVM + PhoBERT

## Cấu trúc repo

```text
VietNamNet News Classification/
├── Dataset/
├── Crawling Data/
├── SVM/
├── PhoBERT/
├── Combined_Model_App/
├── README.md
└── SETUP.md
```

## 19 chủ đề

- Bạn đọc
- Bảo vệ người tiêu dùng
- Bất động sản
- Chính trị
- Công nghệ
- Dân tộc - Tôn giáo
- Đời sống
- Du lịch
- Giáo dục
- Kinh doanh
- Ô tô - Xe máy
- Pháp luật
- Sức khỏe
- Thế giới
- Thể thao
- Thị trường tiêu dùng
- Thời sự
- Tuần Việt Nam
- Văn hóa - Giải trí

## Các nhánh mô hình

### 1. SVM

Nằm trong [SVM/](./SVM/).

Nhánh này dùng:

- `TF-IDF`
- `SVC`

Phù hợp khi cần:

- mô hình nhẹ hơn
- train và inference nhanh hơn
- giải thích quyết định bằng từ khóa

Xem thêm:

- [SVM/README.md](./SVM/README.md)
- [SVM/QUICK_START.md](./SVM/QUICK_START.md)

### 2. PhoBERT

Nằm trong [PhoBERT/](./PhoBERT/).

Nhánh này dùng:

- `vinai/phobert-base-v2`
- head-tail tokenization
- weighted loss
- threshold calibration

Phù hợp khi cần:

- độ chính xác cao hơn SVM
- xác suất class từ transformer
- xử lý tốt hơn các lớp khó hoặc ít mẫu

Xem thêm:

- [PhoBERT/README.md](./PhoBERT/README.md)
- [PhoBERT/QUICK_START.md](./PhoBERT/QUICK_START.md)

### 3. App kết hợp

Nằm trong [Combined_Model_App/](./Combined_Model_App/).

App này:

- load cả SVM và PhoBERT
- lấy xác suất từ từng model
- kết hợp theo xác suất có điều kiện

Xem thêm:

- [Combined_Model_App/README.md](./Combined_Model_App/README.md)
- [Combined_Model_App/QUICK_START.md](./Combined_Model_App/QUICK_START.md)

## Bắt đầu nhanh

### Cách 1: theo thứ tự đầy đủ

1. crawl dữ liệu bằng [Crawling Data/crawl_data.ipynb](./Crawling%20Data/crawl_data.ipynb)
2. train một trong hai nhánh:
   - [SVM/main_SVM.ipynb](./SVM/main_SVM.ipynb)
   - [PhoBERT/main_PhoBERT.ipynb](./PhoBERT/main_PhoBERT.ipynb)
3. chạy app tương ứng:
   - `streamlit run SVM/app/app_SVM.py`
   - `streamlit run PhoBERT/app/app_PhoBERT.py`
   - `streamlit run Combined_Model_App/app_combined.py`

### Cách 2: nếu đã có model

Bạn chỉ cần chạy app.

Ví dụ:

```bash
streamlit run SVM/app/app_SVM.py
streamlit run PhoBERT/app/app_PhoBERT.py
streamlit run Combined_Model_App/app_combined.py
```

## Tài liệu theo thư mục

- crawler: [Crawling Data/README.md](./Crawling%20Data/README.md)
- SVM: [SVM/README.md](./SVM/README.md)
- PhoBERT: [PhoBERT/README.md](./PhoBERT/README.md)
- app combine: [Combined_Model_App/README.md](./Combined_Model_App/README.md)
- hướng dẫn setup chung: [SETUP.md](./SETUP.md)

## Ghi chú

- tài liệu trong từng thư mục được viết theo logic của chính thư mục đó
- nếu một folder thay đổi đường dẫn chạy, artifact, preprocessing hoặc model, README và QUICK_START của folder đó cần được cập nhật theo
