# VietNamNet News Classification

Project này phân loại bài báo tiếng Việt của VietNamNet vào 19 chủ đề bằng 3 hướng chính:

- `SVM/`: TF-IDF + `LinearSVC`
- `PhoBERT/`: `vinai/phobert-base-v2`
- `Combined_Model_App/`: app kết hợp điểm từ SVM và PhoBERT

Ngoài ra repo còn có:

- `Crawling Data/`: notebook crawl dữ liệu và tạo dataset
- `LR/`: notebook thử nghiệm Logistic Regression để so sánh nhanh với SVM

## Tải Full Project

Do GitHub không đẩy kèm toàn bộ dataset, cache và artifact model lớn, bạn có thể tải bản đầy đủ tại:

- Google Drive: <https://drive.google.com/drive/folders/1gW393KCdnYU4TDWjDZZBluvZT9aqSLIp?usp=drive_link>

Phù hợp khi bạn muốn:

- mở project và chạy ngay mà không phải tạo lại toàn bộ cache
- có sẵn model để `Run All` notebook hoặc chạy app
- lấy các thư mục không được đưa lên GitHub

## Cấu trúc repo

```text
VietNamNet News Classification/
├── Dataset/
├── Crawling Data/
├── SVM/
├── PhoBERT/
├── Combined_Model_App/
├── LR/
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

## Tóm tắt từng thư mục

### `Crawling Data/`

Notebook `crawl_data.ipynb` dùng để:

- crawl danh sách URL bài viết theo chuyên mục
- lấy `title` và `content`
- lưu dữ liệu về `Dataset/`

Tài liệu:

- [Crawling Data/README.md](./Crawling%20Data/README.md)
- [Crawling Data/QUICK_START.md](./Crawling%20Data/QUICK_START.md)

### `SVM/`

Nhánh SVM hiện dùng:

- `TfidfVectorizer(max_features=150000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)`
- `LinearSVC(C=1.5, class_weight="balanced")`

Notebook `main_SVM.ipynb` có logic cache:

- nếu đã có `model/model_results.pkl`, notebook sẽ không train lại
- nếu chỉ có `model/inference_pipeline.pkl`, notebook sẽ predict lại tập test rồi tiếp tục vẽ / đánh giá
- chỉ train khi chưa có artifact model

Tài liệu:

- [SVM/README.md](./SVM/README.md)
- [SVM/QUICK_START.md](./SVM/QUICK_START.md)

### `PhoBERT/`

Nhánh PhoBERT hiện dùng:

- `vinai/phobert-base-v2`
- head-tail tokenization
- weighted loss
- threshold calibration

Notebook `main_PhoBERT.ipynb` có logic cache:

- nếu `PhoBERT/model/` đã có model export, notebook bỏ qua fine-tune
- nếu đã có `PhoBERT/model/thresholds.json`, notebook bỏ qua calibration grid search
- notebook vẫn load model để chạy đánh giá và vẽ biểu đồ

Tài liệu:

- [PhoBERT/README.md](./PhoBERT/README.md)
- [PhoBERT/QUICK_START.md](./PhoBERT/QUICK_START.md)

### `Combined_Model_App/`

App kết hợp:

- load pipeline SVM
- load model PhoBERT
- chuẩn hóa điểm của từng nhánh
- đưa ra dự đoán cuối cùng và hiển thị kết quả riêng của từng model

Tài liệu:

- [Combined_Model_App/README.md](./Combined_Model_App/README.md)
- [Combined_Model_App/QUICK_START.md](./Combined_Model_App/QUICK_START.md)

### `LR/`

Thư mục thử nghiệm để so sánh nhanh với SVM:

- notebook chính: `LR/main_LR.ipynb`
- không có app
- không phải nhánh chính của project

Nếu mô hình này không hữu ích, bạn có thể xóa cả thư mục `LR/`.

## Cách bắt đầu nhanh

### Cách đầy đủ

1. Tạo dataset bằng [Crawling Data/crawl_data.ipynb](./Crawling%20Data/crawl_data.ipynb).
2. Chạy một trong hai notebook train:
   - [SVM/main_SVM.ipynb](./SVM/main_SVM.ipynb)
   - [PhoBERT/main_PhoBERT.ipynb](./PhoBERT/main_PhoBERT.ipynb)
3. Chạy app:
   - `streamlit run SVM/app/app_SVM.py`
   - `streamlit run PhoBERT/app/app_PhoBERT.py`
   - `streamlit run Combined_Model_App/app_combined.py`

### Cách nhanh nếu đã có model

```bash
streamlit run SVM/app/app_SVM.py
streamlit run PhoBERT/app/app_PhoBERT.py
streamlit run Combined_Model_App/app_combined.py
```

## Ghi chú

- Các notebook hiện được chỉnh theo hướng `Run All` an toàn hơn với cache có sẵn.
- Các app đều có giao diện sáng và khung preview nội dung bài báo có thể cuộn, chọn và copy.
- Sau khi thay đổi model, preprocessing, artifact hoặc đường dẫn, cần cập nhật lại tài liệu của thư mục liên quan.
- Hướng dẫn cài đặt và chạy tổng thể nằm ở [SETUP.md](./SETUP.md).
