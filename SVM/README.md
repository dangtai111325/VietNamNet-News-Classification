# Phân Loại Tin Tức Vietnamnet — LinearSVC

Hệ thống phân loại tự động bài báo tiếng Việt vào **19 chủ đề** sử dụng
TF-IDF + LinearSVC. Bao gồm pipeline training đầy đủ và app web Streamlit
để demo và sử dụng thực tế.

---

## Cấu Trúc Thư Mục

```
SVM/
├── main_SVM.ipynb              ← Notebook training (chạy để tạo model)
├── app_SVM.ipynb               ← Notebook khởi động app Streamlit
├── README                      ← File này
│
├── app/
│   └── main_app.py             ← Source code app Streamlit
│
├── model/                      ← Sinh ra sau khi chạy main_SVM.ipynb
│   ├── model_results.pkl       ← Kết quả training (metrics, y_pred...)
│   └── inference_pipeline.pkl  ← Pipeline đầy đủ cho inference (app dùng file này)
│
├── results/                    ← Sinh ra sau khi chạy main_SVM.ipynb
│   ├── 01_class_distribution.png
│   ├── 02_text_length.png
│   ├── 03_tfidf_vocab.png
│   ├── 04_confusion_matrix.png
│   ├── 05_f1_per_class.png
│   └── classification_report.txt
│
└── temp/                       ← Cache trung gian (có thể xoá để train lại từ đầu)
    ├── processed_data.pkl      ← Corpus đã tokenize
    └── tfidf_data.pkl          ← Ma trận TF-IDF + train/test split
```

---

## 19 Chủ Đề

Bạn đọc · Bảo vệ người tiêu dùng · Bất động sản · Chính trị · Công nghệ ·
Dân tộc - Tôn giáo · Đời sống · Du lịch · Giáo dục · Kinh doanh ·
Ô tô - Xe máy · Pháp luật · Sức khỏe · Thế giới · Thể thao ·
Thị trường tiêu dùng · Thời sự · Tuần Việt Nam · Văn hóa - Giải trí

---

## Yêu Cầu

Python 3.9+ và các thư viện sau:

```
pip install scikit-learn pandas numpy pyvi joblib
pip install matplotlib seaborn
pip install streamlit requests beautifulsoup4
```

Dữ liệu cần có trong `../Dataset/` (19 file `.parquet`, mỗi file 1 chủ đề)
và file `../vietnamese-stopwords.txt`.

---

## Hướng Dẫn Sử Dụng

### 1. Training Model

Mở `main_SVM.ipynb` trong Jupyter và chọn **Run All**.

Pipeline gồm 7 section, mỗi section nặng đều có cache — lần đầu chạy
mất khoảng 20–40 phút (tokenize), các lần sau load từ cache trong vài giây.

| Section | Nội dung | Cache |
|---------|----------|-------|
| 0. Setup | Import, cấu hình đường dẫn | — |
| 1. Load data | Đọc 19 parquet, lọc bài thiếu cả title lẫn content | — |
| 2. EDA | Biểu đồ phân bố, độ dài văn bản, bảng tổng hợp | — |
| 3. Tiền xử lý | ViTokenizer song song (joblib), loại stopwords | `temp/processed_data.pkl` |
| 4. TF-IDF | Vectorize, train/test split 85/15 | `temp/tfidf_data.pkl` |
| 5. Training | LinearSVC C=1, class_weight=balanced | `model/model_results.pkl` |
| 6. Đánh giá | Report, confusion matrix, F1 per class | `results/` |
| 7. Export | Đóng gói pipeline cho app | `model/inference_pipeline.pkl` |

Để **train lại từ đầu**: xoá thư mục `temp/` và `model/`, sau đó Run All.

Để **chỉ train lại model** (giữ tokenize + TF-IDF): xoá chỉ `model/`, chạy lại từ Section 5.

---

### 2. Chạy App

**Cách A — Dùng notebook** (khuyến nghị):

Mở `app_SVM.ipynb`, chạy tuần tự:
- Cell 0: Kiểm tra thư viện và pipeline
- Cell 1: Ghi code vào `app/main_app.py`
- Cell 2: Khởi động Streamlit

Mở trình duyệt tại: **http://localhost:8501**

**Cách B — Dùng terminal trực tiếp**:

```bash
cd SVM
streamlit run app/main_app.py
```

> Yêu cầu: `model/inference_pipeline.pkl` phải tồn tại (đã chạy Section 7
> trong `main_SVM.ipynb`).

---

## Tính Năng App

| Tab | Mô tả |
|-----|-------|
| 🔗 Nhập URL | Dán link bài báo → tự động scrape → phân loại (nhấn Enter hoặc nút) |
| 📝 Nhập text | Paste title + content thủ công (dùng khi URL bị chặn / paywall) |
| 📋 Batch URL | Nhiều URL cùng lúc, hiển thị bảng kết quả, tải CSV |
| 📜 Lịch sử | Toàn bộ lịch sử phân loại trong session, tải CSV, xoá |

Mỗi kết quả phân loại hiển thị:
- Chủ đề dự đoán + màu tin cậy (xanh / cam / đỏ)
- Top 5 chủ đề khả năng kèm thanh progress
- **Top 10 từ khoá** đóng góp vào quyết định (từ `LinearSVC.coef_`)
- **Cảnh báo** nếu top-1 và top-2 chênh lệch < 5%
- Nội dung bài báo đã scrape (xem trước 600 ký tự)

---

## Thông Số Mô Hình

| Thành phần | Cấu hình |
|------------|----------|
| Vectorizer | `TfidfVectorizer(max_features=150_000, ngram_range=(1,2), min_df=2, sublinear_tf=True)` |
| Classifier | `LinearSVC(C=1, class_weight='balanced', max_iter=5000)` |
| Tiền xử lý | lowercase → bỏ dấu câu → bỏ số → ViTokenizer → loại stopwords |
| Tiêu đề | Lặp 2 lần trước content để tăng trọng số |
| Split | 85% train / 15% test, stratified |

---

## Workflow Cập Nhật Model

Khi train lại model (dữ liệu mới, thay đổi tham số):

```
1. Chỉnh sửa tham số trong main_SVM.ipynb (Section 0 — Config)
2. Xoá các folder cần train lại: temp/ và/hoặc model/
3. Run All main_SVM.ipynb
4. Mở app_SVM.ipynb → chạy Cell 2 (Streamlit tự reload pipeline mới)
```

App không cần cấu hình thêm — đường dẫn pipeline được tự động tính
tương đối từ vị trí `main_app.py`.
