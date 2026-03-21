# SETUP — Hướng dẫn chạy dự án

Dự án gồm **3 bước** theo thứ tự. Mỗi bước chỉ cần bấm **Run All** một lần.

```
Bước 1  →  Crawling Data/crawl_data.ipynb       (tạo Dataset)
Bước 2  →  SVM/main_SVM.ipynb                   (train mô hình SVM)
         hoặc  PhoBERT/main_PhoBERT.ipynb       (train mô hình PhoBERT)
Bước 3  →  SVM/app/app_SVM.ipynb                (chạy web app SVM)
         hoặc  PhoBERT/app/app_PhoBERT.ipynb    (chạy web app PhoBERT)
```

---

## Bước 1 — Crawl Data (`Crawling Data/crawl_data.ipynb`)

### Yêu cầu
| Thư viện | Cài đặt |
|----------|---------|
| `requests` | `pip install requests` |
| `beautifulsoup4` | `pip install beautifulsoup4` |
| `lxml` | `pip install lxml` |
| `tqdm` | `pip install tqdm` |
| `pyarrow` | `pip install pyarrow` |

### Luồng chạy
1. **Section 1 — Chuẩn bị**: kiểm tra thư viện, tạo `Dataset/` và `data_URLs.json` nếu chưa có.
2. **Section 2 — Crawl URL**: crawl danh sách URL từ VietNamNet, lưu vào `Dataset/data_URLs.json`.
   Chỉ crawl các category chưa có URL.
3. **Section 3 — Crawl nội dung**: crawl tiêu đề và nội dung từng bài, lưu thành `Dataset/<category>.parquet`.
   Chỉ crawl các category chưa có file `.parquet`.

### Kết quả
- `Dataset/` chứa 19 file `.parquet` (một file per chủ đề)
- `Dataset/data_URLs.json` chứa danh sách URL đã crawl

### Lưu ý
- Mặc định crawl tối đa **499 trang** mỗi category (`MAX_PAGES = 500`).
- Có thể chỉnh `MAX_PAGES`, `NUM_WORKERS` trong Section 1.
- Nếu muốn crawl lại từ đầu: xóa `Dataset/data_URLs.json` và các file `.parquet`.

---

## Bước 2a — Train SVM (`SVM/main_SVM.ipynb`)

### Yêu cầu hệ thống
- RAM: **≥ 8 GB** (temp files ~2 GB)
- Disk: **≥ 3 GB** trống (processed_data ~1.4 GB, tfidf ~584 MB)
- Không cần GPU

### Yêu cầu thư viện
```
pip install pandas numpy matplotlib seaborn scikit-learn pyvi joblib pyarrow tqdm
```

### Luồng chạy
| Section | Nội dung | Cache |
|---------|----------|-------|
| 1 | Import + config | — |
| 2 | Kiểm tra Dataset | — |
| 3 | Load & EDA | — |
| 4 | Tiền xử lý + tokenize | `temp/processed_data.pkl` |
| 5 | TF-IDF vectorization | `temp/tfidf_data.pkl` |
| 6 | Đánh giá | `results/` |
| 7 | Export inference pipeline | `model/inference_pipeline.pkl` |

> Nếu cache đã có, các bước nặng sẽ được bỏ qua tự động.

### Kết quả
- `model/inference_pipeline.pkl` — pipeline dùng cho app
- `model/model_results.pkl` — kết quả training
- `results/` — biểu đồ và classification report

---

## Bước 2b — Train PhoBERT (`PhoBERT/main_PhoBERT.ipynb`)

### Yêu cầu hệ thống
| VRAM GPU | Mô hình được chọn | Batch |
|----------|------------------|-------|
| ≥ 24 GB (RTX 4090, A100 40GB) | `phobert-large` hoặc `phobert-base-v2` | 64 |
| 16–23 GB (RTX 4080, T4 16GB) | `phobert-large` hoặc `phobert-base-v2` | 32 × accum 2 |
| 10–15 GB (RTX 3080 10GB, A3000 12GB) | `phobert-base-v2` | 32 × accum 2 |
| 6–9 GB (RTX 3060, 2060) | `phobert-base-v2` | 16 × accum 4 |
| < 6 GB | `phobert-base-v2` | 8 × accum 8 (chậm) |

> Notebook có bảng gợi ý theo VRAM. Cấu hình hiện tại của repo đang ưu tiên `phobert-base-v2`.

### Yêu cầu thư viện
```
pip install pandas numpy torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate seaborn matplotlib scikit-learn tqdm pyarrow scipy
```
> Thay `cu121` bằng version CUDA phù hợp với máy bạn. Xem: https://pytorch.org/get-started/locally/

### Yêu cầu khác
- **Kết nối internet** để download model PhoBERT lần đầu, thường là `vinai/phobert-base-v2`

### Luồng chạy
| Section | Nội dung | Cache |
|---------|----------|-------|
| 1 | Import + config + GPU auto-tune | — |
| 2 | Kiểm tra Dataset | — |
| 3 | Load & EDA | — |
| 4 | Tokenize (head-tail 127+127) | `temp/processed_data.pkl` |
| 5 | Fine-tune PhoBERT | `temp/checkpoints_ls/` |
| 6 | Đánh giá + visualization | `results/` |
| 7 | Export model | `model/` |
| 8 | Threshold calibration | `model/thresholds.json` |

### Kết quả
- `model/model.safetensors` + các file config — model dùng cho app
- `model/label_config.json` — mapping label
- `model/thresholds.json` — threshold calibration
- `results/` — biểu đồ và classification report

---

## Bước 3a — Web App SVM (`SVM/app/app_SVM.ipynb`)

### Yêu cầu
- File `SVM/model/inference_pipeline.pkl` phải tồn tại (chạy Bước 2a trước)
- Thư viện: `pip install streamlit requests beautifulsoup4 pyvi scikit-learn numpy pandas`

### Cách chạy
1. Bấm **Run All** trong `app/app_SVM.ipynb`
2. Trình duyệt tự mở tại `http://localhost:8501`
3. Nhấn **■ Interrupt Kernel** để dừng app

### Tính năng app
- Nhập URL bài báo → phân loại tự động
- Nhập text thủ công
- Batch nhiều URL → xuất CSV
- Lịch sử phân loại trong session

---

## Bước 3b — Web App PhoBERT (`PhoBERT/app/app_PhoBERT.ipynb`)

### Yêu cầu
- Thư mục `PhoBERT/model/` phải có: `model.safetensors`, `label_config.json`, `config.json`, `tokenizer_config.json`
- Thư viện: `pip install streamlit requests beautifulsoup4 pyvi torch transformers numpy pandas scipy`

### Cách chạy
1. Bấm **Run All** trong `app/app_PhoBERT.ipynb`
2. Trình duyệt tự mở tại `http://localhost:8501`
3. Nhấn **■ Interrupt Kernel** để dừng app

---

## Tóm tắt thứ tự 3 lần Run All

```
Run All  →  Crawling Data/crawl_data.ipynb     (~vài tiếng tùy tốc độ mạng)
Run All  →  SVM/main_SVM.ipynb                 (~30-60 phút)
         hoặc PhoBERT/main_PhoBERT.ipynb       (~2-8 tiếng tùy GPU)
Run All  →  SVM/app/app_SVM.ipynb              (< 1 phút, app chạy liền)
         hoặc PhoBERT/app/app_PhoBERT.ipynb    (< 1 phút, app chạy liền)
```

> Nếu đã có Dataset và Model sẵn, chỉ cần **1 lần Run All** ở bước App là đủ.
