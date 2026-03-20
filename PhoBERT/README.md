# 🇻🇳 Phân loại tin tức Vietnamnet — PhoBERT Large

Ứng dụng tự động phân loại bài báo tiếng Việt vào **19 chủ đề** chỉ bằng cách dán URL hoặc nội dung bài.

> **Kết quả đạt được:** Độ chính xác **93.5%** trên 32,031 bài báo kiểm tra.

---

## 🚀 Chạy app ngay (nếu đã có model)

Mở terminal, gõ:

```bash
cd PhoBERT
streamlit run app/main_app.py
```

Sau đó mở trình duyệt tại `http://localhost:8501`.

> **Nếu chưa có model**, xem phần [Huấn luyện model](#-huấn-luyện-model-lần-đầu) bên dưới.

---

## 🖥️ Tính năng app

| Tính năng | Mô tả |
|-----------|-------|
| 🔗 **Nhập URL** | Dán link bài báo Vietnamnet → app tự tải và phân loại |
| 📝 **Nhập text** | Dán tiêu đề + nội dung thủ công (dùng khi URL bị chặn) |
| 📋 **Batch URL** | Dán nhiều URL cùng lúc → tải kết quả file CSV |
| 📜 **Lịch sử** | Xem lại tất cả phân loại trong phiên, xuất CSV |
| ⚠️ **Cảnh báo** | Thông báo khi bài có thể thuộc 2 chủ đề (độ phân cách thấp) |
| 🎯 **Threshold** | Tự động áp dụng hiệu chỉnh ngưỡng để tăng độ chính xác |

---

## 📁 Cấu trúc thư mục

```
PhoBERT/
├── main_PhoBERT.ipynb       ← Notebook huấn luyện model
├── app_PhoBERT.ipynb        ← Notebook khởi động app
├── app/
│   └── main_app.py          ← Code Streamlit app
├── model/                   ← Model đã huấn luyện (tạo sau khi train)
│   ├── model.safetensors    ← Trọng số PhoBERT fine-tuned (~1.4 GB)
│   ├── config.json          ← Cấu hình kiến trúc model
│   ├── vocab.txt            ← Từ điển PhoBERT (64,000 từ)
│   ├── tokenizer_config.json
│   ├── label_config.json    ← Danh sách 19 chủ đề + cấu hình
│   └── thresholds.json      ← Ngưỡng hiệu chỉnh per-class
├── temp/                    ← File cache tạm (tự động tạo)
│   ├── processed_data.pkl   ← Văn bản đã tiền xử lý (~cache)
│   ├── headtail_train.pkl   ← Dataset train đã tokenize (~cache)
│   └── headtail_test.pkl    ← Dataset test đã tokenize (~cache)
└── results/                 ← Biểu đồ và báo cáo kết quả
    ├── 01_class_distribution.png
    ├── 02_text_length.png
    ├── 03_confusion_matrix.png
    ├── 04_f1_per_class.png
    ├── 05_training_curves.png
    ├── 06_threshold_calibration.png
    └── classification_report.txt
```

---

## 📦 Cài đặt thư viện

**Yêu cầu tối thiểu:**
- Python 3.10+
- GPU với ít nhất 16 GB VRAM để train (khuyến nghị RTX 3090/4090)
- Chỉ cần CPU để chạy app sau khi đã có model

**Cài đặt:**

```bash
# PyTorch với CUDA 12.6 (RTX 40xx series)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Thư viện AI
pip install transformers accelerate

# Xử lý tiếng Việt + dữ liệu
pip install pyvi pandas numpy scikit-learn tqdm joblib

# Visualization
pip install matplotlib seaborn

# App
pip install streamlit requests beautifulsoup4 scipy
```

> Nếu dùng GPU cũ hơn (RTX 30xx), thay `cu126` bằng `cu121`.

---

## 🏷️ 19 Chủ đề phân loại

| | | | |
|--|--|--|--|
| Bạn đọc | Bảo vệ người tiêu dùng | Bất động sản | Chính trị |
| Công nghệ | Dân tộc - Tôn giáo | Đời sống | Du lịch |
| Giáo dục | Kinh doanh | Ô tô - Xe máy | Pháp luật |
| Sức khỏe | Thế giới | Thể thao | Thị trường tiêu dùng |
| Thời sự | Tuần Việt Nam | Văn hóa - Giải trí | |

---

## 🔧 Huấn luyện model (lần đầu)

Mở `main_PhoBERT.ipynb` trong Jupyter và chạy từng section theo thứ tự:

| Section | Thời gian | Nội dung | Ghi chú |
|---------|-----------|----------|---------|
| **0 — Setup** | ~10 giây | Kiểm tra GPU, cấu hình | Chạy mỗi lần mở notebook |
| **1 — Load Data** | ~5 giây | Đọc 213,540 bài báo | — |
| **2 — EDA** | ~30 giây | Vẽ biểu đồ phân tích dữ liệu | — |
| **3 — Tiền xử lý** | ~30 phút (lần đầu) | ViTokenize toàn bộ corpus | Lần sau bỏ qua nhờ cache |
| **4 — Tokenize** | ~5 phút (lần đầu) | Tạo HeadTail dataset | Lần sau bỏ qua nhờ cache |
| **5 — Huấn luyện** | ~93 phút | Fine-tune PhoBERT 5 epochs | Bỏ qua nếu model đã có |
| **6 — Đánh giá** | ~2 phút | Báo cáo, biểu đồ kết quả | — |
| **7 — Export** | ~5 giây | Lưu config cho app | — |
| **8 — Threshold** | ~1 phút | Hiệu chỉnh ngưỡng per-class | Cải thiện F1-macro +0.45% |

> **Lưu ý:** Từ lần 2 trở đi, Section 3 và 4 sẽ tự bỏ qua nhờ cache — tiết kiệm ~35 phút.

---

## 📊 Kết quả

| Mô hình | Accuracy | F1-weighted | F1-macro |
|---------|----------|-------------|----------|
| SVM (baseline) | 91.53% | 91.50% | 90.55% |
| PhoBERT Large | 93.09% | 93.11% | 92.61% |
| **PhoBERT + Threshold** | **93.51%** | **93.51%** | **93.06%** |

**Chủ đề khó nhất** (ít dữ liệu / nội dung chồng lấn):

| Chủ đề | F1 |
|--------|----|
| Dân tộc - Tôn giáo | 83.4% |
| Thời sự | 85.7% |
| Kinh doanh | 86.8% |

**Chủ đề dễ nhất:**

| Chủ đề | F1 |
|--------|----|
| Thể thao | 99.5% |
| Thế giới | 98.2% |
| Ô tô - Xe máy | 98.2% |

---

## ⚙️ Thông số kỹ thuật model

| Thông số | Giá trị |
|----------|---------|
| Kiến trúc | `vinai/phobert-large` (369M tham số) |
| Chiến lược token | Head-Tail: 127 token đầu + 127 token cuối |
| Độ dài tối đa | 256 tokens |
| Batch size | 64 |
| Learning rate | 1e-5 (cosine decay) |
| Số epoch | 5 (EarlyStopping patience=3) |
| Optimizer | AdamW fused + weight_decay=0.01 |
| Warmup | 200 steps |
| Precision | BF16 (RTX 4090 native) |
| Class weighting | `compute_class_weight("balanced")` |
| Threshold | Per-class calibration (grid search 3 passes) |
| Train time | ~93 phút (RTX 4090) |

### Tại sao dùng Head-Tail?

Bài báo Vietnamnet trung bình dài **600 tokens**, nhưng PhoBERT chỉ xử lý được **256 tokens** một lúc. Thay vì cắt bỏ phần cuối, chiến lược Head-Tail giữ lại:
- **127 tokens đầu** — tiêu đề, mở bài, chủ đề chính
- **127 tokens cuối** — kết luận, tóm tắt

```
[CLS] + 127 tokens đầu + 127 tokens cuối + [SEP]  =  256 tokens
```

---

## 🔄 Cập nhật model mới

1. Mở `main_PhoBERT.ipynb`
2. Sửa `MODEL_NAME` (Section 0 — cell c004) nếu muốn đổi model
3. Xóa `model/config.json` để training chạy lại (hoặc giữ nguyên để bỏ qua)
4. Chạy lại từ Section 5
5. Restart Streamlit app → tự load model mới

---

## 🛠️ Ghi chú kỹ thuật

- **Không loại stopwords**: PhoBERT tự học ngữ cảnh — loại stopwords có thể làm giảm độ chính xác.
- **ViTokenizer bắt buộc**: PhoBERT được pretrain trên văn bản đã qua ViTokenizer (tách từ tiếng Việt).
- **Threshold calibration**: Điều chỉnh ngưỡng quyết định riêng cho từng class để cân bằng precision/recall. Quyết định = `argmax(softmax(logits) / thresholds)`. Lưu tại `model/thresholds.json`.
- **WeightedTrainer**: Dùng class-balanced cross-entropy loss để xử lý dữ liệu mất cân bằng (Dân tộc-Tôn giáo chỉ có 3,309 bài vs ~12,500 của class lớn).
- **Cache**: File `.pkl` trong `temp/` giúp bỏ qua bước tiền xử lý tốn thời gian. Xóa nếu muốn chạy lại từ đầu.
