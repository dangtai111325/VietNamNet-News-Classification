# 🇻🇳 Phân Loại Tin Tức Vietnamnet

Hệ thống tự động phân loại bài báo tiếng Việt vào **19 chủ đề** — chỉ cần dán URL hoặc nội dung bài, phần mềm sẽ cho biết bài đó thuộc chủ đề gì và mức độ tin cậy.

---

## Đây là gì?

Đây là đề tài nghiên cứu về **phân loại văn bản tiếng Việt** sử dụng dữ liệu thực tế từ báo Vietnamnet. Dự án so sánh hai hướng tiếp cận khác nhau — một phương pháp truyền thống nhanh và nhẹ, một phương pháp dùng AI hiện đại — để xem cách nào hiệu quả hơn cho bài toán tiếng Việt.

Cả hai đều có **ứng dụng web** dùng được ngay trên trình duyệt, không cần biết lập trình.

---

## Dữ Liệu

| Thông tin | Chi tiết |
|-----------|----------|
| Nguồn | Báo Vietnamnet (vietnamnet.vn) |
| Tổng số bài | ~213.000 bài báo |
| Số chủ đề | 19 chủ đề |
| Định dạng | 19 file `.parquet` trong thư mục `Dataset/` |

**19 chủ đề:**

> Bạn đọc · Bảo vệ người tiêu dùng · Bất động sản · Chính trị · Công nghệ ·
> Dân tộc - Tôn giáo · Đời sống · Du lịch · Giáo dục · Kinh doanh ·
> Ô tô - Xe máy · Pháp luật · Sức khỏe · Thế giới · Thể thao ·
> Thị trường tiêu dùng · Thời sự · Tuần Việt Nam · Văn hóa - Giải trí

---

## Hai Phương Pháp

### 1. SVM (LinearSVC + TF-IDF) — Phương pháp truyền thống

Nằm trong thư mục `SVM/`

Cách hoạt động: Mỗi bài báo được chuyển thành một "danh sách từ khoá có trọng số" (TF-IDF), sau đó thuật toán SVM tìm đường phân chia tốt nhất giữa 19 chủ đề.

| Ưu điểm | Nhược điểm |
|---------|------------|
| Huấn luyện rất nhanh (vài phút) | Không hiểu ngữ cảnh câu văn |
| Chạy được trên máy thường | Phụ thuộc vào từ khoá xuất hiện |
| Dễ giải thích (xem từ nào quyết định) | Độ chính xác thấp hơn AI |

**Kết quả:** F1-weighted = **91.5%**

→ Xem chi tiết: [SVM/README.md](SVM/README.md)

---

### 2. PhoBERT Large — Phương pháp AI hiện đại

Nằm trong thư mục `PhoBERT/`

Cách hoạt động: Dùng mô hình ngôn ngữ PhoBERT (được huấn luyện trước trên hàng tỷ chữ tiếng Việt), sau đó tinh chỉnh thêm trên dữ liệu Vietnamnet. Mô hình này thực sự "đọc hiểu" câu văn theo ngữ cảnh.

| Ưu điểm | Nhược điểm |
|---------|------------|
| Độ chính xác cao hơn | Huấn luyện lâu (cần GPU mạnh) |
| Hiểu ngữ cảnh và sắc thái | Model file lớn (~1.4 GB) |
| Ổn định hơn với bài viết phức tạp | Chạy chậm hơn trên máy yếu |

**Kết quả:** F1-weighted = **93.5%** (sau hiệu chỉnh ngưỡng)

→ Xem chi tiết: [PhoBERT/README.md](PhoBERT/README.md)

---

## So Sánh Kết Quả

| | SVM | PhoBERT |
|---|---|---|
| **F1-weighted** | 91.50% | 93.51% |
| **Thời gian huấn luyện** | ~5 phút | ~8 giờ (GPU RTX 4090) |
| **Bộ nhớ model** | 30 MB | ~1.4 GB |
| **Yêu cầu phần cứng** | Máy thường | GPU NVIDIA khuyến nghị |

> Nếu chỉ muốn dùng app mà không cần train lại: copy thư mục `SVM/model/` hoặc `PhoBERT/model/` từ máy đã train vào đúng vị trí rồi chạy app thẳng.

---

## Cấu Trúc Thư Mục

```
Vietnamnet_News_Classification/
│
├── Dataset/                    ← 19 file .parquet (dữ liệu gốc)
│
├── SVM/                        ← Phương pháp 1: LinearSVC + TF-IDF
│   ├── main_SVM.ipynb          ← Notebook huấn luyện
│   ├── app_SVM.ipynb           ← Notebook khởi động app
│   └── README.md               ← Hướng dẫn chi tiết
│
├── PhoBERT/                    ← Phương pháp 2: PhoBERT Large
│   ├── main_PhoBERT.ipynb      ← Notebook huấn luyện
│   ├── app_PhoBERT.ipynb       ← Notebook khởi động app
│   └── README.md               ← Hướng dẫn chi tiết
│
└── README.md                   ← File này
```

---

## Bắt Đầu Nhanh

**Bước 1 — Cài Python và thư viện**

Cài [Python 3.9+](https://www.python.org/downloads/) và [Jupyter](https://jupyter.org/install), sau đó cài thư viện theo hướng dẫn trong README của từng phương pháp.

**Bước 2 — Chọn phương pháp**

- Máy thường, muốn chạy nhanh → dùng **SVM**
- Có GPU NVIDIA, muốn độ chính xác cao → dùng **PhoBERT**

**Bước 3 — Huấn luyện model**

Mở notebook `main_SVM.ipynb` hoặc `main_PhoBERT.ipynb` → chạy từng cell từ trên xuống.

**Bước 4 — Chạy app**

Mở notebook `app_SVM.ipynb` hoặc `app_PhoBERT.ipynb` → chạy Cell 2 → trình duyệt tự mở tại `http://localhost:8501`.

---

## Yêu Cầu Hệ Thống

| | SVM | PhoBERT |
|---|---|---|
| Python | 3.9+ | 3.9+ |
| RAM | 8 GB | 16 GB trở lên |
| GPU | Không cần | NVIDIA 8 GB VRAM trở lên |
| Dung lượng ổ cứng | ~2 GB | ~10 GB (bao gồm model gốc) |

---

*Dữ liệu thu thập từ vietnamnet.vn phục vụ mục đích nghiên cứu.*
