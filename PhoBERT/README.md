# PhoBERT

Tài liệu này mô tả nhánh PhoBERT của project: notebook train, artifact model và app.

Nếu chỉ cần bắt đầu nhanh, đọc [QUICK_START.md](./QUICK_START.md).

## Mục tiêu

Nhánh này phân loại bài báo VietNamNet bằng transformer tiếng Việt.

Cấu hình hiện tại:

- `vinai/phobert-base-v2`
- head-tail tokenization
- weighted loss
- threshold calibration

Nhánh này phù hợp khi:

- ưu tiên độ chính xác cao hơn SVM
- cần score xác suất từ transformer
- muốn tối ưu tốt hơn cho các lớp khó hoặc ít mẫu

## Cấu trúc thư mục

```text
PhoBERT/
├── main_PhoBERT.ipynb
├── README.md
├── QUICK_START.md
├── app/
├── model/
├── results/
└── temp/
```

## Notebook chính làm gì

[main_PhoBERT.ipynb](./main_PhoBERT.ipynb) đi theo luồng:

1. kiểm tra GPU và gợi ý cấu hình theo VRAM
2. đọc dữ liệu từ `Dataset/`
3. EDA phân bố lớp và độ dài văn bản
4. tiền xử lý tiếng Việt
5. tokenize theo chiến lược head-tail
6. fine-tune PhoBERT
7. đánh giá bằng accuracy, F1-weighted, F1-macro và biểu đồ theo class
8. export model và `label_config.json`
9. calibration và lưu `thresholds.json`
10. in thêm phần chẩn đoán để xem class yếu

## Visualization đầu ra

Notebook lưu trong `results/`:

- `01_class_distribution.png`
- `02_text_length.png`
- `03_confusion_matrix.png`
- `04_f1_per_class.png`
- `05_training_curves.png`
- `06_threshold_calibration.png`
- `07_support_vs_f1.png`

Các biểu đồ hiện tại:

- không dùng pie chart
- có confusion matrix raw counts và normalized
- training curves gói gọn vào 3 chart chính
- calibration có thêm delta F1 theo class và threshold theo class

## Tiền xử lý

Pipeline hiện dùng:

1. ghép `title + content`
2. lowercase
3. bỏ dấu câu
4. bỏ số
5. `ViTokenizer`
6. chuẩn hóa khoảng trắng

Metadata export hiện ghi:

- `title_weight = 1`
- `stopwords = False`

## Cấu hình chính hiện tại

- `MODEL_NAME = "vinai/phobert-base-v2"`
- `MAX_LENGTH = 256`
- `NUM_EPOCHS = 7`
- `LR = 1e-5`
- `metric_for_best_model = "f1_macro"`

## Cache và logic `Run All`

Logic hiện tại:

- nếu `model/` đã có model export, notebook bỏ qua fine-tune
- nếu `model/thresholds.json` đã tồn tại, notebook bỏ qua calibration grid search
- notebook vẫn load model để chạy inference trên tập test, in báo cáo và vẽ

## File đầu ra quan trọng

### `model/`

Chứa:

- `model.safetensors`
- `config.json`
- `tokenizer_config.json`

### `model/label_config.json`

Chứa:

- tên model
- danh sách class
- `label2id`
- `id2label`
- metadata preprocessing

### `model/thresholds.json`

Chứa:

- `temperature`
- threshold theo từng class

## Thư viện cần cài

```bash
pip install transformers accelerate pandas numpy matplotlib seaborn scikit-learn pyvi pyarrow tqdm scipy
```

PyTorch cần cài đúng theo CUDA hoặc CPU của máy.

Nếu muốn chạy app, cài thêm:

```bash
pip install streamlit requests beautifulsoup4
```

## Cách train

Mở [main_PhoBERT.ipynb](./main_PhoBERT.ipynb) và chạy từ trên xuống.

Nếu muốn train lại từ đầu:

1. xóa `temp/`
2. xóa `model/`
3. chạy lại notebook

## Cách đọc nhanh kết quả

Nên xem theo thứ tự:

1. `03_confusion_matrix.png`
2. `04_f1_per_class.png`
3. `06_threshold_calibration.png`
4. `07_support_vs_f1.png`

## Cách chạy app

Từ thư mục `PhoBERT/`, chạy:

```bash
streamlit run app/app_PhoBERT.py
```

Tài liệu app:

- [app/README.md](./app/README.md)
- [app/QUICK_START.md](./app/QUICK_START.md)
