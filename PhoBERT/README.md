# PhoBERT

Tài liệu này mô tả nhánh PhoBERT của project: notebook train, artifact model và app demo.

Nếu bạn chỉ muốn chạy nhanh, đọc [QUICK_START.md](./QUICK_START.md).

## Mục tiêu

Nhánh này phân loại bài báo Vietnamnet vào 19 chủ đề bằng mô hình transformer tiếng Việt.

Cấu hình hiện tại của notebook dùng:

- `vinai/phobert-base-v2`
- chiến lược `head-tail`
- weighted loss cho dữ liệu lệch lớp
- threshold calibration sau train

Nhánh này phù hợp khi:

- ưu tiên độ chính xác cao hơn SVM
- có GPU để train hoặc inference nhanh hơn
- cần xác suất class và calibration tốt hơn

## Cấu trúc thư mục

```text
PhoBERT/
├── main_PhoBERT.ipynb
├── README.md
├── QUICK_START.md
├── app/
│   ├── app_PhoBERT.py
│   ├── app_PhoBERT.ipynb
│   ├── README.md
│   └── QUICK_START.md
├── model/
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── label_config.json
│   └── thresholds.json
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
7. đánh giá accuracy, F1-weighted, F1-macro
8. export `label_config.json`
9. threshold calibration và lưu `thresholds.json`
10. in thêm section chẩn đoán để xem class yếu, delta F1 trước/sau calibration, confusion pairs và gợi ý tối ưu tiếp theo

## Tiền xử lý

Pipeline hiện tại của notebook PhoBERT dùng:

1. ghép `title + content`
2. lowercase
3. bỏ dấu câu
4. bỏ số
5. `ViTokenizer`
6. không loại stopwords

Metadata export hiện ghi:

- `title_weight = 1`
- `stopwords = False`

## Mô hình hiện tại

Cấu hình hiện tại của notebook là:

- `MODEL_NAME = "vinai/phobert-base-v2"`
- `MAX_LENGTH = 256`
- `NUM_EPOCHS = 7`
- `LR = 1e-5`
- `metric_for_best_model = "f1_macro"`

Trên máy 12 GB VRAM, notebook đang ưu tiên `phobert-base-v2` thay vì `phobert-large`.

## Threshold calibration

Sau khi train và đánh giá, notebook có thêm bước threshold calibration.

Kết quả được lưu tại:

- `model/thresholds.json`

File này chứa:

- `temperature`
- threshold riêng cho từng class

Inference sau này có thể dùng:

- softmax sau temperature scaling
- chia tiếp theo threshold để ra quyết định class cuối cùng

## File đầu ra quan trọng

### 1. `model/`

Trainer sẽ lưu:

- model
- tokenizer
- config

### 2. `model/label_config.json`

Chứa:

- tên model
- danh sách class
- `label2id`
- `id2label`
- metadata preprocessing

### 3. `model/thresholds.json`

Chứa:

- `temperature`
- threshold per-class

## Thư viện cần cài

Nhánh PhoBERT cần các thư viện:

- `pandas`
- `numpy`
- `torch`
- `transformers`
- `accelerate`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `pyvi`
- `pyarrow`
- `tqdm`
- `scipy`

Ví dụ cài nhanh:

```bash
pip install transformers accelerate pandas numpy matplotlib seaborn scikit-learn pyvi pyarrow tqdm scipy
```

PyTorch cần cài theo đúng CUDA của máy bạn.

Nếu muốn chạy app PhoBERT, cài thêm:

```bash
pip install streamlit requests beautifulsoup4
```

## Cách train

Mở [main_PhoBERT.ipynb](./main_PhoBERT.ipynb) và chạy từ trên xuống.

Notebook có cache cho các bước nặng như:

- processed data
- dataset đã tokenize

Nếu muốn train lại từ đầu:

1. xóa `temp/`
2. xóa `model/`
3. chạy lại notebook

Nếu chỉ muốn fine-tune lại:

1. xóa `model/`
2. giữ `temp/`
3. chạy lại từ phần train trở xuống

Sau khi notebook chạy xong, section chẩn đoán cuối notebook sẽ in:

- các class yếu nhất sau calibration
- các class ít mẫu nhất
- class tăng hoặc giảm F1 nhiều nhất sau threshold calibration
- class đang thiếu recall hoặc thiếu precision sau calibration
- top cặp class bị nhầm nhiều nhất
- gợi ý tuning tiếp theo cho PhoBERT

## Cách chạy app

App của nhánh này nằm trong [app/](./app/).

Chạy bằng terminal:

```bash
cd PhoBERT
streamlit run app/app_PhoBERT.py
```

Hoặc dùng notebook launcher:

- [app/app_PhoBERT.ipynb](./app/app_PhoBERT.ipynb)

Tài liệu app chi tiết:

- [app/README.md](./app/README.md)
- [app/QUICK_START.md](./app/QUICK_START.md)

## Khi nào cần restart app

Bạn nên restart app khi:

- vừa train lại model
- vừa cập nhật `label_config.json`
- vừa cập nhật `thresholds.json`
- vừa sửa `app/app_PhoBERT.py`

## Tóm tắt

Nhánh PhoBERT là pipeline mạnh hơn về mặt mô hình. Nó phù hợp khi bạn cần:

- mô hình ngữ cảnh tốt hơn SVM
- xác suất class từ transformer
- threshold calibration cho các lớp khó hoặc ít mẫu
