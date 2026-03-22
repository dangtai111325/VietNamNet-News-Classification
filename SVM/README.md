# SVM

Tài liệu này mô tả nhánh SVM của project: notebook train, artifact model và app.

Nếu chỉ cần bắt đầu nhanh, đọc [QUICK_START.md](./QUICK_START.md).

## Mục tiêu

Nhánh này phân loại bài báo VietNamNet bằng:

- `TfidfVectorizer`
- `LinearSVC`

Đây là nhánh nhẹ hơn PhoBERT, phù hợp khi:

- cần train và inference nhanh hơn
- muốn chạy tốt trên CPU
- cần pipeline gọn để triển khai

## Cấu hình hiện tại

- `LinearSVC(C=1.5, class_weight="balanced")`
- `TfidfVectorizer(max_features=150000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)`
- chia train/test theo tỷ lệ `85/15` có `stratify`

## Cấu trúc thư mục

```text
SVM/
├── main_SVM.ipynb
├── README.md
├── QUICK_START.md
├── app/
├── model/
├── results/
└── temp/
```

## Notebook chính làm gì

[main_SVM.ipynb](./main_SVM.ipynb) đi theo luồng:

1. đọc dữ liệu từ `Dataset/`
2. EDA phân bố lớp và độ dài văn bản
3. tiền xử lý tiếng Việt
4. vector hóa bằng TF-IDF
5. train `LinearSVC`
6. đánh giá bằng accuracy, F1-weighted, F1-macro và biểu đồ chẩn đoán
7. export `inference_pipeline.pkl`
8. in thêm phần chẩn đoán để tìm class yếu

## Visualization đầu ra

Notebook lưu các hình chính trong `results/`:

- `01_class_distribution.png`
- `02_text_length.png`
- `03_tfidf_vocab.png`
- `04_confusion_matrix.png`
- `05_f1_per_class.png`
- `06_support_vs_f1.png`

Các biểu đồ hiện tại:

- dùng bar ngang thay cho pie chart
- có confusion matrix raw counts và normalized
- có biểu đồ `support vs F1` để nhìn class ít mẫu nhưng khó

## Tiền xử lý

Pipeline hiện dùng:

1. ghép `title + title + content`
2. lowercase
3. bỏ dấu câu
4. bỏ số
5. `ViTokenizer`
6. loại stopwords

## Cache và logic `Run All`

Các artifact chính:

- `temp/processed_data.pkl`
- `temp/tfidf_data.pkl`
- `model/model_results.pkl`
- `model/inference_pipeline.pkl`

Logic hiện tại:

- nếu đã có `model/model_results.pkl`, notebook load lại kết quả và không train lại
- nếu chưa có `model_results.pkl` nhưng đã có `model/inference_pipeline.pkl`, notebook dùng pipeline đó để predict lại tập test rồi tiếp tục vẽ
- chỉ train khi cả hai artifact model đều chưa tồn tại

## File đầu ra quan trọng

### `model/model_results.pkl`

Chứa:

- model đã train
- nhãn dự đoán trên tập test
- metric
- thời gian train
- config train

### `model/inference_pipeline.pkl`

Đây là file app dùng để inference. Nó chứa:

- `vectorizer`
- `model`
- `stopwords`
- `classes`
- `config`

## Thư viện cần cài

```bash
pip install pandas numpy matplotlib seaborn scikit-learn pyvi pyarrow tqdm
```

Nếu muốn chạy app, cài thêm:

```bash
pip install streamlit requests beautifulsoup4
```

## Cách train

Mở [main_SVM.ipynb](./main_SVM.ipynb) và chạy từ trên xuống.

Nếu muốn train lại từ đầu:

1. xóa `temp/`
2. xóa `model/`
3. chạy lại notebook

Nếu chỉ muốn train lại model:

1. xóa `model/model_results.pkl`
2. xóa `model/inference_pipeline.pkl`
3. chạy lại notebook

## Cách đọc nhanh kết quả

Nên xem theo thứ tự:

1. `04_confusion_matrix.png`
2. `05_f1_per_class.png`
3. `06_support_vs_f1.png`

## Cách chạy app

Từ thư mục `SVM/`, chạy:

```bash
streamlit run app/app_SVM.py
```

Tài liệu app:

- [app/README.md](./app/README.md)
- [app/QUICK_START.md](./app/QUICK_START.md)
