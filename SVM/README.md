# SVM

Tài liệu này mô tả nhánh SVM của project: notebook train, artifact model và app demo.

Nếu bạn chỉ muốn chạy nhanh, đọc [QUICK_START.md](./QUICK_START.md).

## Mục tiêu

Nhánh này phân loại bài báo Vietnamnet vào 19 chủ đề bằng:

- `TF-IDF`
- `SVC`

Đây là nhánh nhẹ hơn PhoBERT, phù hợp khi:

- cần train và inference nhanh hơn
- muốn chạy tốt trên CPU
- muốn có pipeline gọn, dễ triển khai trên CPU

## Cấu trúc thư mục

```text
SVM/
├── main_SVM.ipynb
├── README.md
├── QUICK_START.md
├── app/
│   ├── app_SVM.py
│   ├── app_SVM.ipynb
│   ├── README.md
│   └── QUICK_START.md
├── model/
│   ├── model_results.pkl
│   └── inference_pipeline.pkl
├── results/
└── temp/
```

## Notebook chính làm gì

[main_SVM.ipynb](./main_SVM.ipynb) đi theo luồng:

1. đọc dữ liệu từ `Dataset/`
2. EDA phân bố lớp và độ dài văn bản
3. tiền xử lý tiếng Việt
4. vector hóa bằng TF-IDF
5. train `SVC`
6. đánh giá bằng accuracy, F1-weighted, F1-macro, confusion matrix và các biểu đồ chẩn đoán dễ đọc hơn
7. export `inference_pipeline.pkl`
8. in thêm section chẩn đoán để xem class yếu, cặp class nhầm nhiều, lệch precision/recall và gợi ý tối ưu tiếp theo

## Visualization đầu ra

Notebook hiện lưu thêm các hình trong `results/` để đọc kết quả nhanh hơn:

- `01_class_distribution.png`: số mẫu theo class và tỷ lệ từng class
- `02_text_length.png`: phân bố độ dài văn bản với median, P90, P95 và boxplot theo class
- `03_tfidf_vocab.png`: top unigram / bigram nổi bật theo IDF
- `04_confusion_matrix.png`: confusion matrix chuẩn hóa, raw counts và top confusion pairs
- `05_f1_per_class.png`: Precision / Recall / F1 theo từng class
- `06_support_vs_f1.png`: tương quan giữa support và F1 để nhìn ra lớp ít mẫu nhưng khó

## Tiền xử lý

Pipeline hiện tại của notebook SVM dùng:

1. ghép title theo kiểu tăng trọng số trước content
2. lowercase
3. bỏ dấu câu
4. bỏ số
5. `ViTokenizer`
6. loại stopwords

Điểm quan trọng:

- title được tăng trọng số trước content trong pipeline export
- nếu thay preprocessing trong notebook, artifact export cũng sẽ thay theo

## Mô hình hiện tại

Notebook hiện được cấu hình:

- `SVC(C=1, class_weight="balanced", probability=False)`
- `TfidfVectorizer(max_features=150000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)`
- split `85/15` có `stratify`

Notebook có cache:

- `temp/processed_data.pkl`
- `temp/tfidf_data.pkl`
- `model/model_results.pkl`

Nếu file cache đã tồn tại, notebook sẽ không tạo lại bước tương ứng.

## File đầu ra quan trọng

### 1. `model/model_results.pkl`

Chứa:

- model đã train
- nhãn dự đoán trên test
- metric
- thời gian train
- config train

### 2. `model/inference_pipeline.pkl`

Đây là file dùng cho inference.

Nó chứa:

- `vectorizer`
- `model`
- `stopwords`
- `classes`
- `config`

## Thư viện cần cài

Nhánh SVM cần các thư viện:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `pyvi`
- `joblib`
- `pyarrow`
- `tqdm`

Cài nhanh:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn pyvi joblib pyarrow tqdm
```

Nếu muốn chạy app SVM, cài thêm:

```bash
pip install streamlit requests beautifulsoup4
```

## Cách train

Mở [main_SVM.ipynb](./main_SVM.ipynb) và chạy theo thứ tự từ trên xuống.

Các bước nặng đều có cache, nên:

- lần đầu sẽ lâu hơn
- các lần sau sẽ nhanh hơn nếu không xóa cache

Nếu muốn train lại từ đầu:

1. xóa `temp/`
2. xóa `model/`
3. chạy lại notebook

Nếu chỉ muốn train lại model:

1. xóa `model/model_results.pkl`
2. xóa `model/inference_pipeline.pkl` nếu muốn export lại sạch
3. chạy lại từ section train trở xuống

Sau khi notebook chạy xong, section chẩn đoán cuối notebook sẽ in:

- các class yếu nhất theo F1
- các class ít mẫu nhất
- class đang thiếu recall hoặc thiếu precision
- top cặp class bị nhầm nhiều nhất
- gợi ý tuning tiếp theo cho SVM

Nếu chỉ cần xem nhanh model đang yếu ở đâu, ưu tiên nhìn theo thứ tự:

1. `04_confusion_matrix.png`
2. `05_f1_per_class.png`
3. `06_support_vs_f1.png`

## Cách chạy app

App của nhánh này nằm trong [app/](./app/).

Chạy bằng terminal:

```bash
cd SVM
streamlit run app/app_SVM.py
```

Hoặc dùng notebook launcher:

- [app/app_SVM.ipynb](./app/app_SVM.ipynb)

Tài liệu app chi tiết:

- [app/README.md](./app/README.md)
- [app/QUICK_START.md](./app/QUICK_START.md)

## Khi nào cần cập nhật app

Bạn nên restart app khi:

- vừa train lại model SVM
- vừa export pipeline mới
- vừa sửa `app/app_SVM.py`

## Tóm tắt

Nhánh SVM là baseline mạnh, gọn và dễ chạy. Nó phù hợp khi bạn cần:

- một pipeline truyền thống cho tiếng Việt
- model nhẹ hơn PhoBERT
- inference đơn giản bằng một file pipeline duy nhất
