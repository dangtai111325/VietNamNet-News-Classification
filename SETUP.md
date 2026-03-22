# Setup

Tài liệu này hướng dẫn cách cài môi trường và chạy toàn bộ project theo thứ tự hợp lý.

## 1. Yêu cầu chung

- Python 3.10 hoặc mới hơn
- `pip`
- Jupyter Notebook hoặc VS Code / Jupyter extension
- nếu chạy PhoBERT bằng GPU: PyTorch đúng với CUDA của máy

## 2. Cài thư viện

### Bộ thư viện chung

```bash
pip install pandas numpy matplotlib seaborn scikit-learn pyvi pyarrow tqdm requests beautifulsoup4 streamlit scipy
```

### Thêm cho PhoBERT

```bash
pip install transformers accelerate
```

### Cài PyTorch

Chọn bản phù hợp với máy của bạn tại trang chính thức của PyTorch. Nếu không có GPU phù hợp, có thể dùng bản CPU.

## 3. Thứ tự chạy khuyến nghị

```text
Bước 1  ->  crawl dữ liệu nếu chưa có dataset
Bước 2  ->  chạy notebook SVM hoặc PhoBERT
Bước 3  ->  chạy app tương ứng
         hoặc chạy app kết hợp
```

Lệnh app:

```bash
streamlit run SVM/app/app_SVM.py
streamlit run PhoBERT/app/app_PhoBERT.py
streamlit run Combined_Model_App/app_combined.py
```

## 4. Chuẩn bị dataset

Nếu chưa có dữ liệu trong `Dataset/`, mở:

- [Crawling Data/crawl_data.ipynb](./Crawling%20Data/crawl_data.ipynb)

Tài liệu kèm theo:

- [Crawling Data/README.md](./Crawling%20Data/README.md)
- [Crawling Data/QUICK_START.md](./Crawling%20Data/QUICK_START.md)

## 5. Chạy nhánh SVM

Mở:

- [SVM/main_SVM.ipynb](./SVM/main_SVM.ipynb)

Cấu hình hiện tại:

- `LinearSVC(C=1.5, class_weight="balanced")`
- `TfidfVectorizer(max_features=150000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)`

Logic `Run All` hiện tại:

- nếu đã có `SVM/model/model_results.pkl`, notebook sẽ load lại ngay
- nếu chưa có file đó nhưng đã có `SVM/model/inference_pipeline.pkl`, notebook sẽ dùng pipeline để predict lại tập test
- chỉ train khi chưa có artifact model

Sau khi chạy xong, app dùng:

- `SVM/model/inference_pipeline.pkl`

## 6. Chạy nhánh PhoBERT

Mở:

- [PhoBERT/main_PhoBERT.ipynb](./PhoBERT/main_PhoBERT.ipynb)

Cấu hình chính hiện tại:

- `MODEL_NAME = "vinai/phobert-base-v2"`
- `MAX_LENGTH = 256`
- `NUM_EPOCHS = 7`
- `LR = 1e-5`

Logic `Run All` hiện tại:

- nếu `PhoBERT/model/` đã có model export, notebook bỏ qua fine-tune
- nếu `PhoBERT/model/thresholds.json` đã có, notebook bỏ qua calibration grid search
- notebook vẫn load model để đánh giá và vẽ

Sau khi chạy xong, app dùng các file trong:

- `PhoBERT/model/`

## 7. Chạy app SVM

```bash
streamlit run SVM/app/app_SVM.py
```

## 8. Chạy app PhoBERT

```bash
streamlit run PhoBERT/app/app_PhoBERT.py
```

## 9. Chạy app kết hợp

```bash
streamlit run Combined_Model_App/app_combined.py
```

## 10. Nếu muốn chạy nhanh mà không train lại

Bạn chỉ cần có sẵn artifact model rồi chạy app tương ứng.

Nếu chưa có artifact trên GitHub, dùng bản full project tại:

- <https://drive.google.com/drive/folders/1gW393KCdnYU4TDWjDZZBluvZT9aqSLIp?usp=drive_link>

## 11. Khi nào cần xóa cache

### SVM

Xóa `SVM/model/` khi bạn muốn train lại model từ đầu.

### PhoBERT

Xóa `PhoBERT/model/` khi bạn muốn fine-tune lại hoàn toàn.

### Cảnh báo

Không cần xóa cache chỉ để xem lại biểu đồ hoặc báo cáo nếu artifact model đã tồn tại.
