# Quick Start

Hướng dẫn ngắn để dùng nhánh SVM.

## Cài thư viện

```bash
pip install pandas numpy matplotlib seaborn scikit-learn pyvi pyarrow tqdm
pip install streamlit requests beautifulsoup4
```

## Chạy notebook

1. Mở [main_SVM.ipynb](./main_SVM.ipynb).
2. Bấm `Run All`.
3. Notebook sẽ tạo hoặc dùng lại:
   - `model/model_results.pkl`
   - `model/inference_pipeline.pkl`

## Logic cache

- nếu đã có `model/model_results.pkl`, notebook sẽ không train lại
- nếu chỉ có `model/inference_pipeline.pkl`, notebook sẽ dùng pipeline đó để predict lại tập test và tiếp tục vẽ
- chỉ train khi chưa có artifact model

## File nên xem sau khi chạy

- `results/04_confusion_matrix.png`
- `results/05_f1_per_class.png`
- `results/06_support_vs_f1.png`

## Chạy app

Từ thư mục `SVM/`, chạy:

```bash
streamlit run app/app_SVM.py
```

## Nếu muốn train lại

- train lại toàn bộ: xóa `temp/` và `model/`
- chỉ train lại model: xóa `model/model_results.pkl` và `model/inference_pipeline.pkl`
