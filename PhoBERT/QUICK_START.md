# Quick Start

Hướng dẫn ngắn để dùng nhánh PhoBERT.

## Cài thư viện

```bash
pip install transformers accelerate pandas numpy matplotlib seaborn scikit-learn pyvi pyarrow tqdm scipy
```

Bạn cũng cần cài `torch` phù hợp với máy.

## Chạy notebook

1. Mở [main_PhoBERT.ipynb](./main_PhoBERT.ipynb).
2. Bấm `Run All`.
3. Notebook sẽ tạo hoặc dùng lại:
   - `model/model.safetensors`
   - `model/label_config.json`
   - `model/thresholds.json`

## Logic cache

- nếu đã có model export trong `model/`, notebook sẽ không fine-tune lại
- nếu đã có `model/thresholds.json`, notebook sẽ không chạy lại calibration grid search
- notebook vẫn load model để đánh giá và vẽ

## File nên xem sau khi chạy

- `results/03_confusion_matrix.png`
- `results/04_f1_per_class.png`
- `results/05_training_curves.png`
- `results/06_threshold_calibration.png`
- `results/07_support_vs_f1.png`

## Chạy app

Từ thư mục `PhoBERT/`, chạy:

```bash
streamlit run app/app_PhoBERT.py
```

## Nếu muốn train lại

- train lại toàn bộ: xóa `temp/` và `model/`
- chỉ fine-tune lại model: xóa `model/`
