# Quick Start

Hướng dẫn ngắn để dùng nhánh PhoBERT.

Nếu cần giải thích đầy đủ, đọc [README.md](./README.md).

## Cài thư viện

```bash
pip install transformers accelerate pandas numpy matplotlib seaborn scikit-learn pyvi pyarrow tqdm scipy
pip install streamlit requests beautifulsoup4
```

Bạn cần cài `torch` đúng bản CUDA của máy.

## Train model

1. mở [main_PhoBERT.ipynb](./main_PhoBERT.ipynb)
2. chạy notebook từ trên xuống
3. đợi notebook tạo:
   - `model/model.safetensors`
   - `model/label_config.json`
   - `model/thresholds.json`
4. xem thêm các file trong `results/`:
   - `03_confusion_matrix.png`
   - `04_f1_per_class.png`
   - `05_support_vs_f1.png`
   - `07_threshold_calibration.png`
5. xem section chẩn đoán cuối notebook để biết class nào cần ưu tiên cải thiện ở vòng train tiếp theo

## Chạy app

Từ thư mục `PhoBERT/`, chạy:

```bash
streamlit run app/app_PhoBERT.py
```

Hoặc mở:

- [app/app_PhoBERT.ipynb](./app/app_PhoBERT.ipynb)

## Nếu muốn train lại

- train lại toàn bộ: xóa `temp/` và `model/`
- chỉ fine-tune lại model: xóa `model/` rồi chạy lại phần train

## File quan trọng

- `main_PhoBERT.ipynb`: notebook train
- `model/label_config.json`: metadata class và preprocessing
- `model/thresholds.json`: threshold calibration
- `app/app_PhoBERT.py`: app Streamlit
