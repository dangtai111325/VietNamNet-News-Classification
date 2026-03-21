# Quick Start

Hướng dẫn ngắn để dùng nhánh SVM.

Nếu cần giải thích đầy đủ, đọc [README.md](./README.md).

## Cài thư viện

```bash
pip install pandas numpy matplotlib seaborn scikit-learn pyvi joblib pyarrow tqdm
pip install streamlit requests beautifulsoup4
```

## Train model

1. mở [main_SVM.ipynb](./main_SVM.ipynb)
2. chạy notebook từ trên xuống
3. đợi notebook tạo:
   - `model/model_results.pkl`
   - `model/inference_pipeline.pkl`
4. xem section chẩn đoán cuối notebook để biết class nào cần ưu tiên cải thiện

## Chạy app

Từ thư mục `SVM/`, chạy:

```bash
streamlit run app/app_SVM.py
```

Hoặc mở:

- [app/app_SVM.ipynb](./app/app_SVM.ipynb)

## Nếu muốn train lại

- train lại toàn bộ: xóa `temp/` và `model/`
- chỉ train lại model: xóa `model/model_results.pkl` và chạy lại phần train

## File quan trọng

- `main_SVM.ipynb`: notebook train
- `model/inference_pipeline.pkl`: pipeline inference
- `app/app_SVM.py`: app Streamlit
