# Quick Start

Hướng dẫn ngắn để crawl dữ liệu và tạo dataset.

## Cài thư viện

```bash
pip install requests beautifulsoup4 pandas pyarrow tqdm lxml
```

## Chạy notebook

1. Mở [crawl_data.ipynb](./crawl_data.ipynb).
2. Chạy các cell đầu để nạp thư viện và cấu hình.
3. Bấm `Run All` hoặc chạy lần lượt từng section.

## Kết quả mong đợi

Sau khi chạy xong, dữ liệu sẽ được lưu trong `Dataset/`.

## Kiểm tra nhanh

- có file dữ liệu mới trong `Dataset/`
- dữ liệu có `title`
- dữ liệu có `content`
- số lượng bản ghi không quá thấp so với kỳ vọng

## Bước tiếp theo

Sau khi đã có dataset, chuyển sang:

- [SVM/main_SVM.ipynb](../SVM/main_SVM.ipynb)
- [PhoBERT/main_PhoBERT.ipynb](../PhoBERT/main_PhoBERT.ipynb)
