# Combined Model App

Tài liệu này mô tả chi tiết ứng dụng kết hợp SVM và PhoBERT.

Thư mục này chứa:

- [app_combined.py](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/Combined_Model_App/app_combined.py)  
  Ứng dụng Streamlit dùng để suy luận bằng 2 mô hình cùng lúc.
- [app_combined.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/Combined_Model_App/app_combined.ipynb)  
  Notebook launcher để kiểm tra môi trường và khởi động app.

Nếu bạn chỉ muốn chạy nhanh app, đọc [QUICK_START.md](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/Combined_Model_App/QUICK_START.md).

## Mục tiêu của app

App này cho phép người dùng:

1. phân loại bài báo bằng cả SVM và PhoBERT cùng lúc
2. kết hợp điểm tin cậy của 2 mô hình để ra dự đoán cuối cùng
3. so sánh kết quả riêng của từng model
4. kiểm tra trường hợp hai model đồng thuận hoặc bất đồng
5. xử lý URL đơn, text thủ công và batch URL
6. tải lịch sử hoặc kết quả batch ra CSV

## App dùng những mô hình nào

App combined không tự train mô hình.

Nó sẽ load các artifact đã được export từ hai nhánh:

### 1. Nhánh SVM

- `SVM/model/inference_pipeline.pkl`

Pipeline này chứa:

- `vectorizer`
- `model`
- `stopwords`
- `classes`
- `config`

Model SVM hiện tại là:

- `SVC`
- kết hợp với `TF-IDF`

### 2. Nhánh PhoBERT

- `PhoBERT/model/`

Tối thiểu cần:

- `config.json`
- `label_config.json`
- `model.safetensors`
- `tokenizer_config.json`

Nếu có thêm:

- `thresholds.json`

thì app sẽ bật:

- `temperature scaling`
- `threshold calibration`

## Điều kiện để app chạy được

Trước khi chạy app, bạn cần có:

- Python 3
- đầy đủ thư viện cho Streamlit, SVM và PhoBERT
- pipeline SVM đã export
- model PhoBERT đã export

Nếu thiếu một trong hai bên, app combined sẽ không chạy được vì nó cần cả hai.

Notebook train tương ứng là:

- [main_SVM.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/SVM/main_SVM.ipynb)
- [main_PhoBERT.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/PhoBERT/main_PhoBERT.ipynb)

## Thư viện cần cài

App cần các thư viện sau:

- `streamlit`
- `requests`
- `beautifulsoup4`
- `pyvi`
- `numpy`
- `pandas`
- `scikit-learn`
- `torch`
- `transformers`
- `scipy`

Cài nhanh bằng:

```bash
pip install streamlit requests beautifulsoup4 pyvi numpy pandas scikit-learn torch transformers scipy
```

Lưu ý:

- app có thể chạy bằng CPU
- nếu có GPU thì phần PhoBERT sẽ suy luận nhanh hơn

## App kiểm tra gì khi khởi động bằng notebook

Trong [app_combined.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/Combined_Model_App/app_combined.ipynb), notebook launcher thường kiểm tra:

1. thư viện đã có đủ chưa
2. file `app_combined.py` có tồn tại không
3. pipeline SVM có tồn tại không
4. thư mục model PhoBERT có đủ file cần thiết không
5. Streamlit có khởi động được không

Nếu notebook báo thiếu model, cần quay lại notebook train để export lại artifact tương ứng.

## Luồng hoạt động của app

Luồng chính của app combined như sau:

1. load pipeline SVM bằng `st.cache_resource`
2. load tokenizer + model PhoBERT bằng `st.cache_resource`
3. load `label_config.json`
4. nếu có `thresholds.json`, load thêm:
   - `temperature`
   - vector `thresholds`
5. người dùng nhập dữ liệu bằng URL, text hoặc batch URL
6. app scrape nội dung nếu đầu vào là URL
7. app tiền xử lý riêng cho từng nhánh:
   - SVM theo logic TF-IDF của notebook SVM
   - PhoBERT theo logic của notebook PhoBERT
8. app lấy điểm tin cậy từ từng model
9. app nhân điểm tin cậy của hai model theo từng class
10. app chuẩn hóa lại để tổng bằng 1
11. app hiển thị:
   - kết quả ensemble
   - kết quả riêng của SVM
   - kết quả riêng của PhoBERT
   - top 5 class
   - top từ khóa phía SVM nếu model hỗ trợ
   - preview nội dung

## Tiền xử lý trong app

App combined phải khớp với từng notebook train.

### Nhánh SVM

Preprocess của SVM hiện là:

1. ghép `title + title + content`
2. lowercase
3. bỏ dấu câu
4. bỏ số
5. `ViTokenizer`
6. loại stopwords

### Nhánh PhoBERT

Preprocess của PhoBERT hiện là:

1. ghép `title + content`
2. lowercase
3. bỏ dấu câu
4. bỏ số
5. `ViTokenizer`

Điểm quan trọng:

- SVM và PhoBERT không dùng cùng một preprocessing
- app combined đã được đồng bộ để mỗi model dùng đúng logic train của riêng nó

## Head-Tail tokenization của PhoBERT

Với nhánh PhoBERT, app dùng:

- `MAX_LENGTH = 256`

Nếu văn bản dài:

- lấy phần đầu
- lấy phần cuối
- ghép lại theo chiến lược head-tail

Mục tiêu là giữ được cả mở bài lẫn đoạn cuối bài báo thay vì chỉ cắt phần đầu.

## App tính điểm tin cậy như thế nào

Đây là phần quan trọng nhất của app combined.

### Phía SVM

App dùng:

- `decision_function()`

để lấy score của từng class từ pipeline SVM.

Sau đó app:

1. áp softmax để đổi score sang phân phối điểm tin cậy tương đối
2. dùng phân phối này làm đầu ra của nhánh SVM

Lưu ý:

- đây không phải xác suất calibrated thật từ `predict_proba()`
- nó là score suy ra để phục vụ so sánh và kết hợp hai mô hình

### Phía PhoBERT

PhoBERT sinh:

- `logits`

Sau đó app áp dụng:

1. `temperature scaling`
2. `softmax`
3. nếu có `thresholds.json` thì chia thêm theo từng ngưỡng class

### Phần combined

Sau khi có 2 phân phối điểm tin cậy, app tính:

```text
Score(c | SVM, BERT) ∝ Score_SVM(c) × Score_BERT(c)
```

Rồi chuẩn hóa lại để tổng bằng 1.

Điều này có nghĩa:

- nếu cả hai model cùng đánh giá cao một class, class đó sẽ được đẩy lên mạnh
- nếu hai model bất đồng, phân phối combined sẽ cân bằng giữa hai bên

## App hiển thị những gì

Màn hình kết quả gồm 3 phần chính:

### 1. Kết quả ensemble

Hiển thị:

- class cuối cùng
- điểm tin cậy combined
- top 5 class sau khi kết hợp
- trạng thái đồng thuận hoặc bất đồng

### 2. Kết quả từng model

Hiển thị riêng:

- dự đoán của SVM
- dự đoán của PhoBERT
- điểm tin cậy của mỗi bên

Mục đích là để người dùng nhìn rõ:

- khi nào hai model giống nhau
- khi nào ensemble đang phải phân xử

### 3. Giải thích từ khóa

App lấy top từ khóa mạnh nhất từ SVM để hỗ trợ giải thích.

Phần này chỉ có khi model SVM có `coef_`. Nếu cấu hình SVM hiện tại không hỗ trợ, app sẽ tự ẩn phần đó.

## Các tab trong giao diện

App có 4 tab chính.

### 1. `🔗 Nhập URL`

Người dùng dán URL VietNamNet.

App sẽ:

1. scrape bài báo
2. chạy cả SVM và PhoBERT
3. kết hợp điểm tin cậy
4. hiển thị kết quả ensemble

### 2. `📝 Nhập text`

Dùng khi:

- URL không scrape được
- cần kiểm tra một bài báo copy tay
- muốn thử dữ liệu ngoài VietNamNet

### 3. `📋 Batch URL`

Cho phép dán nhiều URL, mỗi dòng một URL.

App sẽ:

- xử lý tuần tự
- hiển thị tiến độ
- báo URL nào thành công, URL nào lỗi
- cho tải CSV

### 4. `📜 Lịch sử`

Lưu lại các lần phân loại trong session hiện tại.

Người dùng có thể:

- xem bảng lịch sử
- tải CSV
- xóa lịch sử

## Cách app scrape nội dung

App combined dùng cùng kiểu scraper như các app còn lại:

- `requests`
- `BeautifulSoup`

Quy trình:

1. gửi request với header giả lập trình duyệt
2. bỏ các thẻ không cần thiết
3. thử nhiều selector để lấy tiêu đề
4. thử nhiều selector để lấy khối nội dung
5. fallback sang gom các thẻ `p` nếu cần

Vì vậy:

- app hoạt động tốt với đa số bài VietNamNet chuẩn
- nhưng vẫn có thể lỗi nếu cấu trúc website thay đổi

## File đầu ra và dữ liệu phiên

App combined không ghi database cố định.

Những gì app tạo trong lúc chạy:

- lịch sử trong `st.session_state`
- file CSV khi người dùng bấm tải

Lịch sử sẽ mất khi:

- restart app
- refresh session
- bấm xóa lịch sử

## Cách chạy app

### Cách 1: chạy bằng notebook launcher

Mở:

- [app_combined.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/Combined_Model_App/app_combined.ipynb)

Sau đó:

1. chạy cell kiểm tra môi trường
2. chạy cell kiểm tra file app
3. chạy cell khởi động Streamlit

### Cách 2: chạy bằng terminal

Từ thư mục `Combined_Model_App/`, chạy:

```bash
streamlit run app_combined.py
```

Sau khi chạy thành công, app thường mở ở:

```text
http://localhost:8501
```

## Khi nào cần restart app

Bạn nên restart app khi:

- vừa train lại model SVM
- vừa train lại model PhoBERT
- vừa export artifact mới
- vừa sửa `app_combined.py`

Lý do là cả hai model đều được cache bằng `st.cache_resource`.

## Lỗi thường gặp

### 1. Không load được SVM

Nguyên nhân:

- thiếu `SVM/model/inference_pipeline.pkl`

Cách xử lý:

- chạy [main_SVM.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/SVM/main_SVM.ipynb)

### 2. Không load được PhoBERT

Nguyên nhân:

- thiếu file trong `PhoBERT/model/`
- model chưa export
- thiếu `torch` hoặc `transformers`

Cách xử lý:

- chạy [main_PhoBERT.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/PhoBERT/main_PhoBERT.ipynb)
- kiểm tra lại environment

### 3. URL scrape thất bại

Nguyên nhân:

- mạng lỗi
- website thay đổi cấu trúc
- request bị chặn

Cách xử lý:

- chuyển sang tab `📝 Nhập text`
- dán nội dung thủ công

## Tóm tắt nhanh

Nhánh app này phù hợp khi bạn muốn:

- lấy kết quả mạnh hơn một model đơn lẻ
- nhìn rõ SVM nghĩ gì, PhoBERT nghĩ gì, và ensemble kết luận gì
- tận dụng ưu điểm tốc độ của SVM và ngữ cảnh của PhoBERT trong cùng một giao diện
