# QUICK START

## 1. Mục đích của tài liệu này

File này hướng dẫn bạn chạy dự án theo cách đơn giản nhất, dành cho người chưa quen với Python, Jupyter Notebook hoặc bài toán phân loại văn bản.

Mục tiêu là để bạn có thể:

- mở đúng notebook
- cài đúng môi trường
- chạy đúng thứ tự
- hiểu thư mục nào được tạo ra
- biết cách xử lý các lỗi cơ bản

## 2. Bạn sẽ chạy file nào?

Bạn chỉ cần chạy một file duy nhất:

- `VietNamNet_News_Classification.ipynb`

Đây là notebook trung tâm của toàn bộ dự án.

## 3. Trước khi bắt đầu, bạn cần gì?

Bạn cần chuẩn bị:

- một máy Windows
- Python đã được cài
- VS Code hoặc Jupyter Notebook
- kết nối mạng nếu muốn crawl dữ liệu mới từ VietNamNet

Nếu bạn chưa cài Python, hãy cài trước rồi mới làm tiếp.

## 4. Kiểm tra Python đã cài hay chưa

Mở PowerShell hoặc Command Prompt và gõ:

```powershell
python --version
```

Nếu máy bạn hiện ra phiên bản Python, ví dụ `Python 3.10.x` hoặc `Python 3.11.x`, nghĩa là Python đã được cài.

Nếu máy báo không tìm thấy lệnh `python`, bạn cần cài Python trước.

## 5. Cài Python nếu máy chưa có

### Cách dễ nhất trên Windows

1. Truy cập trang chủ Python.
2. Tải bản Python phù hợp cho Windows.
3. Chạy file cài đặt.
4. Khi cài, nhớ bật tùy chọn thêm Python vào `PATH`.
5. Hoàn tất cài đặt.
6. Đóng terminal cũ và mở lại terminal mới.

Sau đó kiểm tra lại:

```powershell
python --version
```

## 6. Mở đúng thư mục dự án

Trong terminal, di chuyển đến thư mục chứa notebook. Ví dụ:

```powershell
cd C:\Users\DELL\Downloads\HTTM\VietNamNet_News_Classification
```

Để kiểm tra bạn đang ở đúng nơi hay chưa, chạy:

```powershell
dir
```

Bạn nên nhìn thấy ít nhất các file:

- `README.md`
- `QUICK_START.md`
- `VietNamNet_News_Classification.ipynb`

và các thư mục:

- `dataset`
- `model`
- `result`
- `temp`

## 7. Tạo môi trường ảo Python

Đây là bước rất nên làm, đặc biệt nếu bạn là người mới.

Môi trường ảo giúp:

- tránh xung đột thư viện với các project khác
- dễ quản lý gói cài đặt
- dễ xóa và làm lại nếu có lỗi

### Tạo môi trường ảo

Trong thư mục dự án, chạy:

```powershell
python -m venv .venv
```

Nếu lệnh chạy thành công, thư mục `.venv` sẽ được tạo ra.

### Kích hoạt môi trường ảo trên PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

Khi kích hoạt thành công, đầu dòng terminal thường sẽ xuất hiện tiền tố như:

```text
(.venv)
```

### Nếu PowerShell chặn script

Nếu gặp lỗi liên quan đến quyền chạy script, dùng lệnh sau trong PowerShell hiện tại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Sau đó chạy lại:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 8. Nâng cấp pip

Sau khi kích hoạt môi trường ảo, nên nâng cấp `pip`:

```powershell
python -m pip install --upgrade pip
```

## 9. Cài các thư viện cần thiết

Notebook có thể dùng nhiều thư viện cho các nhóm công việc khác nhau:

- crawl dữ liệu
- xử lý dữ liệu
- huấn luyện PhoBERT
- vẽ biểu đồ
- lưu kết quả parquet

Bạn có thể cài theo lệnh sau:

```powershell
pip install jupyter notebook ipykernel pandas numpy matplotlib seaborn scikit-learn scipy tqdm requests beautifulsoup4 lxml pyarrow aiohttp newspaper3k trafilatura readability-lxml goose3 torch transformers accelerate
```

Lưu ý:

- tùy cấu hình máy, việc cài `torch` có thể khác nhau
- nếu bạn có GPU NVIDIA và muốn dùng CUDA, bạn có thể cần cài bản `torch` phù hợp riêng
- nếu bạn chưa chắc, hãy cài bản mặc định trước để chạy thử

## 10. Cài kernel cho notebook

Để notebook nhận đúng môi trường ảo, chạy:

```powershell
python -m ipykernel install --user --name vietnamnet-news-classification --display-name "Python (vietnamnet-news-classification)"
```

Lệnh này giúp bạn chọn đúng kernel khi mở notebook.

## 11. Cách mở notebook bằng VS Code

### Bước 1

Mở VS Code.

### Bước 2

Chọn `File` -> `Open Folder...` và mở thư mục:

```text
C:\Users\DELL\Downloads\HTTM\VietNamNet_News_Classification
```

### Bước 3

Mở file:

```text
VietNamNet_News_Classification.ipynb
```

### Bước 4

Ở góc trên bên phải của notebook, chọn đúng kernel:

```text
Python (vietnamnet-news-classification)
```

Nếu chưa thấy, hãy chắc chắn bạn đã làm bước cài kernel ở trên.

## 12. Cách mở notebook bằng Jupyter

Trong terminal đang kích hoạt môi trường ảo, chạy:

```powershell
jupyter notebook
```

Hoặc:

```powershell
python -m notebook
```

Trình duyệt sẽ mở giao diện Jupyter. Sau đó:

1. đi tới thư mục dự án
2. mở `VietNamNet_News_Classification.ipynb`
3. chọn đúng kernel nếu được hỏi

## 13. Chạy notebook đúng cách

Nếu bạn là người mới, hãy làm theo đúng thứ tự sau:

### Bước 1. Mở notebook

Mở `VietNamNet_News_Classification.ipynb`.

### Bước 2. Chọn kernel đúng

Đảm bảo notebook dùng đúng môi trường Python đã cài thư viện.

### Bước 3. Chạy lần lượt từ trên xuống

Không nên chạy nhảy qua cell khác nếu bạn chưa hiểu phụ thuộc giữa các cell.

### Bước 4. Chờ từng bước hoàn thành

Một số cell có thể chạy lâu, đặc biệt là:

- crawl dữ liệu
- tiền xử lý
- tokenize
- huấn luyện PhoBERT

### Bước 5. Quan sát các thư mục đầu ra

Trong quá trình chạy, bạn sẽ thấy các thư mục:

- `dataset`
- `temp`
- `model`
- `result`

được tạo mới hoặc cập nhật.

## 14. Notebook chạy theo những phần nào?

Luồng tổng quát của notebook gồm:

1. chuẩn bị môi trường
2. crawl danh sách URL
3. crawl tiêu đề và nội dung
4. đọc dữ liệu thô
5. khám phá dữ liệu
6. tiền xử lý văn bản
7. tokenize và chia train/test
8. huấn luyện mô hình PhoBERT
9. đánh giá mô hình
10. xuất cấu hình mô hình
11. hiệu chỉnh ngưỡng dự đoán
12. chẩn đoán mô hình

Bạn chỉ cần chạy từ trên xuống, notebook sẽ tự đi theo đúng pipeline.

## 15. Ý nghĩa của 4 thư mục chính

### `dataset`

Chứa dữ liệu gốc và dữ liệu crawl được.

Ví dụ:

- `data_URLs.json`
- các file parquet theo chuyên mục

### `temp`

Chứa cache trung gian để tiết kiệm thời gian chạy lại.

Ví dụ:

- `processed_data.pkl`
- `headtail_train.pkl`
- `headtail_test.pkl`

### `model`

Chứa mô hình và các tệp cấu hình sau khi huấn luyện.

Ví dụ:

- `model.safetensors`
- `config.json`
- `thresholds.json`
- `label_config.json`
- `train_history.pkl`

### `result`

Chứa kết quả đánh giá và các hình ảnh báo cáo.

Ví dụ:

- confusion matrix
- biểu đồ phân phối lớp
- báo cáo phân loại

## 16. Tại sao có lúc notebook không chạy lại một số bước?

Điều này là bình thường.

Notebook có cơ chế kiểm tra file đã tồn tại. Nếu đã có kết quả cũ, notebook có thể bỏ qua một số bước để tiết kiệm thời gian.

Ví dụ:

- đã có URL thì không crawl lại URL
- đã có dữ liệu parquet thì không crawl lại nội dung
- đã có file trung gian thì không xử lý lại
- đã có mô hình thì không train lại
- đã có file ngưỡng thì không hiệu chỉnh lại

Nếu bạn thấy một bước bị bỏ qua, chưa chắc đó là lỗi. Có thể notebook đang tận dụng cache đúng như thiết kế.

## 17. Khi nào cần xóa cache và chạy lại từ đầu?

Bạn nên làm vậy khi:

- muốn thu thập dữ liệu mới hoàn toàn
- muốn train lại toàn bộ mô hình
- muốn kiểm tra pipeline trong trạng thái sạch
- nghi ngờ cache cũ làm kết quả không đúng mong đợi

### Cách chạy lại toàn bộ từ đầu

Xóa các thư mục:

- `dataset`
- `temp`
- `model`
- `result`

Sau đó mở lại notebook và chạy từ cell đầu tiên.

### Cách chỉ huấn luyện lại mô hình

Nếu muốn giữ dữ liệu cũ nhưng train lại mô hình, thường bạn chỉ cần xóa:

- `temp`
- `model`
- `result`

## 18. Làm sao biết notebook đã chạy thành công?

Bạn có thể kiểm tra theo 3 dấu hiệu:

### Dấu hiệu 1

Notebook chạy hết các cell cần thiết mà không dừng ở lỗi đỏ.

### Dấu hiệu 2

Các thư mục đầu ra có thêm file mới.

### Dấu hiệu 3

Bạn thấy các tệp như:

- file mô hình trong `model`
- báo cáo trong `result`
- dữ liệu trong `dataset`

## 19. Các lỗi thường gặp và cách xử lý

### 19.1. Lỗi `ModuleNotFoundError`

Nguyên nhân:

- thiếu thư viện

Cách xử lý:

```powershell
pip install <ten-thu-vien>
```

hoặc cài lại toàn bộ gói ở phần cài thư viện.

### 19.2. Đã cài thư viện nhưng notebook vẫn báo thiếu

Nguyên nhân thường gặp:

- notebook đang dùng sai kernel

Cách xử lý:

- chọn lại đúng kernel đã gắn với môi trường ảo

### 19.3. Lỗi quyền chạy script khi kích hoạt `.venv`

Cách xử lý:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

rồi kích hoạt lại môi trường ảo.

### 19.4. Lỗi tiếng Việt hiển thị sai

Cách xử lý:

- đảm bảo file được mở bằng UTF-8
- trong VS Code, kiểm tra encoding ở góc phải dưới
- không lưu lại file bằng trình soạn thảo cũ dễ gây lỗi mã hóa

### 19.5. Chạy quá chậm

Nguyên nhân có thể là:

- máy không có GPU
- dữ liệu lớn
- đang crawl mạng
- đang train PhoBERT trên CPU

Cách xử lý:

- kiên nhẫn chờ thêm
- kiểm tra xem cell còn đang chạy không
- nếu cần, thử lại với dữ liệu hoặc môi trường phù hợp hơn

## 20. Nếu bạn chỉ muốn chạy nhanh nhất có thể

Làm tối giản như sau:

1. mở terminal tại thư mục dự án
2. tạo `.venv`
3. kích hoạt `.venv`
4. cài thư viện
5. mở notebook
6. chọn đúng kernel
7. chạy từ trên xuống

## 21. Trình tự khuyến nghị cho người mới

Nếu bạn muốn ít rủi ro nhất, hãy làm theo đúng danh sách này:

1. Đọc `README.md` để hiểu tổng quan
2. Mở terminal tại thư mục dự án
3. Chạy `python --version`
4. Tạo môi trường ảo
5. Kích hoạt môi trường ảo
6. Cài thư viện cần thiết
7. Cài kernel cho notebook
8. Mở `VietNamNet_News_Classification.ipynb`
9. Chọn đúng kernel
10. Chạy lần lượt từ trên xuống

## 22. Ghi chú cuối cùng

Nếu đây là lần đầu bạn làm việc với notebook học máy, đừng cố sửa code ngay từ đầu. Cách tốt nhất là:

- chạy thành công một lần
- quan sát các thư mục đầu ra
- hiểu từng phần của pipeline
- sau đó mới bắt đầu tinh chỉnh

Như vậy bạn sẽ dễ phân biệt đâu là lỗi môi trường, đâu là lỗi dữ liệu và đâu là lỗi logic.
