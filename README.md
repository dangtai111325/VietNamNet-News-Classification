# VietNamNet News Classification

## 1. Đây là dự án gì?

Đây là dự án phân loại tin tức tiếng Việt theo **19 chuyên mục** bằng mô hình **PhoBERT**.

Toàn bộ quy trình được gom vào một notebook duy nhất:

- `VietNamNet_News_Classification.ipynb`

Notebook này được thiết kế để chạy theo đúng trình tự thực tế:

1. Thu thập danh sách bài viết từ VietNamNet
2. Crawl tiêu đề và nội dung
3. Làm sạch và chuẩn hóa dữ liệu
4. Tiền xử lý văn bản
5. Tokenize dữ liệu cho PhoBERT
6. Huấn luyện mô hình
7. Đánh giá mô hình
8. Hiệu chỉnh ngưỡng dự đoán
9. Lưu mô hình, cache và kết quả ra thư mục

Nếu bạn là người mới, bạn có thể hiểu ngắn gọn như sau:

- `dataset` là nơi chứa dữ liệu tin tức
- `temp` là nơi chứa các file trung gian để tránh xử lý lại từ đầu
- `model` là nơi chứa mô hình sau khi huấn luyện
- `result` là nơi chứa hình ảnh và báo cáo đánh giá

Notebook đã giữ lại logic xử lý sẵn có, bao gồm cả cơ chế kiểm tra cache để:

- không crawl lại nếu dữ liệu URL đã tồn tại
- không crawl lại nội dung nếu file dữ liệu đã có
- không tokenize lại nếu cache xử lý trung gian đã có
- không huấn luyện lại nếu mô hình đã tồn tại
- không hiệu chỉnh lại ngưỡng nếu file cấu hình ngưỡng đã có

Nói cách khác, đây là một notebook có thể chạy toàn bộ pipeline, nhưng cũng biết cách tận dụng kết quả cũ để tiết kiệm thời gian.

### Tải bản đầy đủ của project

Nếu bạn không muốn chạy lại toàn bộ pipeline từ đầu, hoặc chỉ muốn mở project để xem ngay dữ liệu, cache, mô hình và kết quả đã sinh sẵn, bạn có thể tải bản đầy đủ của project tại đây:

- `https://drive.google.com/drive/folders/14-C9U6N_7IOZmfA17Uf_1IzTO0v08bhQ?usp=drive_link`

Sau khi tải về, hãy giữ nguyên cấu trúc thư mục của project để notebook nhận đúng các thư mục `dataset`, `temp`, `model` và `result`.
### Repo GitHub hiện đang chứa gì?

Bản trên GitHub được giữ ở mức gọn để dễ tải và dễ theo dõi lịch sử thay đổi. Hiện tại repo này có:

- notebook chính `VietNamNet_News_Classification.ipynb`
- tài liệu `README.md` và `QUICK_START.md`
- thư mục `result/` chứa hình ảnh và báo cáo đánh giá
- thư mục `dataset/` nhưng chỉ giữ lại các file JSON cần thiết, ví dụ danh sách URL và file debug

Những phần **không** được đưa lên GitHub:

- các file `*.parquet`
- thư mục `temp/`
- thư mục `model/`

Vì vậy, nếu bạn clone repo từ GitHub thì đó là bản code và tài liệu để đọc, chạy và tái tạo pipeline. Nếu bạn muốn có ngay dữ liệu đầy đủ, cache trung gian và mô hình đã huấn luyện để không phải chạy lại từ đầu, hãy dùng bản đầy đủ từ Google Drive ở trên.

## 2. Dự án này phù hợp với ai?

Tài liệu này được viết cho:

- người mới học về xử lý ngôn ngữ tự nhiên
- người mới dùng Jupyter Notebook hoặc VS Code để chạy notebook
- người muốn hiểu toàn bộ luồng từ thu thập dữ liệu đến huấn luyện mô hình
- người cần một project tương đối gọn để trình bày đồ án, bài tập lớn hoặc nghiên cứu nhỏ

Bạn không cần hiểu hết toàn bộ machine learning ngay từ đầu. Chỉ cần đi đúng thứ tự trong tài liệu và chạy notebook từ trên xuống là có thể bắt đầu.

## 3. Mục tiêu của notebook

Notebook này có 3 mục tiêu chính:

### 3.1. Thu thập dữ liệu tin tức

Notebook crawl dữ liệu từ VietNamNet theo nhiều chuyên mục khác nhau và lưu kết quả xuống thư mục `dataset`.

### 3.2. Huấn luyện mô hình phân loại

Notebook sử dụng PhoBERT để học cách phân biệt bài viết thuộc chuyên mục nào.

### 3.3. Đánh giá và lưu kết quả

Sau khi huấn luyện, notebook sẽ:

- tính các chỉ số đánh giá
- vẽ biểu đồ
- lưu báo cáo
- lưu mô hình để dùng lại

## 4. Các chuyên mục trong bài toán

Bài toán hiện sử dụng 19 chuyên mục tin tức. Tên chuyên mục trong dữ liệu có thể xuất hiện ở dạng slug không dấu để thuận tiện cho lưu tệp. Ví dụ:

- `thoi-su`
- `the-gioi`
- `kinh-doanh`
- `cong-nghe`
- `giao-duc`
- `suc-khoe`
- `the-thao`
- `phap-luat`
- `du-lich`
- `doi-song`
- `oto-xe-may`
- `bat-dong-san`
- `bao-ve-nguoi-tieu-dung`
- `thi-truong-tieu-dung`
- `van-hoa-giai-tri`
- `dan-toc-ton-giao`
- `chinh-tri`
- `ban-doc`
- `tuan-viet-nam`

Bạn không cần tự tạo nhãn bằng tay nếu notebook đã có sẵn logic xử lý dữ liệu và nhãn bên trong.

## 5. Cấu trúc thư mục của dự án

Sau khi chạy notebook, dự án thường có cấu trúc như sau:

```text
VietNamNet_News_Classification/
├─ dataset/
├─ model/
├─ result/
├─ temp/
├─ QUICK_START.md
├─ README.md
└─ VietNamNet_News_Classification.ipynb
```

Lưu ý:

- cấu trúc trên là cấu trúc đầy đủ sau khi bạn chạy pipeline hoặc tải bản full project từ Google Drive
- nếu bạn tải project trực tiếp từ GitHub, bạn sẽ không thấy đầy đủ toàn bộ dữ liệu nặng và model đã huấn luyện
- trên GitHub hiện chỉ giữ `dataset/*.json` và `result/`, còn `*.parquet`, `temp/`, `model/` được loại trừ để repo gọn hơn

Ý nghĩa của từng phần:

### 5.1. `VietNamNet_News_Classification.ipynb`

Đây là file quan trọng nhất của dự án. Bạn chỉ cần mở file này để đọc và chạy toàn bộ pipeline.

### 5.2. `dataset/`

Thư mục này chứa dữ liệu crawl được, ví dụ:

- danh sách URL bài viết
- dữ liệu parquet theo từng chuyên mục
- các file ghi nhận URL lỗi hoặc dữ liệu thiếu

Trong repo GitHub hiện tại, thư mục này chỉ giữ lại các file JSON cần thiết. Các file `parquet` không được push lên GitHub.

Nếu bạn chưa có dữ liệu đầy đủ, notebook sẽ tự tạo hoặc dùng lại thư mục này trong quá trình chạy, tùy theo trạng thái dữ liệu và cache.

### 5.3. `temp/`

Thư mục này chứa các file trung gian để tiết kiệm thời gian xử lý. Ví dụ:

- dữ liệu đã tiền xử lý
- dữ liệu train và test đã tokenize

Nếu có cache trong thư mục này, notebook có thể bỏ qua một số bước nặng để chạy nhanh hơn.

Thư mục `temp/` không được đưa lên GitHub.

### 5.4. `model/`

Thư mục này chứa mô hình sau khi huấn luyện, ví dụ:

- trọng số mô hình
- tokenizer
- cấu hình nhãn
- lịch sử huấn luyện
- file ngưỡng dự đoán

Nếu mô hình đã tồn tại, notebook có thể dùng lại thay vì huấn luyện từ đầu.

Thư mục `model/` không được đưa lên GitHub.

### 5.5. `result/`

Thư mục này chứa kết quả đầu ra phục vụ phân tích và báo cáo, ví dụ:

- biểu đồ phân phối lớp
- biểu đồ độ dài văn bản
- confusion matrix
- biểu đồ F1 theo lớp
- đường cong huấn luyện
- báo cáo phân loại dạng văn bản

Thư mục `result/` hiện vẫn được đưa lên GitHub để bạn có thể xem nhanh kết quả đánh giá mà không cần chạy notebook trước.

## 6. Notebook gồm những phần nào?

Notebook được chia thành các phần lớn để bạn dễ theo dõi:

### 6.1. Phần 1 - Thu thập dữ liệu từ VietNamNet

Phần này tập trung vào dữ liệu đầu vào.

Các công việc chính:

- chuẩn bị môi trường crawl
- crawl danh sách URL
- crawl tiêu đề và nội dung bài viết
- lưu dữ liệu vào `dataset`

### 6.2. Phần 2 - Xây dựng mô hình PhoBERT

Phần này là lõi của bài toán phân loại.

Các công việc chính:

- đọc dữ liệu thô
- khám phá dữ liệu
- tiền xử lý văn bản
- tokenize theo chiến lược đã có trong notebook
- chia train và test
- huấn luyện mô hình
- đánh giá kết quả
- xuất mô hình
- hiệu chỉnh ngưỡng dự đoán
- chẩn đoán lỗi mô hình

## 7. Cơ chế cache và tại sao bạn cần hiểu phần này

Người mới thường gặp một nhầm lẫn: tại sao chạy notebook nhưng có bước chạy rất nhanh, có bước gần như bị bỏ qua?

Lý do là notebook có logic tận dụng file có sẵn. Điều này là bình thường và là chủ đích.

Ví dụ:

- nếu đã có file URL thì không cần crawl lại danh sách URL
- nếu đã có dữ liệu parquet thì không cần crawl lại nội dung
- nếu đã có file xử lý trung gian thì không cần xử lý lại
- nếu đã có mô hình thì có thể bỏ qua train
- nếu đã có file ngưỡng thì có thể bỏ qua hiệu chỉnh lại

Điều này giúp:

- tiết kiệm thời gian
- tránh lặp lại bước nặng
- dễ tiếp tục công việc nếu notebook bị dừng giữa chừng

## 8. Khi nào nên chạy lại từ đầu?

Bạn nên chạy lại từ đầu khi:

- muốn thu thập dữ liệu mới hoàn toàn
- muốn tạo lại toàn bộ cache
- muốn huấn luyện lại mô hình từ đầu
- muốn kiểm tra lại toàn bộ pipeline trong môi trường sạch

Trong trường hợp đó, bạn có thể xóa các thư mục sau:

- `dataset`
- `temp`
- `model`
- `result`

Sau đó mở notebook và chạy lại từ ô đầu tiên.

Nếu bạn chỉ muốn huấn luyện lại mô hình mà vẫn dùng dữ liệu cũ, bạn thường chỉ cần xóa:

- `model`
- `temp`
- `result`

Việc xóa thư mục nào là quyết định kỹ thuật quan trọng, vì nó ảnh hưởng trực tiếp đến thời gian chạy.

## 9. Cần chuẩn bị gì trước khi chạy?

Bạn cần có:

- máy tính chạy Windows
- Python đã cài sẵn
- Jupyter Notebook hoặc VS Code có hỗ trợ notebook
- các thư viện Python cần thiết
- kết nối mạng nếu muốn crawl dữ liệu mới

Nếu bạn chỉ muốn đọc notebook thì không cần cài đầy đủ mọi thứ. Nhưng nếu muốn chạy toàn bộ pipeline, bạn cần môi trường Python hoạt động bình thường.

Phần hướng dẫn thiết lập cụ thể đã được gộp vào file `QUICK_START.md`.

## 10. Cách đọc notebook nếu bạn là người mới hoàn toàn

Đây là cách đọc dễ nhất:

### Bước 1. Đọc phần tiêu đề và mô tả đầu notebook

Mục tiêu là hiểu notebook làm gì, đầu ra là gì, và các thư mục nào sẽ được tạo.

### Bước 2. Chạy lần lượt từ trên xuống

Không nên chạy nhảy cóc nếu bạn chưa hiểu rõ phụ thuộc giữa các ô.

### Bước 3. Quan sát thư mục sinh ra sau mỗi giai đoạn

Ví dụ:

- sau phần crawl, thư mục `dataset` sẽ thay đổi
- sau phần tokenize, thư mục `temp` sẽ có thêm file cache
- sau phần train, thư mục `model` sẽ có mô hình
- sau phần evaluate, thư mục `result` sẽ có biểu đồ và báo cáo

### Bước 4. Đọc log trong từng ô

Log cho biết notebook đang làm gì:

- đang đọc dữ liệu
- đang bỏ qua bước nào vì đã có cache
- đang train mô hình
- đang đánh giá

Nếu notebook dừng vì lỗi, log thường là nơi đầu tiên bạn cần xem.

## 11. Kết quả bạn sẽ nhận được sau khi chạy

Tùy theo trạng thái dữ liệu và cache, sau khi chạy thành công bạn thường sẽ có:

- dữ liệu tin tức đã crawl trong `dataset`
- dữ liệu trung gian trong `temp`
- mô hình PhoBERT trong `model`
- các biểu đồ và báo cáo trong `result`

Ví dụ một số tệp đầu ra thường gặp:

- `model.safetensors`
- `label_config.json`
- `train_history.pkl`
- `thresholds.json`
- `classification_report.txt`
- các hình `.png` trong thư mục `result`

## 12. Trường hợp nào notebook có thể chạy lâu?

Một số bước có thể mất thời gian đáng kể:

- crawl dữ liệu
- tokenize dữ liệu lớn
- huấn luyện PhoBERT
- đánh giá và sinh biểu đồ

Nếu máy không có GPU, bước huấn luyện có thể chậm hơn nhiều.

Điều này không có nghĩa là notebook bị lỗi. Hãy quan sát ô đang chạy và log đi kèm.

## 13. Những lỗi người mới hay gặp

### 13.1. Thiếu thư viện

Dấu hiệu:

- notebook báo không import được một thư viện nào đó

Cách xử lý:

- cài thư viện còn thiếu theo hướng dẫn trong `QUICK_START.md`

### 13.2. Sai môi trường Python

Dấu hiệu:

- bạn đã cài thư viện nhưng notebook vẫn báo thiếu

Cách xử lý:

- kiểm tra xem notebook đang dùng đúng kernel Python hay chưa
- nếu dùng VS Code, chọn lại kernel tương ứng với môi trường đã cài thư viện

### 13.3. Lỗi do cache cũ

Dấu hiệu:

- dữ liệu hoặc mô hình cũ làm kết quả không giống mong đợi

Cách xử lý:

- xóa thư mục cache phù hợp như `temp`, `model`, `result`, hoặc `dataset` nếu cần chạy lại hoàn toàn

### 13.4. Lỗi tiếng Việt hiển thị sai

Dấu hiệu:

- chữ tiếng Việt bị vỡ dấu, sai dấu hoặc hiện ký tự lạ

Cách xử lý:

- mở file bằng UTF-8
- trong VS Code, kiểm tra góc phải dưới và bảo đảm encoding là `UTF-8`
- tránh mở và lưu lại file bằng phần mềm cũ không hỗ trợ UTF-8 đúng cách

## 14. Khuyến nghị chạy cho người mới

Nếu đây là lần đầu bạn chạy dự án, cách an toàn nhất là:

1. Đọc hết `QUICK_START.md`
2. Mở `VietNamNet_News_Classification.ipynb`
3. Chọn đúng Python kernel
4. Chạy từng ô từ trên xuống
5. Không sửa code ngay trong lần chạy đầu tiên
6. Chỉ bắt đầu chỉnh sửa khi bạn đã hiểu thư mục nào sinh ra file gì

## 15. Nên đọc file nào trước?

Thứ tự khuyến nghị:

1. `README.md`
2. `QUICK_START.md`
3. `VietNamNet_News_Classification.ipynb`

Nếu bạn chỉ muốn chạy nhanh, có thể đọc ngay `QUICK_START.md`.

## 16. Tóm tắt ngắn gọn

Nếu bạn chỉ cần nắm ý chính:

- dự án này dùng một notebook duy nhất
- notebook làm từ crawl dữ liệu đến train và đánh giá mô hình
- dữ liệu, cache, mô hình và kết quả được tách thành 4 thư mục rõ ràng
- notebook giữ lại logic dùng cache để tránh chạy lại từ đầu
- người mới nên chạy theo thứ tự từ trên xuống và không bỏ qua phần chuẩn bị môi trường

## 17. Tài liệu đi kèm

Trong thư mục gốc hiện chỉ còn 2 file hướng dẫn:

- `README.md`: giải thích tổng thể, dành cho người mới đọc để hiểu dự án
- `QUICK_START.md`: hướng dẫn cài đặt và chạy thật chi tiết theo từng bước

Nếu bạn cần bắt đầu ngay, hãy mở `QUICK_START.md`.

