# Crawling Data

Tài liệu này giải thích chi tiết notebook [crawl_data.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/Crawling%20Data/crawl_data.ipynb) theo cách dành cho người chưa biết gì về project.

Notebook này có nhiệm vụ thu thập dữ liệu bài viết từ VietNamNet để tạo dataset huấn luyện cho các mô hình ở các thư mục `SVM/` và `PhoBERT/`.

## Mục tiêu của notebook

Notebook thực hiện 2 việc chính:

1. Crawl danh sách URL bài viết cho từng chuyên mục.
2. Từ các URL đó, crawl `title` và `content` của từng bài, sau đó lưu ra file `.parquet`.

Ngoài ra notebook còn có bước kiểm tra chất lượng dữ liệu sau khi crawl xong.

Kết quả cuối cùng được lưu trong thư mục [`Dataset`](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/Dataset).

## Notebook tạo ra những file gì

Sau khi chạy xong, bạn sẽ thấy các file sau:

- `Dataset/data_URLs.json`
  Lưu danh sách URL bài viết theo từng chuyên mục.
- `Dataset/<category>.parquet`
  Mỗi chuyên mục có một file parquet riêng, chứa dữ liệu bài viết đã crawl.
- `Dataset/data_URLs_empty_title.json`
  Danh sách URL của các bài bị thiếu `title`.
- `Dataset/data_URLs_empty_content.json`
  Danh sách URL của các bài bị thiếu `content`.

Ví dụ:

- `Dataset/chinh-tri.parquet`
- `Dataset/the-thao.parquet`
- `Dataset/cong-nghe.parquet`

## Cấu trúc dữ liệu đầu ra

Mỗi file parquet được lưu theo schema cố định:

- `class`
  Tên chuyên mục, ví dụ `chinh-tri`, `the-thao`.
- `url`
  URL bài viết gốc trên VietNamNet.
- `title`
  Tiêu đề bài viết.
- `content`
  Nội dung bài viết.

Điểm quan trọng:

- `class` là slug không dấu, dùng để map sang nhãn ở notebook train.
- `title` và `content` có thể rỗng nếu bài viết không trích xuất được hoàn toàn.
- các bài thiếu cả `title` lẫn `content` sẽ bị loại ở bước train, không loại ngay tại notebook crawl.

## 19 chuyên mục được crawl

Notebook hiện crawl 19 chuyên mục:

- `chinh-tri`
- `thoi-su`
- `kinh-doanh`
- `dan-toc-ton-giao`
- `the-thao`
- `giao-duc`
- `the-gioi`
- `doi-song`
- `van-hoa-giai-tri`
- `suc-khoe`
- `cong-nghe`
- `phap-luat`
- `oto-xe-may`
- `du-lich`
- `bat-dong-san`
- `ban-doc`
- `tuan-viet-nam`
- `bao-ve-nguoi-tieu-dung`
- `thi-truong-tieu-dung`

## Luồng hoạt động tổng quát

Notebook có 4 section thực tế:

1. Chuẩn bị môi trường và cấu hình.
2. Crawl URL bài viết cho từng chuyên mục.
3. Crawl `title` và `content` từ các URL.
4. Kiểm tra chất lượng dữ liệu sau khi crawl.

Luồng chạy đầy đủ:

1. Kiểm tra thư viện cần thiết.
2. Tạo thư mục `Dataset/` nếu chưa có.
3. Tạo hoặc sửa `data_URLs.json` nếu file chưa tồn tại hoặc bị lỗi.
4. Crawl URL cho các chuyên mục còn thiếu.
5. Crawl nội dung cho các chuyên mục chưa có file parquet.
6. Kiểm tra số lượng bài, số bài thiếu `title`, thiếu `content`, thiếu cả hai.
7. Xuất thêm 2 file JSON debug để xem các URL lỗi.

## Cơ chế “thông minh” giúp không phải crawl lại từ đầu

Đây là phần quan trọng nhất của notebook.

Notebook không mặc định làm lại toàn bộ từ đầu. Nó có logic bỏ qua các phần đã có dữ liệu:

### 1. Section crawl URL

Notebook chỉ crawl URL cho những chuyên mục:

- chưa có trong `data_URLs.json`
- hoặc có key nhưng danh sách URL đang rỗng

Những chuyên mục đã có URL sẽ được giữ nguyên.

### 2. Section crawl content

Notebook chỉ crawl nội dung cho những chuyên mục:

- chưa có file `Dataset/<category>.parquet`

Nếu file parquet đã tồn tại, notebook xem như chuyên mục đó đã crawl xong và bỏ qua.

### 3. Tự crawl URL nếu thiếu

Trong bước crawl content, nếu một chuyên mục chưa có URL trong `all_urls`, notebook sẽ:

1. crawl URL trước
2. cập nhật lại `data_URLs.json`
3. rồi mới crawl `title` và `content`

Nghĩa là bạn không nhất thiết phải chạy Section 2 riêng. Section 3 có thể tự bù phần URL còn thiếu.

## Giải thích từng section

## Section 1: Chuẩn bị

Phần này làm 3 việc:

### 1. Kiểm tra thư viện

Notebook kiểm tra các thư viện:

- `requests`
- `tqdm`
- `beautifulsoup4`
- `lxml`
- `pyarrow`
- `aiohttp`
- `newspaper3k`
- `trafilatura`
- `readability-lxml`
- `goose3`

Nếu thiếu thư viện nào, notebook sẽ in ra lệnh `pip install ...` và dừng.

### 2. Thiết lập cấu hình

Ba biến cấu hình chính:

- `MAX_PAGES = 500`
  Số trang tối đa dùng để dò URL cho mỗi chuyên mục.
- `BATCH_SIZE = 100`
  Số bài được ghi xuống parquet mỗi lần.
- `NUM_WORKERS = 32`
  Số worker đồng thời khi crawl nội dung bằng `aiohttp`.

Ý nghĩa thực tế:

- tăng `MAX_PAGES` thì có thể lấy được nhiều URL hơn, nhưng crawl URL lâu hơn.
- tăng `BATCH_SIZE` thì ghi file ít lần hơn, nhưng giữ dữ liệu trong RAM lâu hơn.
- tăng `NUM_WORKERS` thì crawl nhanh hơn, nhưng cũng tăng tải mạng và tải server.

### 3. Chuẩn bị thư mục và file

Notebook tự:

- tạo thư mục `Dataset/` nếu chưa tồn tại
- tạo `data_URLs.json` nếu chưa có
- reset `data_URLs.json` về `{}` nếu file đang rỗng hoặc hỏng JSON

Điều này giúp notebook ít bị lỗi khi chạy lần đầu hoặc khi file bị hỏng giữa chừng.

## Section 2: Crawl URL

Mục tiêu của phần này là lấy danh sách URL bài viết cho từng chuyên mục.

### URL trang chuyên mục được xây như thế nào

Ví dụ với category `ban-doc`:

- trang đầu:
  `https://vietnamnet.vn/ban-doc`
- các trang tiếp theo:
  `https://vietnamnet.vn/ban-doc-page1`
  `https://vietnamnet.vn/ban-doc-page2`
  ...

Notebook dùng hàm `get_urls_of_category(category, max_pages=MAX_PAGES)`.

### Cách notebook lấy URL bài viết

Cho mỗi trang chuyên mục:

1. gửi request bằng `requests.Session()`
2. parse HTML bằng `BeautifulSoup`
3. tìm các phần tử có class `vnn-title`
4. lấy thẻ `a` bên trong để trích `href`
5. nếu `href` là dạng tương đối như `/the-thao/...` thì nối thêm domain `https://vietnamnet.vn`
6. thêm URL vào danh sách

### Khi nào dừng crawl URL

Notebook dừng khi:

- request lỗi
- hoặc trang không còn bài viết, tức không còn phần tử `vnn-title`

Điều này giúp không cần biết chính xác một chuyên mục có bao nhiêu trang.

### Kết quả của Section 2

Sau mỗi chuyên mục, notebook in:

- số URL đã tìm được
- thời gian crawl của chuyên mục đó

Cuối cùng:

- ghi tất cả vào `Dataset/data_URLs.json`
- in tổng thời gian crawl URL

## Section 3: Crawl title và content

Đây là phần nặng nhất và quan trọng nhất.

Mục tiêu:

- đọc từng URL bài viết
- trích `title` và `content`
- lưu dần ra `parquet`

### Vì sao phần này dùng async

Notebook dùng:

- `aiohttp`
- `asyncio`
- `Semaphore(NUM_WORKERS)`

Mục tiêu là gửi nhiều request cùng lúc để tăng tốc crawl.

Mỗi URL được xử lý bất đồng bộ, nhưng vẫn bị giới hạn số lượng đồng thời bởi `NUM_WORKERS`.

### Cách notebook crawl nội dung của một chuyên mục

Hàm chính là `_crawl_category_async(category, urls)`.

Cho mỗi category:

1. tạo `aiohttp.ClientSession`
2. tạo task cho từng URL
3. chạy các task bằng `asyncio.as_completed`
4. bài nào xong trước xử lý trước
5. gom kết quả vào `batch`
6. đủ `BATCH_SIZE` thì ghi xuống file parquet
7. cuối cùng flush phần còn lại

Ưu điểm:

- tốc độ nhanh hơn crawl tuần tự
- không phải giữ toàn bộ dữ liệu trong RAM đến cuối
- nếu notebook dừng giữa chừng, thường chỉ mất phần batch đang xử lý, không mất toàn bộ

## Cơ chế trích xuất nội dung: 5 lớp fallback

Đây là phần tinh vi nhất của notebook.

Một bài viết không được trích xuất bằng đúng 1 thư viện duy nhất. Notebook thử theo thứ tự ưu tiên:

### 1. BeautifulSoup với selector chuyên biệt cho VietNamNet

Ưu tiên cao nhất vì:

- đúng cấu trúc site
- nhanh
- ít noise nhất nếu selector còn đúng

Notebook thử các selector cho title:

- `h1.content-detail-title`
- `h1.ArticleTitle`
- `h1.title-detail`
- `h1.title`
- `h1`

Selector cho nội dung:

- `div.ArticleContent`
- `div.content-detail-body`
- `div.maincontent`
- `div.main-content`
- `article`
- `main`

Nếu không tìm được khối nội dung rõ ràng, notebook fallback sang gom tất cả thẻ `p` dài hơn ngưỡng tối thiểu.

### 2. trafilatura

Đây là extractor khá mạnh cho nội dung bài báo.

Notebook dùng `trafilatura` để:

- trích text chính
- lấy metadata title nếu cần

`trafilatura` phù hợp khi HTML còn sạch nhưng selector riêng của site không còn hoạt động tốt.

### 3. readability-lxml

Thư viện này hoạt động theo ý tưởng giống Reader Mode của Firefox.

Notebook dùng nó để:

- lấy title
- lấy phần summary/article body
- rồi parse tiếp bằng `BeautifulSoup` để gom các đoạn `p`

### 4. goose3

`goose3` là extractor theo kiểu structured article extraction.

Notebook dùng nó như một lớp fallback tiếp theo khi 3 bước trước chưa đủ tốt.

### 5. newspaper3k

Đây là fallback cuối cùng.

Nếu các lớp trước không lấy được đủ dữ liệu, notebook thử `newspaper3k`.

### Vì sao phải dùng nhiều lớp như vậy

Không có extractor nào luôn đúng với mọi URL.

Lý do phải dùng nhiều lớp:

- layout bài viết có thể thay đổi
- một số bài có HTML khác chuẩn
- một số bài có nội dung nhúng, quảng cáo, hoặc cấu trúc đặc biệt
- extractor này thất bại nhưng extractor khác có thể cứu được

Nhờ cơ chế nhiều lớp, tỷ lệ mất dữ liệu sẽ thấp hơn rõ rệt so với chỉ dùng một thư viện.

## Tối ưu quan trọng: chỉ tải HTML một lần

Notebook có cả bản sync và async của hàm trích xuất nội dung.

Điểm tối ưu rất quan trọng:

- HTML của bài viết được tải một lần
- sau đó nhiều extractor dùng chung phần HTML này

Điều này tránh:

- gửi nhiều request lặp lại cho cùng một URL
- chậm không cần thiết
- tăng nguy cơ lỗi mạng

Trong bản async:

- `BeautifulSoup`, `trafilatura`, `readability` tận dụng trực tiếp HTML đã tải
- `goose3` và `newspaper3k` được chạy trong thread executor để tránh block event loop

## Ghi file parquet theo batch

Notebook không gom tất cả dữ liệu rồi mới lưu.

Thay vào đó:

1. tạo `ParquetWriter`
2. giữ tạm một `batch`
3. mỗi khi `batch` đủ `BATCH_SIZE`, gọi `flush_batch`
4. ghi tiếp vào cùng file parquet

Ưu điểm:

- giảm RAM
- ổn định hơn với tập dữ liệu lớn
- phù hợp khi crawl hàng chục nghìn URL

## Section 4: Kiểm tra chất lượng dữ liệu

Sau khi crawl xong, notebook đọc lại toàn bộ file parquet để thống kê chất lượng.

### Các chỉ số được kiểm tra

Cho từng category, notebook tính:

- tổng số bài
- số bài thiếu `title`
- số bài thiếu `content`
- số bài thiếu cả `title` lẫn `content`

Sau đó notebook in:

- bảng thống kê theo từng chuyên mục
- tổng cộng toàn bộ dataset

### Hai file JSON debug

Notebook còn xuất:

- `data_URLs_empty_title.json`
- `data_URLs_empty_content.json`

Mục đích:

- xem nhanh URL nào đang bị lỗi trích xuất
- có thể dùng để debug lại extractor
- hoặc crawl lại một nhóm URL có vấn đề

### Vì sao bước này quan trọng

Crawler không chỉ cần “chạy xong”, mà còn phải biết chất lượng dữ liệu thực tế.

Ví dụ:

- nếu một category có quá nhiều bài thiếu `content`
  có thể selector của site đã thay đổi
- nếu một file parquet cũ không có cột `url`
  notebook sẽ cảnh báo để bạn re-crawl cho dễ debug

## Cách chạy notebook đúng

Nếu bạn chạy lần đầu:

1. mở `crawl_data.ipynb`
2. chạy toàn bộ notebook từ trên xuống dưới
3. chờ Section 2 crawl URL
4. chờ Section 3 crawl content
5. xem Section 4 để kiểm tra chất lượng

Nếu bạn đã có dữ liệu một phần:

1. mở notebook
2. chạy lại toàn bộ
3. notebook sẽ tự bỏ qua phần đã có
4. chỉ crawl tiếp phần còn thiếu

## Cách notebook quyết định phần nào còn thiếu

### Thiếu URL

Một category bị xem là thiếu URL nếu:

- không có key trong `data_URLs.json`
- hoặc danh sách URL đang rỗng

### Thiếu content

Một category bị xem là thiếu content nếu:

- chưa có file `Dataset/<category>.parquet`

## Khi nào nên xóa file cũ để crawl lại

Bạn nên xóa có chọn lọc, không cần xóa toàn bộ dataset.

### Muốn crawl lại URL của một category

Xóa key category đó trong `data_URLs.json`, hoặc sửa nó thành danh sách rỗng.

### Muốn crawl lại content của một category

Xóa file:

- `Dataset/<category>.parquet`

Notebook sẽ nhận ra category đó chưa có parquet và crawl lại.

### Muốn crawl lại toàn bộ

Xóa:

- `Dataset/data_URLs.json`
- toàn bộ các file `Dataset/*.parquet`

Rồi chạy lại notebook từ đầu.

## Ý nghĩa của các hàm chính

### `get_urls_of_category(...)`

Lấy danh sách URL bài viết của một chuyên mục.

### `_fetch_html(...)`

Tải HTML bằng `requests`, dùng cho bản sync.

### `extract_content(...)`

Trích xuất `title` và `content` theo 5 lớp fallback, dùng cho bản sync.

### `_fetch_html_async(...)`

Tải HTML bằng `aiohttp`, dùng cho bản async.

### `extract_content_async(...)`

Phiên bản async của hàm trích nội dung.

### `flush_batch(...)`

Ghi một batch bài viết xuống file parquet.

### `fmt_dur(...)`

Định dạng thời gian theo dạng dễ đọc như `3m 12s` hoặc `1h 4m 7s`.

### `now()`

Lấy giờ hiện tại để in log.

## Ưu điểm của thiết kế hiện tại

- Có thể chạy tiếp phần còn thiếu thay vì làm lại từ đầu.
- Tách URL và content thành 2 bước rõ ràng.
- Dùng nhiều extractor để tăng tỷ lệ lấy được dữ liệu.
- Ghi parquet theo batch nên tiết kiệm RAM.
- Có bước kiểm tra chất lượng sau crawl.
- Có file debug để rà các URL lỗi.

## Hạn chế cần biết

- Crawler phụ thuộc vào cấu trúc HTML hiện tại của VietNamNet.
- Nếu VietNamNet đổi layout, selector `BeautifulSoup` có thể kém hiệu quả.
- Dùng nhiều worker có thể làm tốc độ nhanh hơn nhưng cũng dễ gặp lỗi mạng hơn.
- Một số bài vẫn có thể thiếu `title` hoặc `content` dù đã qua nhiều lớp fallback.
- Notebook không tự retry nhiều lần theo chiến lược phức tạp, nên lỗi mạng ngẫu nhiên vẫn có thể làm mất một phần dữ liệu.

## Khi nào dữ liệu được xem là “đủ tốt” để train

Thông thường bạn có thể chuyển sang notebook train khi:

- cả 19 category đều có file parquet
- số bài thiếu cả `title` và `content` là rất thấp
- các file debug JSON không có quá nhiều URL lỗi bất thường

Một số bài thiếu `title` hoặc thiếu `content` riêng lẻ vẫn có thể chấp nhận được, vì notebook train đã có bước lọc bài thiếu cả hai trường.

## Mối liên hệ với các notebook khác

Dataset tạo ra ở đây sẽ được dùng bởi:

- [SVM/main_SVM.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/SVM/main_SVM.ipynb)
- [PhoBERT/main_PhoBERT.ipynb](C:/Users/DELL/Downloads/HTTM/VietNamNet%20News%20Classification/PhoBERT/main_PhoBERT.ipynb)

Hai notebook train này đều đọc dữ liệu từ thư mục `Dataset/`.

Vì vậy nếu bước crawl sai hoặc thiếu dữ liệu, toàn bộ pipeline phía sau cũng bị ảnh hưởng.

## Tóm tắt ngắn

Nếu chỉ cần hiểu nhanh:

- Section 2 lấy URL bài viết và lưu vào `data_URLs.json`.
- Section 3 lấy `title` và `content`, lưu mỗi category thành một file `.parquet`.
- notebook có cơ chế bỏ qua phần đã crawl xong.
- notebook dùng 5 lớp fallback để tăng khả năng trích xuất nội dung.
- Section 4 kiểm tra chất lượng dữ liệu và xuất file debug.

Nếu bạn muốn, tôi có thể làm tiếp một phiên bản README thứ hai ngắn hơn, kiểu “quick start 1 trang”, còn file này giữ vai trò tài liệu đầy đủ. 
