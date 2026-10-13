# Phân loại chủ đề tin tức tiếng Việt

Bài tập lớn môn **Cách tiếp cận hiện đại trong xử lý ngôn ngữ tự nhiên (CO5087)**, Trường Đại học Bách khoa – ĐHQG TP.HCM, Khoa Khoa học và Kỹ thuật Máy tính.

**Đặng Nguyễn Tấn Tài — MSSV 2570357**, lớp CH01, học kỳ 261 (2026–2027). Giảng viên: Lê Thành Sách, Nguyễn Quốc Minh.

Dự án xây dựng corpus VietNamNet có bằng chứng chuyên mục, phân loại đơn nhãn 20 lớp bằng PhoBERT, BiLSTM, Linear SVM và Logistic Regression, cùng ứng dụng hỗ trợ biên tập dùng PhoBERT. Notebook dataset cung cấp đầu vào chung; mỗi notebook huấn luyện tự chứa quy trình, cache và artifact riêng. Gói model phục vụ ứng dụng, số liệu thực nghiệm PhoBERT và báo cáo sử dụng cùng taxonomy và quy tắc suy luận.

## Mục lục

- [Kết quả và phạm vi](#kết-quả-và-phạm-vi)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Phạm vi repository Git](#phạm-vi-repository-git)
- [Thực thi notebook](#thực-thi-notebook)
- [Thực nghiệm LSTM, SVM và Logistic Regression](#thực-nghiệm-lstm-svm-và-logistic-regression)
- [Dataset và taxonomy](#dataset-và-taxonomy)
- [Mô hình và huấn luyện](#mô-hình-và-huấn-luyện)
- [Hiệu chỉnh và suy luận](#hiệu-chỉnh-và-suy-luận)
- [Chạy ứng dụng](#chạy-ứng-dụng)
- [Database minh họa và sao lưu](#database-minh-họa-và-sao-lưu)
- [Gói model và registry](#gói-model-và-registry)
- [Báo cáo](#báo-cáo)
- [Kiểm chứng](#kiểm-chứng)
- [Xử lý sự cố](#xử-lý-sự-cố)
- [Nguồn dữ liệu và giấy phép](#nguồn-dữ-liệu-và-giấy-phép)

## Kết quả và phạm vi

Dataset sạch theo chuyên mục có **200.337 bài**. Sau khử trùng văn bản và loại nhóm xung đột nhãn, corpus dùng chia tập có **199.965 bài**: train **139.975**, validation **29.995**, test **29.995**. Chia phân tầng 70/15/15, seed 42. Bốn notebook mô hình có output đầy đủ; các chỉ số chia tập giống nhau và toàn bộ URL/nhãn test khớp theo thứ tự.

| Mô hình được chọn bằng validation | Accuracy test | Macro-F1 test | Weighted-F1 test |
|---|---:|---:|---:|
| Linear SVM, TF-IDF unigram/bigram, C=0,5, balanced | 89,9150% | 85,3308% | 89,8771% |
| Logistic Regression, TF-IDF unigram/bigram, C=3,0, balanced | 89,0615% | 84,9797% | 89,0983% |
| BiLSTM, 512 từ, embedding 200, hidden 128 mỗi hướng, α=0,5 | 88,6648% | 84,5995% | 88,7105% |
| PhoBERT-base-v2, Head–Tail 256 subword, argmax | 91,0818% | 87,2191% | 91,1176% |

Đối chứng chính dùng argmax score/logit; temperature dương không đổi nhãn. SVM/LR tìm 12 cấu hình mỗi model, BiLSTM tìm ba cấu hình; bảng chỉ chứa cấu hình được chọn trên validation. SVM/LR dùng toàn văn, bỏ stopword và lặp tiêu đề ba lần; BiLSTM học embedding từ đầu và cắt theo từ, PhoBERT có pretraining và cắt theo subword. Đây là so sánh pipeline, không phải ablation cô lập kiến trúc.

Trạng thái PhoBERT phục vụ ứng dụng có thêm hệ số điều chỉnh lớp:

| Chỉ số test | Softmax chuẩn | Temperature và hệ số class |
|---|---:|---:|
| Accuracy | 91,0818% | **91,1252%** |
| Weighted-F1 | 91,1176% | **91,1450%** |
| Macro-F1 | 87,2191% | **87,2529%** |

Temperature **1,45511457** được fit bằng validation. Trên test, temperature làm NLL giảm từ 0,383892 xuống 0,322030 và ECE 15 bins giảm từ 0,060291 xuống 0,025071. ECE này đo xác suất cực đại và argmax của phân bố temperature; không phải ECE của quyết định sau hệ số class.

Hiệu chỉnh quyết định sửa đúng 112 bài và làm sai thêm 99 bài, cải thiện ròng 13 bài. Nông thôn mới có F1 0,4639 trên 146 mẫu test; Dân tộc – Tôn giáo 0,6759 trên 555 mẫu; Dân tộc thiểu số và miền núi 0,6877 trên 163 mẫu. Chênh lệch này cần được xét cùng chỉ số tổng.

Thực nghiệm hoàn thành bảy epoch trên một RTX A3000 Laptop 12 GB trong 12.429,91 giây, khoảng 3 giờ 27 phút. Checkpoint tốt nhất theo Macro-F1 validation ở epoch 7. Loss validation thấp nhất ở epoch 3 rồi tăng; lựa chọn checkpoint theo F1 và theo loss có thể khác nhau.

BiLSTM và LR đạt F1 Nông thôn mới 48,24% và 47,06%, so với PhoBERT argmax 45,57%; điểm tổng cao nhất không bảo đảm tốt nhất mọi lớp. Fit SVM được chọn mất 27,32 giây trên CPU, BiLSTM 29,66 phút và PhoBERT 207,17 phút trên GPU; tổng tìm cấu hình và phạm vi timer được phân biệt trong báo cáo. Bootstrap ghép cặp từ dự đoán đã xuất cho thấy khoảng chênh lệch PhoBERT–SVM nằm trên 0 với model cố định, còn khoảng SVM–LR và LR–BiLSTM chứa 0. Một seed, một nguồn và thiếu ablation chưa đủ kết luận ưu thế trên mọi corpus hoặc khả năng tổng quát hóa ngoài miền.

## Cấu trúc dự án

```text
BTL_NLP_VietnamNet/
├── README.md                       Tài liệu vận hành và mô tả dự án
├── Dataset_VietNamNet.ipynb         Thu thập, xác minh nhãn, xuất corpus sạch
├── PhoBERT_Classification.ipynb     Fine-tuning, calibration, đánh giá và hình
├── LSTM_Classification.ipynb        BiLSTM, chọn checkpoint bằng validation
├── SVM_Classification.ipynb         TF-IDF và LinearSVC
├── Logistic_Regression.ipynb        TF-IDF và Logistic Regression đa lớp
├── resources/                      Stopword riêng cho SVM và Logistic
├── requirements-notebook.txt       Thư viện thực thi notebook
├── requirements-codegen.txt        Thư viện tạo bindings Protobuf
├── dataset/                        20 Parquet và bốn JSON theo giai đoạn URL
├── temp/                           Cache tiền xử lý, tensor và split manifest
├── model/                          Gói PhoBERT; lstm/, svm/, logistic/ khi chạy
├── result/
│   ├── evaluation/                 JSON, NPZ, Parquet, lịch sử và phân tích lỗi
│   ├── figures/                    69 PNG độc lập 300 dpi
│   ├── lstm/, svm/, logistic/       Đầu ra riêng của từng notebook tương ứng
│   └── classification_report.txt   Báo cáo chỉ số do notebook xuất
├── nlp/                            Hợp đồng model và pipeline suy luận của app
├── app/
│   ├── compose.yaml                Sáu dịch vụ Docker Compose
│   ├── ui/                         React, TypeScript, Vite và lockfile
│   ├── api-service/                FastAPI, phân quyền và nghiệp vụ
│   ├── model-service/              PhoBERT inference qua gRPC
│   ├── worker-service/             Redis–Dramatiq và scheduler
│   ├── artifacts/active.json       Gói đang phục vụ và SHA-256
│   └── demo/postgres.backup        Snapshot database minh họa schema vnn
├── report/
│   ├── CO5087_Report.pdf           Báo cáo học thuật
│   └── source_latex/               XeLaTeX, chương, bảng, ảnh, TikZ, bibliography
└── artifact_manifest.json          Manifest artifact của project
```

`nlp/` là mã runtime cần cho ứng dụng. Các notebook tự chứa mã nguồn và không phụ thuộc các module này. `temp/` là cache thực nghiệm; giữ lại giúp giảm thời gian tiền xử lý và tokenize. Ba notebook LSTM/SVM/Logistic chỉ dùng subfolder của mình trong `temp/`, `model/`, `result/`; không đọc hoặc ghi cache, trọng số, registry và assets báo cáo của PhoBERT. Các `requirements*.txt`, `model/vocab.txt` và `result/classification_report.txt` là file chức năng hoặc kết quả thực nghiệm, không phải tài liệu hướng dẫn riêng.

## Phạm vi repository Git

Mã nguồn ứng dụng, notebook, báo cáo PDF/LaTeX, hình PNG, chỉ số đánh giá và cấu hình model phục vụ việc đọc và tái hiện. Năm notebook trong snapshot có output của lượt thực nghiệm tương ứng. Các file dung lượng lớn được loại bằng `.gitignore`: corpus Parquet và ba danh sách URL (`data_URLs_raw.json`, `data_URLs_removed.json`, `data_URLs_clean.json`), trọng số model, pipeline `.joblib`, cache trong `temp/` và `result/evaluation/article_metadata.parquet`. File `dataset/data_URLs_meta.json` mô tả snapshot dữ liệu. Cấu hình VS Code trên máy và file `app/.env` không được đưa vào Git; nội dung repository thực tế phụ thuộc các file đã commit/push.

Sau khi clone, chạy notebook dataset để tạo corpus, rồi chạy notebook PhoBERT để tạo trọng số, cache và kết quả. Muốn phục vụ đúng checkpoint của thực nghiệm trong báo cáo, đặt gói model tương ứng vào `model/` và đối chiếu SHA-256 trong `app/artifacts/active.json`. Chỉ có cấu hình và tokenizer trong repository chưa đủ để chạy dịch vụ suy luận. Khi huấn luyện một checkpoint khác, sử dụng metadata và registry do notebook xuất cho checkpoint đó.

`artifact_manifest.json` mô tả bộ artifact đầy đủ của thực nghiệm, gồm cả các file dung lượng lớn được giữ ngoài Git. Đối chiếu manifest cần có bộ file đầy đủ; bản clone riêng không chứa toàn bộ artifact trong manifest.

## Thực thi notebook

Mở thư mục gốc trong VS Code hoặc Jupyter và chọn kernel Python 3.14. Kernel dataset cần Internet; kernel PhoBERT cần PyTorch CUDA và GPU NVIDIA. Các cell đầu kiểm tra môi trường và thư viện trước các bước tốn thời gian.

1. **Dataset_VietNamNet.ipynb — Run All**: thu thập URL, tải nội dung cùng bằng chứng chuyên mục, chuẩn hóa nhãn, đối chiếu và xuất dataset sạch.
2. **PhoBERT_Classification.ipynb — Run All**: đọc corpus, tiền xử lý, khử trùng, chia tập, tokenize, fine-tuning, fit calibration trên validation, đánh giá test và xuất artifact.
3. **LSTM_Classification.ipynb**, **SVM_Classification.ipynb**, **Logistic_Regression.ipynb**: mỗi notebook có thể Run All độc lập sau notebook dataset, không cần chạy notebook huấn luyện nào trước nó. LSTM dùng GPU local; SVM và Logistic dùng CPU.

Môi trường thực nghiệm: Python 3.14.7; PyTorch 2.11.0+cu128; CUDA 12.8; Transformers 5.17.0; Accelerate 1.15.0; NumPy 2.5.2; pandas 3.0.6; scikit-learn 1.9.1; PyArrow 26.0.0; PyVi 0.1.1. `requirements-notebook.txt` khai báo thư viện. Với máy khác, chọn bản PyTorch CUDA phù hợp theo [hướng dẫn chính thức](https://pytorch.org/get-started/locally/) rồi cài các dependency còn lại. Chạy trên máy hiện tại không yêu cầu tạo môi trường ảo trong project.

Các notebook xác định đường dẫn từ thư mục dự án. Đầu vào training chỉ đọc; các đầu ra nằm trong `temp/`, `model/`, `result/`, assets báo cáo và metadata ứng dụng. Tiến độ là văn bản với thanh `#`, cập nhật tại chỗ trên một dòng cho mỗi tiến trình. Nội dung HTML thay đổi theo thời điểm; một lần crawl khác không cam kết tạo đúng số bài của snapshot này.

### Ba chế độ PhoBERT

| `TRAINING_MODE` | Trọng số khởi đầu | Huấn luyện | Lịch sử |
|---|---|---|---|
| `"fresh"` — mặc định | Backbone PhoBERT | Tối đa 7 epoch | Lịch sử của lượt hiện tại |
| `"continue"` | Gói fine-tune tương thích trong `model/` | Thêm tối đa 7 epoch | Nối epoch và bước với offset |
| `"evaluate"` | Gói fine-tune tương thích | Không train | Đánh giá và xuất kết quả |

`continue` kiểm tra dữ liệu, tiền xử lý, mapping và fingerprint cả ba tập. Optimizer, scheduler và bộ đếm early stopping được khởi tạo lại; đây không phải resume một bước đang chạy dở. Nếu validation không cải thiện, model đầu vào được giữ. Hai lượt đủ bảy epoch có thể tạo đường học 14 epoch; early stopping có thể làm số điểm ít hơn. Cache hợp lệ được tái sử dụng ngay cả khi chọn `fresh`.

Muốn thu thập hoàn toàn từ đầu, dừng kernel và chủ động dọn đầu ra dataset trước Run All. Muốn train từ backbone trên corpus đang có, chọn `fresh`; không cần xóa cache hoặc dataset. Đọc output và `result/evaluation/` đủ để xem thực nghiệm mà không train lại.

## Thực nghiệm LSTM, SVM và Logistic Regression

Ba notebook thực hiện cùng quy tắc chuẩn hóa/khử trùng corpus và chia phân tầng 70/15/15, seed 42. Khử trùng diễn ra trước khi loại stopword hay tạo biểu diễn riêng. Từng notebook tự tạo corpus/split, không đọc split của notebook khác. Thứ tự Parquet, thứ tự giữ bài trùng và hai lần `train_test_split` được cố định; `split_manifest.json` xuất indices và hash danh sách URL theo thứ tự cho cả ba tập. Với cùng snapshot dataset và phiên bản tiền xử lý, các mô hình được đánh giá trên cùng các bài viết.

| Notebook | Biểu diễn và cấu hình tìm kiếm | Thiết bị | Chế độ |
|---|---|---|---|
| LSTM | Vocabulary train ≤50.000 từ, embedding học từ đầu 200 chiều, BiLSTM hidden 128 mỗi hướng, dropout 0,35; thử Head–Tail 256/512 từ và trọng số lớp lũy thừa 0,5/1 | GPU NVIDIA local | `fresh`, `continue`, `evaluate` |
| SVM | TF-IDF unigram hoặc unigram+bigram, ≤150.000 đặc trưng, min_df=3, C=0,5/1,5/3, trọng số đều/balanced; LinearSVC L2 squared hinge, dual auto | CPU, giới hạn 4 thread | `fresh`, `evaluate` |
| Logistic | TF-IDF cùng không gian tìm kiếm; multinomial Logistic Regression L2, solver L-BFGS | CPU, giới hạn 4 thread | `fresh`, `evaluate` |

SVM và Logistic có bản stopword riêng tại `resources/svm/vietnamese-stopwords.txt` và `resources/logistic/vietnamese-stopwords.txt`, tăng trọng số tiêu đề ba lần sau khi chốt corpus. LSTM giữ stopword, học vocabulary chỉ từ train, dùng PAD/UNK và packed sequence để loại padding khỏi phép tính recurrent. Độ dài LSTM tính bằng từ đã phân đoạn; độ dài PhoBERT tính bằng subword, không coi hai giới hạn token là cùng một đơn vị.

LSTM dùng AdamW lr=0,001, weight decay=0,0001, batch train 64/eval 128, clip norm=1, tối đa 10 epoch/cấu hình. Early stopping có patience=2, min delta=0,001 theo Macro-F1 validation; scheduler giảm learning rate một nửa sau một epoch không cải thiện đủ ngưỡng, trước lượt huấn luyện tiếp theo. Checkpoint được giữ khi điểm validation tốt hơn, kể cả cải thiện chưa vượt min delta của patience. `continue` nối lịch sử epoch, tạo optimizer/scheduler mới và giữ checkpoint đầu vào nếu các epoch bổ sung không cải thiện. SVM/Logistic tối ưu đến hội tụ; nếu đạt giới hạn vòng lặp sẽ thử một giới hạn lớn hơn. Cấu hình chưa hội tụ không được chọn.

Chỉ validation chọn cấu hình: Macro-F1 trước, Weighted-F1/Accuracy để phá hòa, rồi giữ cấu hình gặp trước. Mỗi model chỉ xuất test của cấu hình được chọn; không refit bằng train+validation, không chọn số tốt nhất từ nhiều lượt test. Temperature dương fit NLL validation trước đánh giá test; không có hệ số quyết định riêng từng lớp. Temperature giữ nguyên argmax nên Accuracy/F1 trước và sau giống nhau. Xác suất SVM là softmax của margin đã hiệu chỉnh, không phải xác suất gốc của LinearSVC.

| Loại đầu ra | Vị trí của mỗi model (`lstm`, `svm`, `logistic`) |
|---|---|
| Gói model | `model/<model>/`: checkpoint hoặc pipeline, config, nhãn, calibration, manifest |
| Kết quả cho bảng so sánh | `result/<model>/evaluation/comparison_summary.json` |
| Dữ liệu đối chiếu | Split/corpus fingerprints, URL hash, predictions Parquet, confusion matrix JSON, CSV chỉ số từng lớp |
| Tìm tham số | `search_results.json`: cấu hình, Macro-F1 validation, thời gian fit; không có điểm test từng ứng viên |
| Hình báo cáo | `result/<model>/figures/`: từng PNG riêng 300 dpi, ma trận, P/R/F1, calibration, PR 20 lớp, ROC, phân bố và kiểm tra split |
| Chẩn đoán riêng | LSTM: đường học, OOV, token retention, trọng số lớp; SVM/Logistic: hệ số từ vựng riêng từng lớp |

`comparison_summary.json` phân biệt thời gian tìm tham số, thời gian fit cấu hình được chọn và thời gian inference. Inference SVM/Logistic gồm tạo TF-IDF và tính score; LSTM gồm DataLoader/chuyển GPU/forward. Các thời gian này không gồm phân đoạn và cần nêu phạm vi khi so sánh. Khoảng tin cậy bootstrap iid 95% trên test đo bất định với model cố định, không thay thế nhiều seed. Khi so sánh PhoBERT, phân biệt F1 softmax chuẩn với F1 sau hệ số lớp; không dùng cải thiện của calibration xác suất làm bằng chứng cải thiện phân loại.

Model mới không tự được đăng ký vào ứng dụng: app phục vụ gói PhoBERT trong `model/`. Báo cáo và số liệu PhoBERT giữ hợp đồng đó; kết quả đầy đủ của từng baseline được notebook tương ứng sinh khi Run All.

Tài liệu thuật toán: [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html), [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html).

## Dataset và taxonomy

Thu thập tối đa 500 trang cho mỗi chuyên mục; số trang khác số URL hoặc số bài. Tải và kiểm tra bài dùng 32 worker, thử tối đa ba lượt. Một lần đọc HTML cung cấp tiêu đề, nội dung và bằng chứng chuyên mục. Nhãn cuối dựa trên chuyên mục của chính bài; mục con được quy về class cha đã cấu hình.

**Sắc màu Việt Nam thuộc Dân tộc – Tôn giáo.** Công nghiệp hỗ trợ sử dụng URL `hop-tac/cong-nghiep-ho-tro` nhưng slug/class file là `cong-nghiep-ho-tro`. Bài thuộc class khác trong taxonomy được chuyển cả URL và nội dung. Bài xác định ngoài taxonomy hoặc không đạt sau ba lượt thử bị loại khỏi corpus và có lý do trong audit. Chỉ xuất Parquet khi URL, nhãn, bằng chứng và số lượng đối chiếu thành công.

| Giai đoạn | Số lượng |
|---|---:|
| Lượt URL theo nguồn | 205.463 |
| URL đầu vào duy nhất | 204.280 |
| URL giữ chuyên mục | 199.463 |
| URL được chuyển chuyên mục | 874 |
| Ngoài taxonomy | 3.866 |
| Hết ba lượt tải/kiểm tra | 76 |
| Request canonical trùng được gộp | 1 |
| Bài sạch trong Parquet | 200.337 |
| Dòng thuộc nhóm trùng nhưng xung đột nhãn bị loại | 8 |
| Dòng trùng cùng nhãn dư được gộp | 364 |
| Corpus dùng chia tập | 199.965 |

Đối soát: **204.280 = 200.337 + 3.866 + 76 + 1**. Một bài có thể xuất hiện ở nhiều nguồn. Ma trận nguồn–đích có 201.444 quan hệ và 1.981 quan hệ ngoài đường chéo; đây không phải số bài duy nhất được chuyển class (874).

### File dữ liệu

| File trong `dataset/` | Vai trò |
|---|---|
| `data_URLs_raw.json` | URL đầu vào theo chuyên mục nguồn |
| `data_URLs_removed.json` | URL bị loại/gộp, lý do và bằng chứng |
| `data_URLs_meta.json` | Cấu hình, hash, số lượng, thống kê và audit |
| `data_URLs_clean.json` | URL canonical theo nhãn đã xác minh |
| `<slug>.parquet` | URL, title, content, nhãn và bằng chứng chuyên mục |

Trước chuẩn hóa dùng số lượng URL gốc; sau chuẩn hóa dùng URL sạch và Parquet. Checkpoint thu thập được dọn sau đối chiếu hoàn tất. JSON URL gốc/bị loại là hồ sơ của pipeline, không phải đầu vào training. Dataset sạch không đồng nghĩa nhãn đã được con người kiểm toán ngữ nghĩa trên mọi bài.

### Nhãn và phép chia

ID model theo thứ tự trong `model/label_config.json`. Trong bảng, dấu chấm phân cách hàng nghìn. Val và test giữ phân bố tự nhiên; không oversample.

| ID | Nhãn | Đường dẫn nguồn | Parquet | Sau khử trùng | Train | Val | Test |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | Bạn đọc | `ban-doc` | 12.310 | 12.119 | 8.483 | 1.818 | 1.818 |
| 1 | Bất động sản | `bat-dong-san` | 12.311 | 12.309 | 8.616 | 1.847 | 1.846 |
| 2 | Chính trị | `chinh-tri` | 12.243 | 12.242 | 8.569 | 1.837 | 1.836 |
| 3 | Công nghiệp hỗ trợ | `hop-tac/cong-nghiep-ho-tro` | 1.297 | 1.292 | 905 | 193 | 194 |
| 4 | Công nghệ | `cong-nghe` | 11.062 | 11.061 | 7.743 | 1.659 | 1.659 |
| 5 | Du lịch | `du-lich` | 10.980 | 10.979 | 7.685 | 1.647 | 1.647 |
| 6 | Dân tộc - Tôn giáo | `dan-toc-ton-giao` | 3.700 | 3.700 | 2.590 | 555 | 555 |
| 7 | Dân tộc thiểu số và miền núi | `dan-toc-thieu-so-va-mien-nui` | 1.088 | 1.088 | 762 | 163 | 163 |
| 8 | Giáo dục | `giao-duc` | 12.261 | 12.259 | 8.581 | 1.839 | 1.839 |
| 9 | Kinh doanh | `kinh-doanh` | 12.649 | 12.643 | 8.850 | 1.896 | 1.897 |
| 10 | Nông thôn mới | `nong-thon-moi` | 969 | 969 | 678 | 145 | 146 |
| 11 | Nội dung chuyên đề | `noi-dung-chuyen-de` | 12.569 | 12.568 | 8.798 | 1.885 | 1.885 |
| 12 | Sức khỏe | `suc-khoe` | 12.385 | 12.385 | 8.670 | 1.858 | 1.857 |
| 13 | Thế giới | `the-gioi` | 11.889 | 11.889 | 8.322 | 1.783 | 1.784 |
| 14 | Thể thao | `the-thao` | 12.401 | 12.267 | 8.587 | 1.840 | 1.840 |
| 15 | Thời sự | `thoi-su` | 12.420 | 12.420 | 8.694 | 1.863 | 1.863 |
| 16 | Tuần Việt Nam | `tuan-viet-nam` | 11.280 | 11.276 | 7.893 | 1.692 | 1.691 |
| 17 | Văn hóa - Giải trí | `van-hoa-giai-tri` | 12.318 | 12.317 | 8.622 | 1.848 | 1.847 |
| 18 | Ô tô - Xe máy | `oto-xe-may` | 12.300 | 12.299 | 8.609 | 1.845 | 1.845 |
| 19 | Đời sống | `doi-song` | 11.905 | 11.883 | 8.318 | 1.782 | 1.783 |

## Mô hình và huấn luyện

Đầu vào ghép một lần `title + " " + content`, chuyển chữ thường, thay dấu câu/chữ số bằng khoảng trắng, phân đoạn bằng PyVi và chuẩn hóa khoảng trắng; không bỏ stopwords. Tokenizer PhoBERT thực hiện BPE sau phân đoạn từ. PyVi thống nhất giữa training và serving, nhưng khác VnCoreNLP của tiền huấn luyện; chưa có ablation để định lượng khác biệt.

Head–Tail có **256 vị trí**: chuỗi trên 254 BPE token giữ 127 đầu và 127 cuối, thêm `<s>` và `</s>`. Chuỗi ngắn được thêm token đặc biệt và padding. Attention mask loại padding. Khảo sát 1.000 bài seed 42 cho thấy 85,4% dài hơn 256 token tính cả token đặc biệt; Head–Tail bỏ phần giữa, không đọc toàn bài bằng attention.

PhoBERT-base-v2 có 12 encoder layers, hidden size 768, 12 attention heads, intermediate size 3.072 và vocabulary 64.001. Checkpoint có **135.013.652 tham số**, bao gồm head 20 nhãn. Fine-tuning cập nhật toàn bộ backbone và head.

| Tham số | Giá trị |
|---|---|
| Seed / split | 42 / phân tầng 70–15–15 |
| GPU | Một RTX A3000 Laptop 12 GB |
| Batch train / gradient accumulation | 32 / 2 |
| Batch hiệu dụng / batch evaluation | 64 / 64 |
| Epoch / early stopping | Tối đa 7 / patience 2, cải thiện tối thiểu 0,001 |
| Best checkpoint | Macro-F1 validation, đánh giá và lưu mỗi epoch |
| Learning rate / weight decay | 2e-5 / 0,01 |
| Warmup / optimizer | 200 bước / AdamW fused |
| Precision | BF16, TF32 bật; FP16 nếu GPU không hỗ trợ BF16 |
| Loss | Weighted cross-entropy |
| Class weight | `N_train / (20 × n_class_train)` |
| Sampling | Ngẫu nhiên; không boost cố định hoặc oversample |
| Dataloader workers / logging | 0 / 100 bước |
| Gradient checkpointing / compile | Tắt / tắt |

Class weights chỉ tính từ train. Tỷ lệ class lớn/nhỏ khoảng 13,05; trọng số Nông thôn mới khoảng 10,3226, Dân tộc thiểu số và miền núi 9,1847. Trọng số lớn ưu tiên loss nhưng không tạo thêm ví dụ hoặc giải quyết nhãn mơ hồ. Test độc lập không phục vụ chọn checkpoint, fit temperature hay hệ số.

### Artifact đánh giá

`result/evaluation/` lưu audit dataset, split manifest, class balance, training history, logits validation/test, dự đoán Parquet, classification reports, ma trận trước/sau, calibration metrics, routing simulation và ví dụ lỗi. `article_metadata.parquet` truy vết article index sang URL và nội dung. `evaluation_manifest.json` ghi thông tin môi trường và artifact tại thời điểm xuất; SHA notebook ghi trong đó không được hiểu là hash của file đã bổ sung toàn bộ output cuối cùng.

`result/figures/` gồm 69 PNG riêng 300 dpi: bốn hình từ notebook dataset và 65 hình từ notebook PhoBERT. Các nhóm hình gồm phân bố/độ dài/overlap, class weights, đường học, ma trận, chỉ số từng lớp, calibration, coverage, margin, lỗi và 20 Precision–Recall độc lập. Bảng tiền xử lý trong báo cáo được dàn lại bằng LaTeX từ output để văn bản không chồng nhau.

## Hiệu chỉnh và suy luận

Với logits `z`, xác suất là `p = softmax(z / T)`. Nhãn cuối chọn **`argmax(p / tau)`**; `tau` có một hệ số dương cho mỗi class, không phải ngưỡng xác suất xuất nhiều nhãn. Validation fit temperature trong [0,5; 5,0] bằng NLL và fit hệ số trên 200 giá trị [0,3; 3,5], ba lượt tìm kiếm tọa độ để tăng Macro-F1 toàn bộ 20 lớp.

- Top-k xếp theo điểm `p / tau`; thanh hiển thị xác suất `p` của từng nhãn.
- Confidence là xác suất của nhãn cuối; có thể thấp hơn xác suất một nhãn khác.
- Margin là chênh lệch hai điểm cao nhất sau chuẩn hóa điểm quyết định; không tự động là xác suất đã calibration.
- Tie chọn ID nhỏ hơn. Top-k không thay nhãn hoặc margin.

Ngưỡng nghiệp vụ mặc định: confidence ≥ 0,75 vào auto; từ 0,68 đến dưới 0,75 vào review; dưới 0,68 vào escalate. Các ngưỡng này lưu trong PostgreSQL, khác `tau` trong model. Thay routing không sửa trọng số hoặc tự train lại. Model luôn chọn trong 20 lớp; confidence thấp không phải một bộ phát hiện ngoài miền đã kiểm chứng.

## Chạy ứng dụng

Yêu cầu Docker Desktop có Compose. Từ thư mục gốc:

```powershell
docker compose -f app/compose.yaml up --build -d --wait
docker compose -f app/compose.yaml ps
```

UI: [localhost:5173](http://localhost:5173). API/OpenAPI: [localhost:8000/docs](http://localhost:8000/docs). Health: [localhost:8000/api/health](http://localhost:8000/api/health). Model service gRPC cổng 50051; PostgreSQL 5432; Redis 6379.

| Workspace | Email | Mật khẩu minh họa |
|---|---|---|
| Editor | editor@vnn-lab.edu.vn | vnn-password |
| Admin | admin@vnn-lab.edu.vn | vnn-password |
| Data Scientist | scientist@vnn-lab.edu.vn | vnn-password |
| Editor — Education queue | education-editor@vnn-lab.edu.vn | vnn-password |

Sáu dịch vụ gồm UI React/Vite, API FastAPI, model service gRPC, worker Dramatiq, PostgreSQL 16 và Redis 7. Schema ứng dụng là `vnn`. Volume PostgreSQL/Redis giữ dữ liệu khi dừng stack. Container inference chạy CPU float32 với model/tokenizer local; notebook train dùng GPU BF16. Logits sát ranh giới có thể chênh theo precision.

### Chức năng

| Vai trò | Chức năng |
|---|---|
| Editor | Nhập URL hoặc title/content, theo dõi job, đọc bài, xem top-k/confidence/margin, approve/override/escalate và phân trang |
| Admin | Quản lý người dùng, routing thresholds, preview tác động và vận hành |
| Data Scientist | Monitoring, registry/model versions, xem artifact/ma trận và Dataset Lab |

API kiểm tra phiên và vai trò độc lập với UI. Feedback gắn đúng prediction ID; suy luận lại không biến phản hồi cho prediction khác thành nhãn xác minh. Monitoring tính từ prediction có phản hồi, chỉ trên các nhãn quan sát; khác test offline 20 lớp. Drift score là heuristic, không phải kiểm định drift thống kê. Dataset Lab mô tả corpus vận hành và ứng viên relabel; không ghi đè `dataset/` hoặc tự fine-tuning.

Log và dừng dịch vụ:

```powershell
docker compose -f app/compose.yaml logs --tail 100 model-service api-service worker-service
docker compose -f app/compose.yaml down
```

Các biến cổng `UI_PORT`, `API_PORT`, `MODEL_PORT`, `POSTGRES_PORT`, `REDIS_PORT` và `POSTGRES_PASSWORD` được Compose hỗ trợ. Khi đổi cổng UI/API cần build lại frontend để URL và CORS khớp. Cấu hình mặc định phục vụ minh họa localhost; triển khai production cần secret riêng, TLS, giới hạn request và quản trị nội dung.

## Database minh họa và sao lưu

`app/demo/postgres.backup` chứa schema `vnn` với **32 bài từ corpus, 32 dự đoán thật, 10 phản hồi minh họa**, tài khoản, routing và registry model 20 lớp. Phiên đăng nhập và dữ liệu audit events không nằm trong bản sao lưu. Backup không chứa trọng số hoặc corpus huấn luyện. Các bài được chọn để minh họa đủ lớp và trường hợp khó, không phải một test set độc lập.

Trên máy mới với volume PostgreSQL trống, khởi động riêng PostgreSQL/Redis rồi nạp snapshot trước khi API tạo schema:

```powershell
docker compose -f app/compose.yaml up -d --wait postgres redis
docker compose -f app/compose.yaml cp app/demo/postgres.backup postgres:/tmp/vnn-demo.backup
docker compose -f app/compose.yaml exec -T postgres pg_restore -U vnn -d vnn_ml --no-owner --no-privileges --exit-on-error /tmp/vnn-demo.backup
docker compose -f app/compose.yaml up --build -d --wait
docker compose -f app/compose.yaml exec -T postgres rm /tmp/vnn-demo.backup
```

Lệnh restore trên dành cho database chưa có schema `vnn`; nếu schema đã chứa dữ liệu, PostgreSQL sẽ báo đối tượng tồn tại. Dùng một database minh họa riêng khi cần nạp snapshot; không chạy restore thay thế lên dữ liệu vận hành. Các volume không nằm trong folder mã nguồn, nên sao chép folder không tự sao chép trạng thái PostgreSQL/Redis.

Xuất snapshot ứng dụng:

```powershell
docker compose -f app/compose.yaml exec -T postgres pg_dump -U vnn -d vnn_ml -n vnn -Fc --exclude-table-data=vnn.auth_sessions --exclude-table-data=vnn.audit_events -f /tmp/vnn-export.backup
docker compose -f app/compose.yaml cp postgres:/tmp/vnn-export.backup app/demo/postgres.backup
docker compose -f app/compose.yaml exec -T postgres rm /tmp/vnn-export.backup
```

## Gói model và registry

Gói `model/` gồm `model.safetensors`, `config.json`, `label_config.json`, `thresholds.json`, `tokenizer_config.json`, `added_tokens.json`, `vocab.txt`, `bpe.codes`, `training_args.bin`, `train_history.pkl`. `app/artifacts/active.json` sử dụng đường dẫn tương đối `model` và run ID `phobert_headtail_seed42`.

SHA-256 trọng số:

```text
607f3240935fa1a977d37555f2dc980f68645dcad1da0578ff3b036aa060deb4
```

Startup kiểm tra file bắt buộc, mapping ID/label, preprocessing, Head–Tail, temperature/hệ số dương và hash model. Registry của gói root lấy `metrics.json` cùng `confusion_matrix_after.json` từ `result/evaluation/`; ma trận 20×20 có tổng 29.995. Calibration hash được dùng để nhận diện thay đổi hệ số cùng trọng số.

Upload package chấp nhận ZIP hoặc nhóm file, gồm trọng số, metadata và tokenizer; thêm `metrics.json` và `confusion_matrix_after.json` để hiển thị kết quả đánh giá tương ứng. Package được kiểm tra trước đăng ký. Kích hoạt cập nhật active manifest nguyên tử, xác nhận model service đúng phiên bản và phục hồi manifest nếu kích hoạt thất bại. Sau khi notebook tạo model, khởi động/restart model, API và worker để các tiến trình nạp package trên đĩa:

```powershell
docker compose -f app/compose.yaml restart model-service api-service worker-service
docker compose -f app/compose.yaml ps
```

Bindings Protobuf đi kèm mã ứng dụng. Nếu phát triển schema, cài `requirements-codegen.txt` rồi chạy `bash app/model-service/scripts/generate_proto.sh`. Không cần tạo bindings khi chạy app thông thường.

## Báo cáo

[CO5087_Report.pdf](report/CO5087_Report.pdf) gồm bài toán, tổng quan nghiên cứu, corpus, biểu diễn/kiến trúc bốn model, huấn luyện, calibration, phân tích sâu PhoBERT, so sánh SVM/LR/BiLSTM/PhoBERT, ứng dụng và kiểm chứng. Chương so sánh dùng cùng test, tách argmax khỏi hệ số lớp, phân tích F1 từng chuyên mục, lỗi chung, bootstrap ghép cặp và phạm vi số đo tài nguyên. Các chỉ số nghiên cứu trước có vai trò riêng, không thay thế kết quả của project.

Nguồn vào là `report/source_latex/main.tex`; nội dung tại `chapters/`, bảng tại `tables/`, ảnh tại `assets/`, tham khảo tại `references.bib`. Sơ đồ TikZ được định nghĩa trong `architectures.tex`, chương tổng quan và phần kiến trúc BiLSTM. Hình notebook là PNG độc lập; các hình đối chứng được sao chép nguyên byte vào `assets/baselines/`. Ảnh ứng dụng là screenshot từ hệ thống phục vụ checkpoint PhoBERT. Số liệu phân biệt train/validation/test, corpus sạch và sau khử trùng, offline và monitoring minh họa.

Biên dịch từ `report/source_latex/`, cần XeLaTeX, BibTeX, Times New Roman và Cambria Math:

```powershell
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
Copy-Item main.pdf ../CO5087_Report.pdf
```

Khổ A4, font, cỡ chữ, lề, heading, caption và macro hình/bảng được quy định trong `preamble.tex`. Khi tạo một thực nghiệm khác, bảng và nội dung diễn giải cần đối chiếu với `result/evaluation/` trước khi biên dịch; thay hình đơn thuần không tự thay số liệu trong văn bản.

## Kiểm chứng

Năm notebook có output đầy đủ và không có cell error: 15 cell mã dữ liệu, 72 cell mã PhoBERT, 31 cell mã cho mỗi notebook SVM/LR/BiLSTM. Artifact đối soát 20 class, 200.337 Parquet sạch, 199.965 mẫu chia tập và 29.995 mẫu test. Ba đối chứng có cùng mảng split với PhoBERT; URL, nhãn và thứ tự test khớp. Chỉ số, ma trận và NLL được tái tính từ score/dự đoán đã xuất, không train lại để tạo số cho báo cáo. Chín phép kiểm tra giao nhau exact theo URL/raw text/clean text trên ba cặp tập đều bằng không.

46 kiểm tra ứng dụng/hợp đồng model đạt, gồm:

- SHA-256, taxonomy, metadata và toàn bộ 29.995 quyết định calibration khớp công thức app.
- Head–Tail ở 0, 1, 253, 254, 255, 256, 1.000 token; đối chiếu preprocessing thực tế và số tham số checkpoint.
- 32 dự đoán CPU thực tế khớp nhãn offline của các bài minh họa.
- Health, đăng nhập ba vai trò, sai mật khẩu, thiếu phiên, phân quyền và logout hủy phiên.
- Đầu vào rỗng/chỉ ký hiệu, nhãn ngoài taxonomy, ngưỡng đảo thứ tự, article không tồn tại và temperature không hợp lệ.
- Worker import, API các trang, recompute monitoring và registry model.
- Khôi phục backup trên database riêng: đúng 32 bài, 32 dự đoán, 10 phản hồi và không có phiên đăng nhập.

Frontend build TypeScript/Vite thành công; sáu dịch vụ Compose hoạt động; ba workspace được kiểm tra qua trình duyệt. Phụ lục báo cáo liệt kê từng kiểm tra đã thực thi. Kiểm chứng áp dụng cho snapshot/môi trường nêu trên; dữ liệu web, phần cứng hoặc dependency khác cần kiểm tra tương ứng.

## Xử lý sự cố

| Hiện tượng | Kiểm tra |
|---|---|
| Kernel thiếu import | Interpreter đúng Python và dependency notebook; không dùng kernel khác với môi trường đã cài |
| `torch.cuda.is_available()` là False | PyTorch CUDA, driver NVIDIA, kernel đang chọn; app Docker CPU không yêu cầu CUDA |
| Model thiếu/không tương thích | File trong `model/`, label config và active manifest; không trộn tokenizer/thresholds của package khác |
| App không thấy package vừa xuất | Restart model/API/worker, xem health và registry SHA-256 |
| `continue` từ chối fingerprint | Dùng đúng dataset/split/cache; chọn `fresh` nếu muốn huấn luyện corpus khác |
| Crawl URL lỗi | Internet, response HTML/breadcrumb; ba lượt thất bại được audit và loại khỏi training |
| Frontend không gọi API | Cổng, URL build frontend, CORS và tình trạng API |
| Compose chậm startup | `ps`, log model/health; nạp checkpoint CPU cần thời gian |
| Restore báo schema tồn tại | Chọn database trống hoặc database minh họa riêng; tránh ghi đè vận hành |
| Monitoring khác Macro-F1 report | Monitoring từ feedback vận hành; report từ test độc lập, khác mẫu/nhãn quan sát |
| Notebook chạy lâu | Số corpus, tokenize, 7 epoch và phần cứng; cache giảm tiền xử lý, không thay số bước fine-tuning |

## Nguồn dữ liệu và giấy phép

Nội dung bài báo và quyền tác giả thuộc [VietNamNet](https://vietnamnet.vn/). Corpus lưu URL và chuyên mục nguồn để truy vết; nội dung không được nhận là do tác giả dự án sáng tác. Không có tuyên bố giấy phép mở chung cho corpus.

Backbone do VinAI cung cấp: [PhoBERT](https://github.com/VinAIResearch/PhoBERT), [model card PhoBERT-base-v2](https://huggingface.co/vinai/phobert-base-v2). Model card công bố AGPL-3.0; sử dụng/phân phối trọng số fine-tune cần tuân theo giấy phép upstream. Phân đoạn từ dùng [PyVi](https://github.com/trungtv/pyvi) 0.1.1. PyTorch, Transformers, thư viện Python, Docker images và dependency frontend có giấy phép riêng trong bản phân phối; `package-lock.json` ghi dependency UI.

Bibliography báo cáo dẫn các nghiên cứu phân loại tin tức tiếng Việt, khảo sát phương pháp, Transformer, BERT, RoBERTa, PhoBERT, AdamW, temperature scaling và công cụ phân đoạn. Bảng tổng quan ghi rõ corpus, phương pháp, metric và giới hạn từng nguồn; chỉ số bên ngoài không được dùng để xếp hạng mô hình trên corpus này. Sơ đồ ba hướng biểu diễn trong chương tổng quan là khung lựa chọn phương pháp, còn các sơ đồ kỹ thuật mô tả pipeline PhoBERT và ứng dụng đang phục vụ.
