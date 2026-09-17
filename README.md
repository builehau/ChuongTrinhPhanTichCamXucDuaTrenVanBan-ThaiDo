# Phân tích cảm xúc / thái độ từ văn bản tiếng Việt

Dự án bài tập lớn môn Trí tuệ nhân tạo, phân loại văn bản thành ba lớp:

- `tich_cuc`: khen ngợi, hài lòng, ủng hộ.
- `trung_lap`: thông tin hoặc mô tả không thể hiện rõ thái độ.
- `tieu_cuc`: phàn nàn, thất vọng, phản đối.

Mô hình kết hợp **TF-IDF từ/cụm từ**, **TF-IDF ký tự**, một nhóm tín hiệu từ điển cảm xúc nhỏ và **Logistic Regression**. Dự án không cần GPU, huấn luyện nhanh, có CLI, giao diện Streamlit, xử lý CSV hàng loạt, Accuracy, Macro F1, 5-fold cross-validation và ma trận nhầm lẫn.

## 1. Cấu trúc dự án

```text
sentiment_analysis_vi/
├── app.py                         # Giao diện web Streamlit
├── predict.py                     # Dự đoán bằng dòng lệnh
├── train.py                       # Huấn luyện và đánh giá
├── data/
│   ├── create_dataset.py          # Tạo lại dữ liệu minh họa
│   └── sentiment_vi.csv           # Dữ liệu 3 lớp cân bằng
├── src/
│   ├── sentiment_model.py         # Tiền xử lý, pipeline, dự đoán
│   └── training.py                # Đọc dữ liệu, đánh giá, lưu mô hình
├── tests/test_sentiment.py        # Kiểm thử tự động
├── models/sentiment_model.joblib  # Mô hình sau khi huấn luyện
├── results/metrics.json           # Chỉ số đánh giá
├── results/confusion_matrix.png   # Ma trận nhầm lẫn
└── docs/BAO_CAO_MAU.md            # Khung báo cáo và thuyết trình
```

## 2. Cài đặt trên macOS / Linux

Mở Terminal tại thư mục dự án và chạy:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Trên Windows:

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Khuyến nghị Python 3.11–3.12. Mô hình đi kèm được tạo bằng scikit-learn 1.8.0; tệp `requirements.txt` đã cố định phiên bản này để tránh lỗi không tương thích.

## 3. Chạy chương trình

Mô hình đã được huấn luyện sẵn. Sau khi cài thư viện, mở giao diện bằng:

```bash
python -m streamlit run app.py
```

Trong tab **Phân tích tệp CSV**, có thể dùng ngay `data/demo_input.csv` để trình diễn.

Nếu muốn huấn luyện lại:

```bash
python train.py
```

Dự đoán nhanh bằng Terminal:

```bash
python predict.py --text "Giao hàng nhanh, sản phẩm rất đẹp"
python predict.py --text "Cuộc họp diễn ra lúc 9 giờ sáng" --json
```

Chạy kiểm thử:

```bash
python -m unittest discover -s tests -v
```

Tạo lại dữ liệu mẫu rồi huấn luyện lại:

```bash
python data/create_dataset.py
python train.py
```

## 4. Định dạng dữ liệu riêng

Tạo CSV UTF-8 có hai cột:

```csv
text,label
"Sản phẩm rất tốt",tich_cuc
"Đơn hàng được tạo lúc 9 giờ",trung_lap
"Ứng dụng liên tục bị lỗi",tieu_cuc
```

Sau đó chạy:

```bash
python train.py --data duong_dan/du_lieu.csv
```

Mỗi lớp cần tối thiểu 10 mẫu. Nên có ít nhất vài trăm câu thật cho mỗi lớp và giữ tỉ lệ các lớp tương đối cân bằng.

## 5. Quy trình AI

1. Chuẩn hóa Unicode, chữ thường, khoảng trắng, URL, email, số và emoji.
2. Biểu diễn văn bản bằng TF-IDF word n-gram `(1, 2)`, character n-gram `(3, 5)` và bốn đặc trưng từ điển cảm xúc.
3. Chia dữ liệu theo tỉ lệ 80% huấn luyện, 20% kiểm tra có phân tầng.
4. Huấn luyện Logistic Regression có cân bằng trọng số lớp.
5. Đánh giá bằng Accuracy, Macro F1, classification report, confusion matrix và 5-fold CV.
6. Lưu toàn bộ pipeline để dữ liệu mới luôn được xử lý giống dữ liệu huấn luyện.

## 6. Giới hạn cần nêu trong báo cáo

- Dữ liệu đi kèm được xây dựng để minh họa, chưa đại diện cho toàn bộ tiếng Việt.
- Mô hình có thể nhầm câu mỉa mai, câu nhiều ý trái chiều hoặc tiếng lóng chưa gặp.
- Độ tin cậy là xác suất của mô hình, không phải bảo đảm tuyệt đối.
- Khi ứng dụng cho nhà hàng, thương mại điện tử hoặc giáo dục, cần thu thập và gán nhãn dữ liệu đúng lĩnh vực.

## 7. Gợi ý nâng cấp để lấy điểm cộng

- So sánh Logistic Regression với Naive Bayes và SVM trên cùng tập dữ liệu.
- Thu thập thêm đánh giá thật, ẩn dữ liệu cá nhân và gán nhãn thủ công.
- Thêm phát hiện cảm xúc theo khía cạnh như `đồ ăn`, `giao hàng`, `nhân viên`.
- Thử mô hình học sâu PhoBERT và so sánh thời gian/độ chính xác.
