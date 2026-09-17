# Khung báo cáo bài tập lớn

## Tên đề tài

**Xây dựng chương trình phân tích cảm xúc/thái độ từ dữ liệu văn bản tiếng Việt**

## 1. Lý do chọn đề tài

Đánh giá của người dùng trên mạng xã hội, website bán hàng và biểu mẫu góp ý chứa nhiều thông tin về mức độ hài lòng. Việc đọc thủ công tốn thời gian và khó mở rộng. Đề tài xây dựng một hệ thống tự động phân loại nội dung thành tích cực, trung lập hoặc tiêu cực để hỗ trợ tổng hợp ý kiến.

## 2. Mục tiêu

- Nhận một câu hoặc nhiều dòng văn bản tiếng Việt.
- Tiền xử lý và biểu diễn văn bản thành vector số.
- Huấn luyện mô hình học máy có giám sát.
- Trả về nhãn, xác suất từng lớp và độ tin cậy.
- Đánh giá bằng Accuracy, Precision, Recall, F1-score và confusion matrix.
- Cung cấp giao diện web để trình diễn.

## 3. Phạm vi

Hệ thống xử lý ba lớp cảm xúc tổng quát. Phiên bản minh họa chưa xử lý đầy đủ mỉa mai, cảm xúc theo khía cạnh, văn bản quá dài hoặc ngôn ngữ ngoài tiếng Việt.

## 4. Dữ liệu

Bộ dữ liệu đi kèm gồm các câu tiếng Việt được gán một trong ba nhãn `tich_cuc`, `trung_lap`, `tieu_cuc`. Dữ liệu được cân bằng để tránh mô hình thiên lệch sang một lớp. Khi làm báo cáo chính thức, ghi đúng nguồn và quy trình gán nhãn của dữ liệu nhóm tự bổ sung; không mô tả dữ liệu minh họa là dữ liệu thu thập thực tế.

Quy tắc gán nhãn:

| Nhãn | Ý nghĩa | Ví dụ |
| --- | --- | --- |
| Tích cực | Khen, hài lòng, đồng ý | “Dịch vụ rất nhanh và chu đáo.” |
| Trung lập | Thông tin, mô tả không đánh giá | “Cửa hàng mở cửa lúc 8 giờ.” |
| Tiêu cực | Chê, thất vọng, phản đối | “Ứng dụng chậm và thường bị lỗi.” |

## 5. Tiền xử lý

1. Chuẩn hóa Unicode NFC và chuyển về chữ thường.
2. Chuẩn hóa URL, email và chữ số thành token chung.
3. Chuyển một số emoji phổ biến thành từ thể hiện cảm xúc.
4. Rút gọn khoảng trắng, dấu câu lặp.
5. Không loại từ phủ định như “không”, “chưa”, “chẳng” vì chúng làm đảo chiều cảm xúc.

## 6. Thuật toán

### 6.1. TF-IDF

TF-IDF đo mức quan trọng của từ/cụm từ trong một văn bản so với toàn bộ tập dữ liệu:

```text
TF-IDF(t, d) = TF(t, d) × log(N / DF(t))
```

Hệ thống ghép hai nhóm đặc trưng:

- Word n-gram từ 1 đến 2 từ để học các cụm như “rất tốt”, “không hài lòng”.
- Character n-gram từ 3 đến 5 ký tự để tăng khả năng chịu lỗi chính tả và biến thể từ.
- Bốn tín hiệu bổ sung từ một từ điển nhỏ: số từ/cụm tích cực, số từ/cụm tiêu cực, độ chênh lệch và từ nối tương phản. Đây là đặc trưng hỗ trợ; mô hình vẫn học trọng số từ dữ liệu.

### 6.2. Logistic Regression

Logistic Regression tính xác suất của mỗi lớp từ vector đặc trưng. Lớp có xác suất lớn nhất là kết quả dự đoán. Thuật toán phù hợp vì chạy nhanh trên dữ liệu thưa TF-IDF, dễ tái lập và có thể giải thích qua trọng số đặc trưng.

### 6.3. Giả mã

```text
Đọc và kiểm tra dữ liệu
Chia tập train/test theo tỉ lệ 80/20, giữ tỉ lệ nhãn
Học bộ biến đổi TF-IDF trên tập train
Huấn luyện Logistic Regression
Dự đoán tập test
Tính Accuracy, Macro F1 và ma trận nhầm lẫn
Lưu pipeline

Khi người dùng nhập câu mới:
    Chuẩn hóa câu
    Biến đổi bằng TF-IDF đã học
    Tính xác suất ba lớp
    Hiển thị lớp có xác suất cao nhất
```

## 7. Thực nghiệm

Kết quả dưới đây được sinh từ lần chạy thực tế với `random_state=42`, bộ dữ liệu 216 mẫu cân bằng và cách chia 80/20. Nếu thay đổi dữ liệu hoặc cấu hình, chạy lại `python train.py` và cập nhật bảng từ `results/metrics.json`.

| Chỉ số | Kết quả |
| --- | ---: |
| Accuracy trên tập test | 79,55% |
| Macro F1 trên tập test | 79,13% |
| 5-fold CV Macro F1 trung bình | 84,79% |
| Độ lệch chuẩn CV | 4,47% |

Ma trận nhầm lẫn nằm tại `results/confusion_matrix.png`. Trên 44 câu kiểm tra, mô hình đoán đúng toàn bộ 15 câu trung lập, đúng 12/15 câu tiêu cực và 8/14 câu tích cực. Năm câu tích cực bị nhầm thành trung lập, cho thấy mô hình còn thận trọng khi tín hiệu khen ngợi yếu. Với dữ liệu nhỏ, kết quả cao có thể do câu mẫu rõ ràng; không nên kết luận mô hình đã tổng quát tốt cho dữ liệu ngoài thực tế.

## 8. Kiểm thử đề xuất

| Mã | Đầu vào | Kỳ vọng |
| --- | --- | --- |
| TC01 | “Sản phẩm đẹp và giao hàng nhanh” | Tích cực |
| TC02 | “Buổi học bắt đầu lúc 8 giờ” | Trung lập |
| TC03 | “Phần mềm chậm và liên tục bị lỗi” | Tiêu cực |
| TC04 | Chuỗi rỗng | Hiển thị cảnh báo |
| TC05 | CSV có dòng rỗng | Không làm ứng dụng dừng |
| TC06 | “Giao hơi chậm nhưng hàng rất tốt” | Theo ý chính đã học; kiểm tra độ tin cậy |

## 9. Kết luận và hướng phát triển

Hệ thống chứng minh quy trình hoàn chỉnh của một bài toán học máy có giám sát: chuẩn bị dữ liệu, trích xuất đặc trưng, huấn luyện, đánh giá, lưu mô hình và triển khai giao diện. Hướng phát triển gồm mở rộng dữ liệu thật, phát hiện mỉa mai, phân tích theo khía cạnh và so sánh với PhoBERT.

## 10. Kịch bản thuyết trình 7 phút

1. **0:00–0:45:** Vấn đề và mục tiêu.
2. **0:45–1:30:** Dữ liệu và ba nhãn.
3. **1:30–2:30:** Tiền xử lý và TF-IDF.
4. **2:30–3:15:** Logistic Regression.
5. **3:15–4:15:** Chỉ số đánh giá và confusion matrix.
6. **4:15–6:15:** Demo ba câu và một tệp CSV.
7. **6:15–7:00:** Hạn chế, hướng phát triển, kết luận.

## 11. Câu hỏi phản biện thường gặp

- **Tại sao không bỏ từ “không”?** Vì đó là từ phủ định quan trọng, ví dụ “tốt” và “không tốt” có thái độ trái ngược.
- **Tại sao dùng Macro F1?** Vì chỉ số này tính F1 riêng cho từng lớp rồi lấy trung bình, giúp đánh giá công bằng giữa các lớp.
- **Tại sao dùng Logistic Regression?** Nó hiệu quả với vector TF-IDF thưa, huấn luyện nhanh, làm baseline tốt và dễ giải thích.
- **Accuracy cao có chắc tốt không?** Không. Cần xem phân bố lớp, Macro F1, confusion matrix, cross-validation và dữ liệu ngoài miền.
- **Mô hình có hiểu ngữ nghĩa không?** Mô hình học mẫu thống kê của từ và cụm ký tự; khả năng hiểu ngữ cảnh sâu còn hạn chế so với mô hình Transformer.
