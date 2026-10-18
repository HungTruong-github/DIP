# BÀI TẬP GIỮA KỲ: NHẬP MÔN XỬ LÝ ẢNH SỐ (505060)
## Đề tài: Nhận diện biển báo giao thông hình tròn trong video bằng thị giác máy tính kinh điển

**Trường Đại học Tôn Đức Thắng - Khoa Công nghệ Thông tin**  
**Học kỳ 1 - Năm học 2026 - 2027**  
**Giảng viên phụ trách:** TS. Trần Văn Hùng (GVBM)

---

### Danh sách sinh viên thực hiện:
| STT | Họ và tên | MSSV | Vai trò |
| :---: | :--- | :---: | :---: |
| 1 | **Trương Dương Hưng** | **52300028** | Trưởng nhóm |
| 2 | **Lương Minh Tân** | **52300061** | Thành viên |
| 3 | **Lê Quang Thành** | **52300064** | Thành viên |

---

### 1. Giới thiệu đề tài
Hệ thống tự động phát hiện và vẽ khung bao chữ nhật (bounding box) màu xanh lá cây xung quanh tất cả các biển báo giao thông hình tròn (Circular Traffic Signs) xuất hiện ở dải phân cách giữa và lề đường bên phải trong 02 video hành trình giao thông thực tế tại Việt Nam (`video1.mp4` và `video2.mp4`).

**Quy định thực hiện:**
- Chỉ sử dụng các thư viện chuẩn cho phép: `cv2` (OpenCV), `numpy`, `math`.
- Tuyệt đối không dùng mô hình học sâu (Deep Learning) hay mô hình tiền huấn luyện (Pre-trained Models).
- Không gán cứng (hardcode) số thứ tự frame; một mã nguồn duy nhất xử lý cho mọi video đầu vào.
- Video kết quả giữ nguyên định dạng MP4, độ phân giải gốc 1920x1080, không gian màu và FPS (29.97).
- Chèn MSSV của nhóm sinh viên vào tất cả các khung hình của video đầu ra (`MSSV: 52300028 - 52300061 - 52300064`).

---

### 2. Các tập tin nộp bài chính thức
- **`52300028_52300061_52300064.py`**: Mã nguồn Python xử lý toàn bộ quy trình nhận diện và xuất video.
- **`52300028_52300061_52300064.pdf`**: Báo cáo chi tiết định dạng PDF (tuân thủ 100% mẫu báo cáo quy định của Khoa CNTT - TDTU).

---

### 3. Quy trình xử lý chính (Methodology)
1. **Chuyển đổi không gian màu (BGR $\rightarrow$ HSV):** Khắc phục hiện tượng biến thiên độ sáng, ánh nắng mặt trời và bóng râm.
2. **Phân đoạn ngưỡng màu kép:** Lọc dải màu đỏ (kết hợp dải thấp $[0, 70, 50]-[10, 255, 255]$ và dải cao $[155, 70, 50]-[180, 255, 255]$) và dải màu xanh dương $[100, 90, 50]-[135, 255, 255]$.
3. **Phép toán hình thái học (Morphological Closing):** Sử dụng kernel elip $3 \times 3$ với 2 lần lặp hàn gắn chi tiết nội dung, tạo các mảng nhị phân đặc liền mạch.
4. **Giới hạn vùng quan tâm không gian (ROI Filter):** Giới hạn $x \ge 0.35 \times W$ và $y \in [25, 520]$ để loại trừ mặt đường, vạch sơn kẻ đường và đèn xe máy.
5. **Trích xuất đường bao & Phân tích hình học:** Đánh giá diện tích, tỷ lệ khung bao $0.70 \le W/H \le 1.40$, chỉ số độ tròn $C = \frac{4\pi A}{P^2} \ge 0.78$ và tỷ lệ bao hình lồi Convex Hull $\ge 0.70$.
6. **Xác thực màu trên vành khuyên (Annulus Verification):** Kiểm tra tỷ lệ pixel màu của biển trên vành ngoài bán kính $[0.60R, 1.0R] \ge 28\%$, triệt tiêu hoàn toàn nhiễu từ trang phục, đèn giao thông hay biển tam giác màu vàng.
7. **Triệt tiêu hộp trùng lặp (Non-Maximum Suppression - NMS):** Hợp nhất các bounding box lồng nhau trên các biển báo đa màu (như biển cấm đỗ P.130), tránh phạt trừ điểm trùng lặp theo Rubric.
8. **Chèn thông tin MSSV & Xuất video:** Vẽ bounding box màu xanh lá cây, dán watermark MSSV nhóm và ghi video chuẩn MP4 codec `mp4v`.

---

### 4. Liên kết Google Drive tải Video kết quả
Theo yêu cầu đề bài, các video kích thước lớn được lưu trữ và chia sẻ công khai trên Google Drive:
- **Thư mục Google Drive xem và tải video kết quả:**  
  [https://drive.google.com/drive/folders/15IMKKVqdRqsybV7-hVhwEgIMch1mKz1H?usp=sharing](https://drive.google.com/drive/folders/15IMKKVqdRqsybV7-hVhwEgIMch1mKz1H?usp=sharing)
- **Tên các video kết quả:**
  - `52300028_52300061_52300064_video1.mp4` (Thời lượng: ~103s, 3087 frames, 1920x1080)
  - `52300028_52300061_52300064_video2.mp4` (Thời lượng: ~76s, 2278 frames, 1920x1080)

---

### 5. Hướng dẫn chạy chương trình
Yêu cầu cài đặt Python 3.8+ và các thư viện:
```bash
pip install opencv-python numpy
```

Chạy chương trình:
```bash
python 52300028_52300061_52300064.py
```
Chương trình sẽ tự động kiểm tra `video1.mp4` và `video2.mp4` trong cùng thư mục, hiển thị thanh tiến trình xử lý và tạo ra hai tệp video đầu ra tương ứng.
