# Hướng dẫn chạy lab trên Kaggle (không cần biết code)

Toàn bộ mã đã viết sẵn trong thư mục `lab10/`. Bạn chỉ cần bấm nút trên Kaggle, chờ, rồi tải một file zip kết quả về.

## Bước 1 · Để repo GitHub ở chế độ công khai (1 lần)

Kaggle cần tải mã từ GitHub, và đề bài cũng yêu cầu repo công khai.

1. Mở https://github.com/tubepvhat1604-bot/K4-Track-4-Day10-VisionLanguageModel
2. Bấm **Settings** (thanh trên cùng của repo) → kéo xuống cuối trang, mục **Danger Zone**.
3. Bấm **Change visibility** → **Change to public** → xác nhận.

## Bước 2 · Chuẩn bị tài khoản Kaggle (1 lần)

1. Đăng nhập https://www.kaggle.com.
2. Bấm ảnh đại diện (góc phải) → **Settings** → mục **Phone Verification** → xác minh số điện thoại.
   Không xác minh thì Kaggle không cho dùng GPU và Internet.

## Bước 3 · Mở notebook trên Kaggle

1. Tải file `kaggle_lab10.ipynb` về máy (có trong repo, hoặc file mình gửi trong thread).
2. Trên Kaggle bấm **Create** (nút + bên trái) → **New Notebook**.
3. Trong notebook mới: menu **File** → **Import Notebook** → chọn file `kaggle_lab10.ipynb` → **Import**.
4. Ở cột bên phải, mục **Session options**:
   - **Accelerator** → chọn **GPU T4 x2**.
   - **Internet** → bật **On**.
   (Nếu không thấy cột bên phải, bấm biểu tượng mũi tên `<` ở mép phải màn hình.)

## Bước 4 · Chạy thử nhanh (≈30 phút, nên làm)

Mục đích: bắt lỗi sớm trước khi chạy thật nhiều giờ.

1. Ô code đầu tiên đang ghi `SMOKE_TEST = True`. Giữ nguyên.
2. Bấm **Run All** (thanh trên cùng).
3. Chờ đến khi ô gần cuối in ra dòng **`XONG, không có lỗi`**.
   - Nếu thấy `XONG, lỗi ở: [...]` hoặc chữ đỏ `Error`: chép đoạn chữ lỗi (hoặc chụp màn hình) gửi lại cho mình, đừng làm tiếp.
4. Điểm số ở bước này đều gần 0 là **bình thường** (chỉ train 5 bước, chấm 8 ảnh).

## Bước 5 · Chạy thật (≈5–7 giờ, chạy nền, có thể tắt máy)

1. Sửa ô code đầu tiên thành `SMOKE_TEST = False`.
2. Bấm **Save Version** (góc phải trên) → chọn **Save & Run All (Commit)** → **Save**.
3. Kaggle sẽ tự chạy ở nền. Bạn có thể đóng trình duyệt. Mỗi tuần Kaggle cho 30 giờ GPU, lần chạy này dùng khoảng 5–7 giờ.
   Muốn tiết kiệm giờ GPU thì đặt `RUN_E5 = False` (bỏ phần thưởng +10 điểm, bớt khoảng 1 giờ).
4. Xem tiến độ: trang notebook → **Version** mới nhất → **Logs**.

## Bước 6 · Tải kết quả và gửi lại

1. Khi version chạy xong (trạng thái **Successful**), mở version đó → tab **Output**.
2. Tải file **`lab10_outputs.zip`** (không cần tải thư mục `adapters`, file đó rất nặng).
3. Gửi file zip đó vào thread này. Mình sẽ viết `REPORT.md`, đưa bằng chứng vào repo và kiểm tra `check_submission.py`.

Nếu version bị dừng giữa chừng (hết giờ, lỗi mạng): mở notebook, **Add Input** → tab **Your Work** → chọn Output của version vừa chạy → chạy lại bước 5. Thí nghiệm nào xong rồi sẽ được bỏ qua.

## Sau khi có báo cáo: những việc chỉ bạn làm được

- Kiểm tra và **merge Pull Request** mình mở trên GitHub (nút xanh **Merge pull request**).
- Nộp link repo công khai theo cách lớp yêu cầu, trước hạn.
- Đọc kỹ `REPORT.md`: giảng viên có thể gọi **vấn đáp 5 phút** về các con số trong đó.

## Mã làm gì (để đối chiếu với GUIDE.md)

| GUIDE | File |
|---|---|
| Bước 2: nhãn YOLO → JSON 0–1000, prompt | `lab10/common.py` |
| Bước 3, 5: nạp 4-bit, LoRA 252/348 module, mở projector fp32 | `lab10/modeling.py` |
| Bước 4: chat template, mask loss, collate | `lab10/data.py` |
| Bước 6: vòng train fp16 + GradScaler, lưu adapter + projector | `lab10/train.py` |
| Bước 3, 7: suy luận theo batch | `lab10/infer.py` |
| Bước 8: E4 (`data_size` 0.1 và 1.0 trên setup tốt nhất) | `lab10/run.py` |
| Bước 9: reward IoU + test, GRPO | `lab10/reward.py`, `lab10/tests/test_reward.py`, `lab10/grpo.py` |
| Ảnh minh hoạ đúng/sai | `lab10/figures.py` |
| Chạy tất cả | `python -m lab10.run all` |
