# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026 (theo ngày trao đổi; đối chiếu thời điểm chạy thực tế trong Colab nếu khác)

**Runtime Colab:** Đã chạy trên Google Colab; **CPU/GPU cần xác nhận trong Runtime**.

**Python / PyTorch / Ultralytics:** Python và PyTorch: **cần xem output ô thông tin môi trường để điền chính xác**; Ultralytics: `8.4.145` (theo JSON).

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`.

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`.

**Thay đổi so với notebook nguồn:** Chưa ghi nhận thay đổi; cần xác nhận nếu đã sửa mã hoặc tham số.

> **Phân biệt:** Các record bên dưới là *prediction* do mô hình tạo ra, không phải *ground truth* đã được người gán và kiểm tra. Những nhận xét về vị trí/chất lượng cần kiểm tra bằng ảnh PNG kèm theo.

## 1. Phân loại ảnh

**Nguồn:** `day1_lab_outputs/classification_predictions.json`, ảnh `traffic`.

1. **Record hạng 1:** `class_id = 468`; `class_name = cab`; `rank = 1`; `score = 0.510915`; `taxonomy_name = ImageNet-1K`.

2. **Phạm vi:** Record này dự đoán một nhãn cho **toàn bộ bức ảnh**, không mô tả một vật thể riêng biệt, vì tác vụ là `image_classification` và không có tọa độ box/mask trong record.

3. **Nguồn danh sách lớp:** Tên lớp không phải do mô hình tự đặt ra trong lúc suy luận; chúng đến từ taxonomy `ImageNet-1K` gắn với checkpoint `yolo11n-cls.pt`.

4. **Vì sao ghi đủ ba trường:** `class_id` là mã số định danh lớp trong bộ nhãn; `class_name` giúp con người đọc được; `taxonomy_name` xác định bộ nhãn mà mã số/tên lớp thuộc về, tránh nhầm lẫn giữa các taxonomy khác nhau.

5. **Guideline cho ảnh nhiều vật:** Quy định rõ tiêu chí chọn một nhãn chính, ví dụ dựa vào đối tượng/chủ đề nổi bật nhất và mức độ bao phủ hoặc ý nghĩa của ảnh; quy định cách xử lý khi hai chủ đề tương đương hoặc không có lớp thích hợp. Nếu không đủ căn cứ, chuyển ca mơ hồ cho người phụ trách, không tự chọn tùy ý.

6. **Không dùng score làm ground truth:** `score = 0.510915` là điểm dự đoán của mô hình cho lớp `cab`, không phải bằng chứng nhãn đó đúng. Ground truth phải do người gán theo guideline và được QC xác minh; mô hình có thể đoán nhầm ngay cả khi score cao.

## 2. Phát hiện vật thể

**Nguồn:** `day1_lab_outputs/detection_predictions.json`, ảnh `kitchen`.

1. **Một record:** `class_name = person`; `score = 0.912625`; `bbox_xyxy = [385.33, 69.24, 498.92, 348.92]`; `bbox_width = 113.58` pixel; `bbox_height = 279.68` pixel.

2. **Giải thích box:** Ảnh rộng 640, cao 427 pixel. Gốc tọa độ nằm ở góc trên-trái ảnh. Box của người có góc trên-trái xấp xỉ `(385.33, 69.24)` và góc dưới-phải `(498.92, 348.92)`. Chiều rộng xấp xỉ `498.92 − 385.33 = 113.59` pixel (JSON lưu `113.58` do sai số làm tròn); chiều cao `348.92 − 69.24 = 279.68` pixel.

3. **Hai ngưỡng:** Từ JSON đã xuất tại `0.35`, ảnh `kitchen` có **11** prediction. Trong 11 record này, có **6** record với `score >= 0.60`. **Lưu ý:** Số 6 là kết quả *lọc lại JSON 0.35 theo score*, không khẳng định chính xác số box khi **chạy lại mô hình** với `conf=0.60`. Để hoàn tất đúng yêu cầu, cần so với số in ở ô thử threshold `0.35`/`0.60` trong notebook; sửa nếu khác.

4. **Khi hạ ngưỡng:** Reviewer phải kiểm tra nhiều box hơn; loại box dự đoán sai, box trùng, nhãn lớp sai, tọa độ không sát vật thể và các vật có độ tin cậy thấp. Ngưỡng không thay thế QC.

5. **Quy tắc vẽ box:** Với mỗi vật thể nhìn thấy đủ rõ và thuộc taxonomy, vẽ một khung chữ nhật nhỏ nhất có thể bao phần *nhìn thấy* của vật thể, không gộp hai vật thể riêng thành một box; thống nhất cách xử lý phần bị che/cắt mép trong guideline.

6. **Một trường hợp cần xem ảnh:** Record `oven` trong ảnh `kitchen` có `bbox_xyxy = [0.11, 187.54, 195.96, 292.81]`, tức box gần sát mép trái (x≈0). Đây là **ứng viên vật thể bị cắt mép**. Quy tắc chung nên quy định vẽ box chỉ cho phần nhìn thấy và đánh dấu trường hợp cắt mép; **cần mở `detection_predictions.png` kiểm tra** trước khi khẳng định lò nướng thực sự bị cắt. Nếu không phân biệt được vật thể hoặc phạm vi box thì escalation tới Lab Coach.

## 3. Phân đoạn theo từng đối tượng

**Nguồn:** `day1_lab_outputs/segmentation_predictions.json`, ảnh `kitchen`.

1. **Một record:** `instance_id = kitchen-002`; `class_name = bowl`; `score = 0.735743`; `polygon_point_count = 67`; 5 điểm đầu của `polygon_xy` là `[[53.0, 344.0], [52.0, 345.0], [50.0, 345.0], [49.0, 346.0], [48.0, 346.0]]`.

2. **Khác với box:** Box chỉ cho biết hình chữ nhật bao vật thể. Polygon chứa nhiều tọa độ nối thành đường bao, nhờ đó thể hiện hình dạng/vùng chiếm chỗ của đối tượng chi tiết hơn, nhất là vật thể cong hoặc không có dạng chữ nhật.

3. **Ý nghĩa instance_id:** `kitchen-002` đánh dấu **một đối tượng được dự đoán cụ thể trong kết quả này**. Đây không phải `class_id` (mã lớp `bowl`) và cũng không phải tracking ID theo dõi cùng vật thể qua nhiều khung hình/video.

4. **Quy tắc vẽ mask:** Vẽ đường biên mask sát **vùng nhìn thấy** của từng vật thể, tránh tô lan sang nền hoặc vật bên cạnh; các vật cùng lớp phải có instance riêng. Không tự vẽ thêm phần bị che nếu guideline không quy định suy luận phần khuất.

5. **Tình huống biên khó:** Record `kitchen-004` (`potted plant`) có polygon bắt đầu tại `(1, 2)` và nhiều điểm sát cạnh trái ảnh (`x≈1`), gợi ý một vùng nằm sát/có thể bị cắt bởi mép ảnh. **Cần xem `segmentation_prediction.png`** để kiểm tra đường biên có thực sự bị cắt hay không. Với trường hợp cắt mép thường gặp, guideline nên chỉ rõ mask dừng tại mép ảnh; nếu ranh giới giữa cây/chậu và nền bị mơ hồ đến mức không nhất quán, chuyển Lab Coach quyết định.

## 4. Vòng đời và kiểm tra chất lượng

Vòng đời áp dụng khi tạo dữ liệu production: **ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework**. Trong lab này chỉ xem prediction của mô hình có sẵn, không trực tiếp tạo ground truth hay huấn luyện lại.

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp chính cho mỗi ảnh theo taxonomy/guideline. | Ảnh `kitchen` có top-1 `gong` (0.420192), trong khi tên mẫu là kitchen: đây là dấu hiệu **cần kiểm tra bằng ảnh**, không đủ chứng minh AI sai chỉ bằng tên mẫu. | Xem toàn ảnh, chọn nhãn chính theo guideline và đánh dấu trường hợp mơ hồ. | Đối chiếu nhãn với ảnh và quy tắc; kiểm tra nhãn có thuộc taxonomy, yêu cầu sửa nếu chưa phù hợp. |
| Phát hiện vật thể | Với mỗi vật thể: lớp và box (tọa độ bốn góc) được gán bởi người. | Box `oven` ở ảnh kitchen sát cạnh trái (x=0.11), có khả năng cần quy tắc đối với vật cắt mép. | Vẽ box sát phần vật thể thấy được, gán lớp, đánh dấu vật thể che khuất/cắt mép theo quy tắc. | Kiểm tra thiếu/thừa/trùng box, lớp và độ sát mép, yêu cầu rework khi cần. |
| Instance segmentation | Mỗi vật thể có nhãn lớp, instance riêng và vùng mask/polygon theo biên nhìn thấy. | Polygon của `potted plant` gần mép trái ảnh; cần xem PNG để đánh giá biên và cắt mép. | Vẽ mask theo biên nhìn thấy, tách riêng các instance và đánh dấu chỗ mơ hồ. | Kiểm tra mask lan nền, bỏ sót vùng, gộp nhầm hai đối tượng và gửi ca bất định lên Lab Coach. |

## 5. An toàn dữ liệu

1. **Quy tắc bảo vệ dữ liệu:** Chỉ dùng ảnh COCO công khai và output đúng phạm vi bài lab, không đưa thông tin cá nhân, ảnh riêng, dữ liệu khách hàng hay dữ liệu không có quyền sử dụng vào repo công khai.

2. **Khi gặp dữ liệu ngoài phạm vi:** Dừng xử lý/không upload hoặc chia sẻ và báo cho **Lab Coach/giảng viên phụ trách** để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

- [ ] `REPORT.md` (đã soạn; cần xác nhận thông tin môi trường và xem PNG)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu cá nhân trong báo cáo
- [ ] Đã nộp link repository trên LMS
