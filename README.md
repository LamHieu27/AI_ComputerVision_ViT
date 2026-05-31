# 🩺 HỆ THỐNG NHẬN DIỆN TỔN THƯƠNG PHỔI BẰNG VISION TRANSFORMER (ViT)
## Đồ án Môn học / Nghiên cứu Chuyên sâu - Xử lý ảnh Y tế với NIH Chest X-ray 14

Dự án này xây dựng một hệ thống hoàn chỉnh phục vụ nghiên cứu và thực nghiệm lâm sàng: Chẩn đoán 14 hội chứng tổn thương phổi từ ảnh chụp X-Ray kỹ thuật số bằng kiến trúc tiên tiến **Vision Transformer (ViT)**. 

Hệ thống được phát triển trên nền tảng ngôn ngữ **Python**, tận dụng thế mạnh tối ưu của **PyTorch** cho việc huấn luyện, kiểm thử và chạy ứng dụng chẩn đoán tương tác thời gian thực.

---

## 📂 Cấu trúc Thư mục Dự án (Modular Architecture)

Thư mục dự án được tổ chức khoa học, tách biệt hoàn toàn giữa phần xử lý dữ liệu, mô hình hóa, kịch bản huấn luyện, giao diện ứng dụng và báo cáo học thuật:

```text
f:\Test\ViT\
├── dataset/                     # Thư mục chứa dữ liệu đầu vào (NIH)
│   ├── images-224/images-224/   # Chứa toàn bộ các file ảnh X-Ray (224x224)
│   ├── Data_Entry_2017.csv      # File metadata chứa nhãn bệnh lý dạng chuỗi
│   ├── train_val_list_NIH.txt   # Danh sách phân chia tập huấn luyện/validate gốc
│   └── test_list_NIH.txt        # Danh sách phân chia tập test kiểm thử gốc
├── src/                         # Mã nguồn cốt lõi (Core Package)
│   ├── __init__.py
│   ├── config.py                # Định nghĩa tham số cấu hình hệ thống & dịch tiếng Việt
│   ├── dataset.py               # Dataset PyTorch, tiền xử lý, phân rã không rò rỉ bệnh nhân
│   ├── models/                  # Định nghĩa các kiến trúc Deep Learning
│   │   ├── __init__.py
│   │   └── vit.py               # Vision Transformer tự viết từ Scratch + Pretrained adapter
│   └── utils.py                 # Tiện ích đo lường y tế (AUC-ROC, F1, Plotting curves)
├── checkpoints/                 # Nơi tự động lưu các bản sao trọng số tốt nhất (.pth)
├── reports/                     # Thư mục xuất báo cáo (ROC Curves, Loss graph, CSV report)
├── train.py                     # Kịch bản huấn luyện Vision Transformer trên PyTorch
├── eval.py                      # Kịch bản kiểm thử độc lập trên tập test NIH, vẽ ROC Curves
├── app.py                       # Giao diện Web chẩn đoán tương tác thông minh (Gradio UI)
├── requirements.txt             # Danh mục thư viện Python cần cài đặt
└── README.md                    # Hướng dẫn sử dụng dự án (File này)
```

---
LINK DATASET: https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized

## ⚙️ Hướng dẫn Cài đặt & Chuẩn bị

### 1. Cài đặt các thư viện cần thiết
Đảm bảo bạn đã cài đặt Python 3.10 trở lên. Mở Terminal tại thư mục dự án và chạy lệnh sau:
```bash
pip install -r requirements.txt
```

### 2. Cấu trúc nhãn dữ liệu (14 Bệnh lý)
Hệ thống chẩn đoán đồng thời **14 dạng tổn thương phổi** theo tiêu chuẩn y khoa quốc tế:
1. `Atelectasis` (Xẹp phổi)
2. `Cardiomegaly` (Phì đại cơ tim / Tim to)
3. `Effusion` (Tràn dịch màng phổi)
4. `Infiltration` (Thâm nhiễm phổi)
5. `Mass` (U phổi)
6. `Nodule` (Nốt mờ phổi)
7. `Pneumonia` (Viêm phổi)
8. `Pneumothorax` (Tràn khí màng phổi)
9. `Consolidation` (Đông đặc phổi)
10. `Edema` (Phù phổi)
11. `Emphysema` (Khí phế thũng)
12. `Fibrosis` (Xơ hóa phổi)
13. `Pleural_Thickening` (Dày màng phổi)
14. `Hernia` (Thoát vị cơ hoành)

---

## 🚀 Hướng dẫn Sử dụng Hệ thống

Dự án hỗ trợ 3 giai đoạn hoạt động độc lập qua các tệp script chuyên dụng:

### 1. Huấn luyện Mô hình ViT (`train.py`)
Tiến hành huấn luyện kiến trúc Vision Transformer trên tập dữ liệu ảnh NIH:
- **Chạy chế độ Demo nhanh (Mặc định)**: Chỉ lấy ngẫu nhiên 1000 mẫu ảnh để kiểm tra luồng hoạt động chạy trơn tru trong thời gian ngắn:
  ```bash
  python train.py --model scratch --epochs 5 --batch_size 16 --limit 1000
  ```
- **Huấn luyện toàn diện**: Sử dụng toàn bộ hàng nghìn ảnh thuộc phân chia train/val chính thức:
  ```bash
  python train.py --model pretrained --epochs 10 --batch_size 32 --limit -1
  ```
*Sau khi hoàn thành, file trọng số tốt nhất sẽ tự động lưu vào thư mục `checkpoints/best_vit_scratch.pth` hoặc `checkpoints/best_vit_pretrained.pth`.*

### 2. Kiểm thử & Đánh giá Độc lập (`eval.py`)
Đánh giá hiệu năng mô hình trên tập Test chính thức độc lập với tập huấn luyện:
```bash
python eval.py --model scratch --limit 500
```
- Chương trình sẽ tự động trích xuất bảng báo cáo chi tiết **AUC Score** cho từng bệnh lý.
- Xuất tệp báo cáo dạng bảng Excel `reports/evaluation_report_scratch.csv`.
- Vẽ biểu đồ ROC Curves y học cực kỳ trực quan tại `reports/roc_curves_scratch.png`.

### 3. Khởi chạy Giao diện Web tương tác (`app.py`)
Mở trình duyệt web để chẩn đoán lâm sàng tương tác trực quan qua giao diện Gradio cao cấp:
```bash
python app.py
```
- Mở liên kết được in ở console (ví dụ: `http://127.0.0.1:7860`).
- **Tính năng độc quyền**: Tải ảnh chụp phổi lên, trượt thanh chọn ngưỡng quyết định nhạy cảm y tế để nhận kết quả chẩn đoán tự động phản hồi chớp nhoáng với giao diện sáng bừng trực quan từ mô hình Vision Transformer!

---

## 📖 Cơ sở Lý thuyết & Điểm số Đồ án Cao cấp

Để đạt điểm số tối đa cho môn học hoặc báo cáo tốt nghiệp, đồ án này được tích hợp đầy đủ các giải pháp học thuật cao cấp:

1. **Kháng rò rỉ thông tin bệnh nhân (Patient Leakage Prevention)**:
   - X-Ray thường được chụp nhiều lần trên cùng một bệnh nhân (Patient ID). Nếu phân chia ngẫu nhiên theo ảnh, hình ảnh cùng một bệnh nhân có thể vừa ở tập Train vừa ở tập Validate, dẫn đến hiện tượng học vẹt (overfitting).
   - `NIHChestXrayDataset` thực hiện chia tập dữ liệu **theo danh sách Patient ID độc lập** nhằm đảm bảo tập Validate và Test chứa các bệnh nhân hoàn toàn mới lạ.
2. **Kiến trúc Vision Transformer (ViT) tự thiết kế**:
   - Khác với việc gọi thư viện có sẵn một dòng, tệp `src/models/vit.py` xây dựng chi tiết từng lớp toán học của Transformer (Patch Embedding, Multi-head Attention, MLP Blocks). Điều này chứng minh năng lực hiểu sâu mã nguồn trong phần báo cáo lý thuyết đồ án.
3. **Mô hình Tự chú ý Toàn cục (Global Self-Attention)**:
   - Sử dụng cơ chế Self-Attention cho phép mô hình thu nhận mối quan hệ không gian xa giữa các vùng phổi chụp X-Ray, vượt qua giới hạn vùng nhìn cục bộ (receptive field) của các mạng CNN truyền thống.
