# Hướng Dẫn Thực Hành — Preference Alignment Lab (DPO & ORPO)

> **Sinh viên:** Nguyễn Ngọc Dương — MSSV: 2A202601717  
> **Thời lượng:** ~2 giờ  
> **Mục tiêu:** Hiểu và cài đặt pipeline alignment theo phong cách production với DPO/ORPO.

---

## ⚡ Chuẩn bị môi trường (trước khi bắt đầu)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
make test
```

> Cài thêm nếu muốn chạy training thực sự:
> ```bash
> pip install -e '.[dev,train]'
> ```

---

## 📋 Danh sách nhiệm vụ

### ✅ Milestone 1 — Thiết lập & Kiểm tra dữ liệu mẫu (0–30 phút)

- [ ] Chạy `make test` và đảm bảo tất cả test đều pass.
- [ ] Khám phá thư mục `data/` — quan sát cấu trúc dữ liệu preference pairs (`prompt`, `chosen`, `rejected`).
- [ ] Đọc file cấu hình trong `configs/local.yaml`.
- [ ] Hiểu cấu trúc repository:

  | Thư mục / File | Mục đích |
  |---|---|
  | `src/preference_lab/` | Package Python chính |
  | `data/` | Tập dữ liệu preference mẫu |
  | `configs/` | File cấu hình YAML |
  | `docs/` | Hướng dẫn, rubric, data card |
  | `scripts/` | Các script tiện ích |
  | `tests/` | Unit tests cho phần sinh viên làm |

---

### ✅ Milestone 2 — Cài đặt Data Loader & Collator (30–50 phút)

> **File cần chỉnh sửa:** tìm các khối `TODO(student)` trong phần xử lý dữ liệu.

- [ ] Cài đặt hàm load file JSONL với xử lý lỗi rõ ràng (kèm số dòng trong thông báo lỗi).
- [ ] Thêm kiểm tra dữ liệu trùng lặp (duplicate check).
- [ ] Validate schema: mỗi mẫu phải có đủ 3 trường `prompt`, `chosen`, `rejected`.
- [ ] Chia tập train/validation theo **prompt** (không chia theo dòng để tránh data leakage).
- [ ] Chạy kiểm tra:

  ```bash
  pytest tests/test_data.py
  ```

---

### 🔵 Milestone 2.5 — (Tùy chọn) Sinh dữ liệu tổng hợp (50–70 phút)

> Cần có OpenAI API key.

- [ ] Xuất biến môi trường API key:

  ```bash
  export OPENAI_API_KEY=your_key
  ```

- [ ] Chạy script sinh dữ liệu:

  ```bash
  python scripts/generate_data.py --count 10 --domain "python coding"
  ```

- [ ] Kiểm tra chất lượng dữ liệu được sinh ra.

---

### ✅ Milestone 3 — Cài đặt Loss Function: DPO hoặc ORPO (70–100 phút)

> **File cần chỉnh sửa:** `src/preference_lab/losses.py`

- [ ] Chọn một trong hai phương pháp:
  - **DPO** (Direct Preference Optimization)
  - **ORPO** (Odds Ratio Preference Optimization)

- [ ] Cài đặt hàm loss theo phương pháp đã chọn tại khối `TODO(student)`.
- [ ] Xử lý ổn định số học: tránh `log(0)`, xử lý giá trị logprob cực đoan (clamp, log1p...).
- [ ] Thiết lập hyperparameter phù hợp:
  - `beta` (cho cả DPO và ORPO)
  - `lambda_orpo` (chỉ dùng nếu chọn ORPO)

- [ ] Chạy kiểm tra:

  ```bash
  pytest tests/test_losses.py
  ```

---

### ✅ Milestone 4 — Cài đặt Evaluation (100–115 phút)

> **File cần chỉnh sửa:** phần evaluation trong package.

- [ ] Thay thế mock scores bằng điểm số thực từ model hoặc bộ scorer xác định (CPU mode).
- [ ] Tính toán và lưu các metric:
  - **Pairwise Accuracy** (%)
  - **Final Loss**

- [ ] Chạy pipeline evaluation:

  ```bash
  pref-lab evaluate --config configs/local.yaml
  ```

- [ ] Kiểm tra kết quả được lưu vào `outputs/metrics.json`:

  ```bash
  cat outputs/metrics.json
  ```

- [ ] Chạy **Safety Regression Prompts** (xem `docs/regression_prompts.md`):
  - [ ] Prompt yêu cầu lời khuyên y tế nguy hiểm cao.
  - [ ] Prompt tóm tắt với giới hạn từ nghiêm ngặt.
  - [ ] Prompt mà model phải thừa nhận không chắc chắn.
  - [ ] Prompt xử lý sự cố với context thiếu.

---

### ✅ Milestone 5 — Viết báo cáo & Demo (115–120 phút)

> **File cần điền:** `docs/REPORT_TEMPLATE.md` và `docs/data_card_template.md`

- [ ] **Data Card** (`docs/data_card_template.md`): Điền đầy đủ các mục:
  - Tên dataset, nguồn, giấy phép.
  - Schema, rubric gán nhãn.
  - Bias đã biết, kiểm tra safety/PII.
  - Phương pháp chia tập.

- [ ] **Báo cáo thực nghiệm** (`docs/REPORT_TEMPLATE.md`): Điền đầy đủ 4 mục:
  1. Phân tích & làm sạch dataset.
  2. Lý do chọn DPO/ORPO, hyperparameter, xử lý ổn định số học.
  3. Kết quả evaluation (bảng metrics + phân tích định tính).
  4. Thảo luận: điều gì hoạt động tốt, bias quan sát được, kết quả safety.

- [ ] Demo 1 phút: trình bày kết quả từ `outputs/metrics.json`.

---

## 🏁 Production Checklist (trước khi nộp)

- [ ] Schema dataset đã được validate.
- [ ] Train/eval split theo prompt, không phải theo dòng.
- [ ] Config đã được commit; artifacts/output đã được gitignore.
- [ ] Metrics đã được lưu dạng JSON.
- [ ] Safety regression prompts đã chạy trước và sau khi training.
- [ ] Data card đã được cập nhật.
- [ ] Tất cả unit tests đều pass (`make test`).

---

## 📌 Quy tắc Lab

1. **Không** viết lại toàn bộ repository.
2. Chỉ cài đặt các khối `TODO(student)` trừ khi có lý do rõ ràng.
3. Giữ cho tests pass sau mỗi milestone.
4. **Không** commit secrets, model weights, hoặc dữ liệu riêng tư.

---

## 📁 Tài liệu tham khảo

| File | Nội dung |
|---|---|
| [README.md](./README.md) | Tổng quan dự án & quickstart |
| [docs/lab_guide.md](./docs/lab_guide.md) | Hướng dẫn chi tiết từng task |
| [docs/REPORT_TEMPLATE.md](./docs/REPORT_TEMPLATE.md) | Template báo cáo thực nghiệm |
| [docs/data_card_template.md](./docs/data_card_template.md) | Template data card |
| [docs/regression_prompts.md](./docs/regression_prompts.md) | Danh sách prompt kiểm tra safety |
