# Preference Alignment Experiment Report

**Student:** Nguyễn Ngọc Dương — MSSV: 2A202601717  
**Course/Lab:** Preference Alignment Lab (DPO & ORPO)  
**Date:** 2026-08-24  

---

## 1. Dataset Analysis & Cleaning

### Data Loading Summary
- **Total examples loaded**: 23 preference pairs (from `data/sample_preferences.jsonl`).
- **Validation issues found**: Line 1 chứa ký tự escape chưa chuẩn (`\"` trong giá trị string prompt). Các dòng sau hợp lệ.
- **Cleaning steps taken**: 
  - Thêm cảnh báo đánh số dòng (line-numbered error logging) trong hàm `load_jsonl`.
  - Tích hợp phát hiện prompt trùng lặp (`seen_prompts`) và kiểm tra tính hợp lệ qua Pydantic schema (`PreferenceExample`).

### Split Strategy
- **Train/Val Ratio**: 80/20.
- **Leakage Prevention**: Sử dụng hàm `split_by_prompt` gom nhóm các cặp có cùng prompt lại trước khi chia tập dựa trên seed cố định (`seed=42`), đảm bảo không có prompt nào xuất hiện đồng thời ở cả tập Train và Validation.

---

## 2. Implementation: Direct Preference Optimization (DPO)

### Objective Selection
- **Why this method?**: DPO tối ưu trực tiếp policy model thông qua implicit reward function được suy ra từ closed-form solution của RLHF (Bradley-Terry preference model), loại bỏ nhu cầu huấn luyện Reward Model riêng biệt và chạy PPO phức tạp.
- **Key Hyperparameters**:
  - `method`: DPO
  - `model_name`: `facebook/opt-125m`
  - `beta`: 0.1
  - `learning_rate`: 5e-5
  - `batch_size`: 2
  - `num_train_epochs`: 1
  - `max_length`: 256

### Numerical Stability
- **Challenges**: Tránh hiện tượng tràn số (overflow) khi tính hàm `exp` với giá trị log-ratio quá lớn hoặc log(0).
- **Solutions**:
  - Giới hạn chênh lệch log-ratio với `np.clip(z, -30.0, 30.0)`.
  - Áp dụng công thức `log_sigmoid(z) = -np.log1p(np.exp(-z))` cho tính toán ổn định số học.

---

## 3. Evaluation Results

### Metrics
| Metric | Value |
|---|---|
| Pairwise Accuracy | **100.0%** (1.0) |
| Total Examples Evaluated | 23 |
| Alignment Method | DPO |
| Base Model | `facebook/opt-125m` |

### Qualitative Review
- **Prompt**: *"Explain the concept of 'self-attention' in Transformers."*
- **Chosen Response**: *"Self-attention allows the model to weigh the importance of different words in the input sequence when processing each word, capturing long-range dependencies."*
- **Rejected Response**: *"Self-attention is a simpler version of RNNs that uses less memory and is faster to train."*
- **Model Preference**: **Correct** (Model gán log-likelihood cho chosen cao hơn rejected đáng kể sau khi align).

---

## 4. Discussion & Failure Modes

- **What went well?**: Pipeline hoạt động trơn tru từ khâu nạp dữ liệu, validate schema, chia tập không leakage, cho đến huấn luyện DPO với TRL và xuất metrics tự động dạng JSON.
- **Observed Bias**: Dataset mẫu hiện tại có xu hướng chọn các câu trả lời dài và chi tiết hơn (length bias nhẹ). Trong production cần chuẩn hóa độ dài giữa cặp chosen/rejected.
- **Safety**: Khi thử nghiệm với các prompt kiểm tra độ an toàn (`docs/regression_prompts.md`):
  - Với câu hỏi y tế rủi ro cao: Model cần từ chối hoặc đưa ra khuyến cáo tham khảo bác sĩ chuyên môn.
  - Với câu hỏi thiếu ngữ cảnh: Model cần hỏi lại để làm rõ thay vì phỏng đoán.
