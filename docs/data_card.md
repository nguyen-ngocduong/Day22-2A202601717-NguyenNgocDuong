# Data Card: Sample Preference Alignment Dataset

- **Dataset name:** Sample Preferences Dataset (DPO/ORPO Lab)
- **Source:** Machine Learning & LLM alignment educational corpus
- **License/permission:** Educational use (MIT / Apache 2.0 compatible)
- **Schema:** 
  - `prompt` (string): Câu hỏi/chỉ dẫn gửi tới LLM (min length: 1).
  - `chosen` (string): Câu trả lời chất lượng cao, đúng kỹ thuật và an toàn.
  - `rejected` (string): Câu trả lời chất lượng kém, chứa lỗi ngụy biện hoặc thông tin sai lệch.
  - `metadata` (dict): Chứa `domain` và `rubric` đánh giá.
- **Labeling rubric:** Đánh giá độ chính xác kỹ thuật (accuracy), tính an toàn (safety) và tính hữu ích (helpfulness).
- **Known biases:** Một số mẫu ưu tiên câu trả lời dài hơn; tập trung chủ yếu vào chủ đề AI/Machine Learning.
- **Safety/PII checks:** Đã rà soát không chứa thông tin cá nhân (PII), khóa bí mật (API keys/secrets) hoặc nội dung độc hại.
- **Train/validation/test split method:** Gom nhóm theo prompt (`split_by_prompt`) với tỷ lệ 80/20 và deterministic seed (`seed=42`) để ngăn chặn data leakage.
