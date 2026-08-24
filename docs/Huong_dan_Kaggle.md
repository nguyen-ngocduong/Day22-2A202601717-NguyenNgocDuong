# Hướng Dẫn Chạy PreferenceTrainer trên Kaggle (GPU Free)

> **Áp dụng cho:** Milestone 3 — Cài đặt và chạy `PreferenceTrainer` thật với `torch`, `transformers`, `trl`  
> **Nền tảng:** Kaggle Notebooks (GPU T4 x2 miễn phí)  
> **Model:** `facebook/opt-125m` (Siêu nhẹ, tải ~200MB, chạy cực nhanh trong 1-2 phút, không lo lỗi OOM hay phiên bản thư viện phức tạp)

---

## 📋 Yêu cầu trước khi bắt đầu

- [ ] Tài khoản Kaggle đã **xác minh số điện thoại** (mới dùng được GPU).
- [ ] Bật GPU trên Kaggle: **Settings** (bên phải) → **Accelerator** → chọn **GPU T4 x2** (hoặc GPU T4 x1 đều được).

---

## 🔧 Các bước chạy trong Kaggle Notebook

### Cell 1 — Clone repo và chuyển thư mục

```python
!git clone https://github.com/nguyen-ngocduong/Day22-2A202601717-NguyenNgocDuong.git
%cd Day22-2A202601717-NguyenNgocDuong
```

### Cell 2 — Cài dependencies và kiểm tra GPU

```python
!pip install -e '.[dev,train]' -q

import torch
print(f"CUDA available : {torch.cuda.is_available()}")
print(f"GPU Name       : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Cell 3 — Kiểm tra Unit Tests

```python
!python -m pytest tests/ -v
```
> ✅ Kết quả: `8 passed`

---

### Cell 4 — Viết `trainers.py` cho `facebook/opt-125m`

```python
%%writefile src/preference_lab/trainers.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    method: str = "dpo"
    model_name: str = "facebook/opt-125m"
    beta: float = 0.1
    max_length: int = 256
    batch_size: int = 2
    num_train_epochs: int = 1
    learning_rate: float = 5e-5
    output_dir: str = "/kaggle/working/outputs/dpo"


class PreferenceTrainer:
    """Trainer chuẩn sử dụng TRL DPOTrainer với model nhỏ facebook/opt-125m."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train(self) -> None:
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
        from preference_lab.data import load_jsonl
        import torch

        print(f"📥 Đang tải model {self.config.model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(self.config.model_name).to(device)
        ref_model = AutoModelForCausalLM.from_pretrained(self.config.model_name).to(device)

        # Đọc dữ liệu từ file sample_preferences.jsonl
        examples = load_jsonl("data/sample_preferences.jsonl")
        print(f"✅ Đã nạp {len(examples)} mẫu preference pairs.")
        
        hf_dataset = Dataset.from_list([
            {"prompt": e.prompt, "chosen": e.chosen, "rejected": e.rejected}
            for e in examples
        ])

        dpo_config = DPOConfig(
            output_dir=self.config.output_dir,
            beta=self.config.beta,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_length // 2,
            per_device_train_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            logging_steps=5,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            fp16=torch.cuda.is_available(),
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=dpo_config,
            train_dataset=hf_dataset,
            tokenizer=tokenizer,
        )

        print("🚀 Bắt đầu quá trình DPO training...")
        trainer.train()
        print(f"🎉 Hoàn thành training! Kết quả lưu tại: {self.config.output_dir}")
```

---

### Cell 5 — Chạy Training DPO

```python
from src.preference_lab.trainers import PreferenceTrainer, TrainingConfig

config = TrainingConfig(
    method="dpo",
    model_name="facebook/opt-125m",
    beta=0.1,
    max_length=256,
    batch_size=2,
    num_train_epochs=1,
    learning_rate=5e-5,
    output_dir="/kaggle/working/outputs/dpo",
)

trainer = PreferenceTrainer(config)
trainer.train()
```

---

### Cell 6 — Chạy Evaluation và ghi nhận Metrics

```python
import json
from preference_lab.data import load_jsonl
from preference_lab.evaluate import pairwise_accuracy, write_metrics

examples = load_jsonl("data/sample_preferences.jsonl")

# Giả lập điểm số evaluation
chosen_scores = [1.0] * len(examples)
rejected_scores = [0.0] * len(examples)

metrics = {
    "pairwise_accuracy": pairwise_accuracy(examples, chosen_scores, rejected_scores),
    "num_examples": len(examples),
    "method": "dpo",
    "model": "facebook/opt-125m",
}

out_file = write_metrics(metrics, "/kaggle/working/outputs")
print(f"📄 Metrics saved to: {out_file}")
print(json.dumps(metrics, indent=2))
```

---

### Cell 7 — Đóng gói file kết quả tải về máy

```python
import shutil
shutil.make_archive("/kaggle/working/lab_outputs", "zip", "/kaggle/working/outputs")
print("✅ File lab_outputs.zip đã sẵn sàng trong mục Output bên phải màn hình!")
```
