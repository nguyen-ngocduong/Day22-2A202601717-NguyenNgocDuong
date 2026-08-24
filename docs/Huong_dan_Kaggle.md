# Hướng Dẫn Chạy PreferenceTrainer trên Kaggle (GPU Free)

> **Áp dụng cho:** Milestone 3 — Cài đặt và chạy `PreferenceTrainer` thật với `torch`, `transformers`, `trl`, `peft`  
> **Nền tảng:** Kaggle Notebooks (GPU T4 x2 miễn phí ~30h/tuần)  
> **Model đề xuất:** `facebook/opt-125m` (nhỏ nhất, chạy nhanh trên T4)

---

## 📋 Yêu cầu trước khi bắt đầu

- [ ] Tài khoản Kaggle đã **xác minh số điện thoại** (mới dùng được GPU).
- [ ] Repo GitHub đã được **đặt là Public** (để Kaggle clone được).
  - Vào: `https://github.com/nguyen-ngocduong/Day22-2A202601717-NguyenNgocDuong` → Settings → Danger Zone → Change visibility → Public

---

## 🚀 Bước 1 — Tạo Notebook mới trên Kaggle

1. Vào [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. Đặt tên: `DPO-ORPO-PreferenceTrainer`
3. Bật GPU: **Settings** (bên phải) → **Accelerator** → chọn **GPU T4 x2**
4. Chọn **Language: Python**

---

## 🔧 Bước 2 — Clone repo và cài đặt

Chạy từng cell sau trong notebook:

### Cell 1 — Clone repo

```python
# Clone project từ GitHub
!git clone https://github.com/nguyen-ngocduong/Day22-2A202601717-NguyenNgocDuong.git
%cd Day22-2A202601717-NguyenNgocDuong
```

### Cell 2 — Cài dependencies (dev + train)

```python
# Cài [dev] + [train] — bước này mất ~3-5 phút
!pip install -e '.[dev,train]' -q

# Kiểm tra torch có nhận GPU không
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

> ✅ Kết quả mong đợi: `CUDA available: True` và tên GPU hiện ra.

### Cell 3 — Chạy tests cơ bản (không cần GPU)

```python
!python -m pytest tests/ -v
```

> ✅ Kết quả mong đợi: `8 passed`

---

## 🏋️ Bước 3 — Implement PreferenceTrainer (phần việc chính)

### Cell 4 — Xem skeleton cần implement

```python
# Xem file trainers.py cần implement
!cat src/preference_lab/trainers.py
```

### Cell 5 — Implement TRL-backed DPO Trainer

Tạo file mới hoặc chỉnh sửa `src/preference_lab/trainers.py`:

```python
%%writefile src/preference_lab/trainers.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass(frozen=True)
class TrainingConfig:
    method: str
    beta: float = 0.1
    lambda_orpo: float = 0.1
    max_length: int = 512
    batch_size: int = 2
    model_name: str = "facebook/opt-125m"
    output_dir: str = "outputs/checkpoints"
    num_train_epochs: int = 1
    learning_rate: float = 1e-5


class PreferenceTrainer:
    """TRL-backed DPO/ORPO trainer."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train(self) -> None:
        """Dispatch to DPO or ORPO trainer via TRL."""
        if self.config.method == "dpo":
            self._train_dpo()
        elif self.config.method == "orpo":
            self._train_orpo()
        else:
            raise ValueError(f"Unknown method: {self.config.method}. Choose 'dpo' or 'orpo'.")

    def _train_dpo(self) -> None:
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
        from preference_lab.data import load_jsonl

        print(f"[DPO] Loading model: {self.config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        ref_model = AutoModelForCausalLM.from_pretrained(self.config.model_name)

        # Load data → HuggingFace Dataset
        examples = load_jsonl("data/sample_preferences.jsonl")
        hf_data = Dataset.from_list([
            {"prompt": e.prompt, "chosen": e.chosen, "rejected": e.rejected}
            for e in examples
        ])

        dpo_config = DPOConfig(
            output_dir=self.config.output_dir,
            beta=self.config.beta,
            max_length=self.config.max_length,
            per_device_train_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=dpo_config,
            train_dataset=hf_data,
            tokenizer=tokenizer,
        )
        print("[DPO] Starting training...")
        trainer.train()
        print(f"[DPO] Done! Checkpoints saved to {self.config.output_dir}")

    def _train_orpo(self) -> None:
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import ORPOConfig, ORPOTrainer
        from preference_lab.data import load_jsonl

        print(f"[ORPO] Loading model: {self.config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(self.config.model_name)

        examples = load_jsonl("data/sample_preferences.jsonl")
        hf_data = Dataset.from_list([
            {"prompt": e.prompt, "chosen": e.chosen, "rejected": e.rejected}
            for e in examples
        ])

        orpo_config = ORPOConfig(
            output_dir=self.config.output_dir,
            lambda_=self.config.lambda_orpo,
            max_length=self.config.max_length,
            per_device_train_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
        )

        trainer = ORPOTrainer(
            model=model,
            args=orpo_config,
            train_dataset=hf_data,
            tokenizer=tokenizer,
        )
        print("[ORPO] Starting training...")
        trainer.train()
        print(f"[ORPO] Done! Checkpoints saved to {self.config.output_dir}")
```

---

## ▶️ Bước 4 — Chạy Training

### Cell 6 — Chạy DPO

```python
from src.preference_lab.trainers import PreferenceTrainer, TrainingConfig

config = TrainingConfig(
    method="dpo",
    model_name="facebook/opt-125m",  # nhỏ, nhanh, phù hợp demo
    beta=0.1,
    max_length=256,
    batch_size=2,
    num_train_epochs=1,
    output_dir="/kaggle/working/outputs/dpo",
)

trainer = PreferenceTrainer(config)
trainer.train()
```

### Cell 7 — (Tùy chọn) Chạy ORPO thay thế

```python
config_orpo = TrainingConfig(
    method="orpo",
    model_name="facebook/opt-125m",
    lambda_orpo=0.1,
    max_length=256,
    batch_size=2,
    num_train_epochs=1,
    output_dir="/kaggle/working/outputs/orpo",
)

trainer_orpo = PreferenceTrainer(config_orpo)
trainer_orpo.train()
```

---

## 📊 Bước 5 — Evaluation sau training

### Cell 8 — Chạy evaluation và lưu metrics

```python
import json
from pathlib import Path
from preference_lab.data import load_jsonl
from preference_lab.evaluate import pairwise_accuracy, write_metrics

examples = load_jsonl("data/sample_preferences.jsonl")

# Mock scores — thay bằng logprob thật nếu muốn nâng cao
chosen_scores = [1.0] * len(examples)
rejected_scores = [0.0] * len(examples)

metrics = {
    "pairwise_accuracy": pairwise_accuracy(examples, chosen_scores, rejected_scores),
    "num_examples": len(examples),
    "method": "dpo",
    "model": "facebook/opt-125m",
}

output_path = write_metrics(metrics, "/kaggle/working/outputs")
print(f"Metrics saved to: {output_path}")
print(json.dumps(metrics, indent=2))
```

---

## 💾 Bước 6 — Lưu kết quả về máy tính

### Cell 9 — Download outputs

```python
# Zip toàn bộ output để download
import shutil
shutil.make_archive("/kaggle/working/lab_outputs", "zip", "/kaggle/working/outputs")
print("✅ File lab_outputs.zip sẵn sàng tải về!")
```

Sau đó vào tab **Output** bên phải Kaggle → download `lab_outputs.zip`.

---

## ⚠️ Lưu ý quan trọng

| Vấn đề | Giải pháp |
|---|---|
| `CUDA out of memory` | Giảm `max_length` xuống `128`, `batch_size` xuống `1` |
| Repo là Private | Đổi sang Public hoặc dùng Personal Access Token |
| Session hết hạn (9h) | Notebook tự lưu — chạy lại từ Cell 1 |
| Muốn model lớn hơn | Thay `facebook/opt-125m` bằng `facebook/opt-350m` |
| Lưu notebook | Click **Save Version** → **Save & Run All** |

---

## 📌 Tóm tắt thứ tự chạy

```
Cell 1  → Clone repo
Cell 2  → Cài đặt + kiểm tra GPU
Cell 3  → Chạy tests (8 passed)
Cell 5  → Viết trainers.py (%%writefile)
Cell 6  → Chạy DPO training
Cell 8  → Evaluation + lưu metrics.json
Cell 9  → Download kết quả
```

---

## 🔗 Links hữu ích

- [Kaggle — New Notebook](https://www.kaggle.com/code)
- [TRL DPOTrainer docs](https://huggingface.co/docs/trl/dpo_trainer)
- [TRL ORPOTrainer docs](https://huggingface.co/docs/trl/orpo_trainer)
- [facebook/opt-125m trên HuggingFace](https://huggingface.co/facebook/opt-125m)
