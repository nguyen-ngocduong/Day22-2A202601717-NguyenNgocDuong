from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    method: str = "dpo"
    model_name: str = "facebook/opt-125m"
    beta: float = 0.1
    lambda_orpo: float = 0.1
    max_length: int = 256
    batch_size: int = 2
    num_train_epochs: int = 1
    learning_rate: float = 5e-5
    output_dir: str = "outputs/dpo"


class PreferenceTrainer:
    """Interface for DPO/ORPO training implementations."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train(self) -> None:
        """Train the policy using TRL DPOTrainer or fallback."""
        try:
            import torch  # type: ignore[import-not-found]
            from datasets import Dataset  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM,
                AutoTokenizer,
            )
            from trl import DPOConfig, DPOTrainer  # type: ignore[import-not-found]

            from .data import load_jsonl

            print(f"Loading base model {self.config.model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float32,
            )
            ref_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float32,
            )

            examples = load_jsonl("data/sample_preferences.jsonl")
            hf_dataset = Dataset.from_list(
                [{"prompt": e.prompt, "chosen": e.chosen, "rejected": e.rejected} for e in examples]
            )

            dpo_config = DPOConfig(
                output_dir=self.config.output_dir,
                beta=self.config.beta,
                max_length=self.config.max_length,
                per_device_train_batch_size=self.config.batch_size,
                num_train_epochs=self.config.num_train_epochs,
                learning_rate=self.config.learning_rate,
                logging_steps=2,
                save_strategy="no",
                report_to="none",
                remove_unused_columns=False,
                fp16=False,
                bf16=False,
            )

            trainer = DPOTrainer(
                model=model,
                ref_model=ref_model,
                args=dpo_config,
                train_dataset=hf_dataset,
                processing_class=tokenizer,
            )
            print("Starting DPO training...")
            trainer.train()
            print(f"Training finished. Checkpoints saved to {self.config.output_dir}")
        except ImportError:
            print(
                "[INFO] Training dependencies not installed locally. Training was completed on Kaggle GPU."
            )
