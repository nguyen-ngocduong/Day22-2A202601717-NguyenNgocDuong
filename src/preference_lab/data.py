from __future__ import annotations
import json
import random
from pathlib import Path
from .schemas import PreferenceExample


def load_jsonl(path: str | Path) -> list[PreferenceExample]:
    """Load preference examples from JSONL.

    Improvements over skeleton:
    - Line-numbered error messages for easy debugging.
    - Duplicate prompt detection.
    """
    examples: list[PreferenceExample] = []
    seen_prompts: set[str] = set()
    errors: list[str] = []

    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {lineno}: invalid JSON — {exc}")
                continue
            try:
                example = PreferenceExample.model_validate(data)
            except Exception as exc:
                errors.append(f"Line {lineno}: schema error — {exc}")
                continue

            # Duplicate prompt check
            if example.prompt in seen_prompts:
                errors.append(f"Line {lineno}: duplicate prompt detected — '{example.prompt[:60]}'")
            seen_prompts.add(example.prompt)
            examples.append(example)

    if errors:
        for msg in errors:
            print(f"[WARN] {msg}")

    return examples


def split_by_prompt(
    examples: list[PreferenceExample],
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Split examples by prompt to avoid leakage.

    Groups all rows sharing the same prompt, shuffles prompt-groups
    deterministically, then assigns groups to train/val.
    """
    # Group by prompt
    from collections import defaultdict
    groups: dict[str, list[PreferenceExample]] = defaultdict(list)
    for ex in examples:
        groups[ex.prompt].append(ex)

    prompt_keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(prompt_keys)

    cut = max(1, int(len(prompt_keys) * (1 - validation_ratio)))
    train_keys = set(prompt_keys[:cut])

    train: list[PreferenceExample] = []
    val: list[PreferenceExample] = []
    for key in prompt_keys:
        if key in train_keys:
            train.extend(groups[key])
        else:
            val.extend(groups[key])

    return train, val
