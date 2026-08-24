from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class PreferenceExample(BaseModel):
    """One preference pair for DPO/ORPO-style alignment."""

    prompt: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt", "chosen", "rejected")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("rejected")
    @classmethod
    def chosen_and_rejected_must_differ(cls, rejected: str, info: Any) -> str:
        chosen = info.data.get("chosen")
        if chosen is not None and chosen.strip().lower() == rejected.strip().lower():
            raise ValueError("chosen and rejected must differ")
        return rejected
