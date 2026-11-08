"""Pydantic-схемы: та же схема входа, что в src.predictor (list[str])."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    spans: list[str] = Field(..., min_length=1, max_length=1000)

    @field_validator("spans")
    @classmethod
    def _no_empty(cls, v: list[str]) -> list[str]:
        if any(not s.strip() for s in v):
            raise ValueError("пустой спан не допускается")
        return v


class Prediction(BaseModel):
    span: str
    label: str
    confidence: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
