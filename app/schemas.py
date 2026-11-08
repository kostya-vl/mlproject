"""
Pydantic-схемы API.

POST /predict — формат JSON:
  {
    "spans": ["<строка>", ...]      # обязательное поле
  }
Ограничения (валидируются и в Pydantic, и в src.predictor):
  * spans — непустой список строк, 1..1000 элементов;
  * каждая строка — непустая, ≤ 1000 символов, без NUL/управляющих символов;
  * суммарно ≤ 1 000 000 символов (MAX_TOTAL_CHARS в src.predictor);
  * тела запроса больше 1 МБ FastAPI/pydantic отвергнет по факту размера.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    spans: list[str] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Список текстовых спанов; 1..1000 элементов, каждый ≤1000 символов.",
    )

    @field_validator("spans")
    @classmethod
    def _no_empty(cls, v: list[str]) -> list[str]:
        if any(not s.strip() for s in v):
            raise ValueError("пустой спан не допускается")
        return v


class Prediction(BaseModel):
    span: str
    label: str
    # confidence = max(predict_proba) модели LogisticRegression.
    # Это осмысленная оценка вероятности класса (сумма по классам = 1),
    # а не произвольный score.
    confidence: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str
    request_id: str


class HealthResponse(BaseModel):
    status: str
    model_version: str | None = None
