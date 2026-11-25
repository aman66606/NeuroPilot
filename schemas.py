from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .core.signal import CLASSES


def _check_label(v):
    if v is not None and v not in CLASSES:
        raise ValueError(f"must be one of {CLASSES}")
    return v


class SessionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class StartRequest(BaseModel):
    session_id: Optional[str] = None
    noise: float = Field(6.0, ge=0.5, le=30)
    source: Literal["simulation", "brainflow_synthetic", "brainflow_hardware"] = "simulation"
    serial_port: Optional[str] = None


class PredictRequest(BaseModel):
    intent: Optional[str] = None
    noise: float = Field(6.0, ge=0.5, le=30)
    session_id: Optional[str] = None
    include_signal: bool = False
    _v = field_validator("intent")(_check_label)


class FeedbackRequest(BaseModel):
    prediction_id: int
    label: str
    _v = field_validator("label")(_check_label)


class ThresholdRequest(BaseModel):
    threshold: float = Field(ge=0.3, le=0.99)


class TrainRequest(BaseModel):
    seed: int = 42
    n_per_class: int = Field(120, ge=20, le=400)


class RetrainRequest(BaseModel):
    field_noise: float = Field(12.0, ge=1, le=30)
