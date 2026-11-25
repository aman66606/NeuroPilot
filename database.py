"""SQLite persistence with SQLAlchemy 2.0."""
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship


def now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class SessionRow(Base):
    __tablename__ = "sessions"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(20), default="created")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    predictions: Mapped[list["PredictionRow"]] = relationship(cascade="all, delete-orphan", back_populates="session")
    feedback: Mapped[list["FeedbackRow"]] = relationship(cascade="all, delete-orphan", back_populates="session")

    def to_dict(self):
        iso = lambda d: d.isoformat() if d else None
        return {"id": self.id, "name": self.name, "status": self.status, "created_at": iso(self.created_at),
                "started_at": iso(self.started_at), "ended_at": iso(self.ended_at)}


class PredictionRow(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"))
    ts: Mapped[datetime] = mapped_column(DateTime, default=now)
    predicted: Mapped[str] = mapped_column(String(16))
    confidence: Mapped[float] = mapped_column(Float)
    intent: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    accepted: Mapped[bool] = mapped_column(Boolean)
    processing_ms: Mapped[float] = mapped_column(Float)
    inference_ms: Mapped[float] = mapped_column(Float)
    total_ms: Mapped[float] = mapped_column(Float)
    quality: Mapped[int] = mapped_column(Integer)
    feedback_label: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    session: Mapped[SessionRow] = relationship(back_populates="predictions")

    def to_dict(self):
        return {"id": self.id, "ts": self.ts.isoformat(), "predicted": self.predicted, "confidence": round(self.confidence, 4),
                "intent": self.intent, "accepted": self.accepted, "processing_ms": round(self.processing_ms, 3),
                "inference_ms": round(self.inference_ms, 3), "total_ms": round(self.total_ms, 3), "quality": self.quality,
                "feedback_label": self.feedback_label, "correct": self.correct}


class FeedbackRow(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"))
    prediction_id: Mapped[int] = mapped_column(Integer)
    ts: Mapped[datetime] = mapped_column(DateTime, default=now)
    predicted: Mapped[str] = mapped_column(String(16))
    label: Mapped[str] = mapped_column(String(16))
    correct: Mapped[bool] = mapped_column(Boolean)
    session: Mapped[SessionRow] = relationship(back_populates="feedback")

    def to_dict(self):
        return {"id": self.id, "prediction_id": self.prediction_id, "ts": self.ts.isoformat(),
                "predicted": self.predicted, "label": self.label, "correct": self.correct}


def make_engine(url):
    engine = create_engine(url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine


@contextmanager
def session_scope(engine):
    s = Session(engine, expire_on_commit=False)
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
