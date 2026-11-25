"""NeuroPilot AI backend. Run: uvicorn app.main:create_app --factory"""
import asyncio
import csv
import io
import logging
import os
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .acquisition import BrainFlowSource
from .ai.decoder import Decoder, itr
from .core.signal import CH, CLASSES, FS, N, extract_features, generate_window, has_artifact, preprocess, signal_quality
from .database import FeedbackRow, PredictionRow, SessionRow, make_engine, session_scope
from .report import DISCLAIMER, build_pdf
from .schemas import (FeedbackRequest, PredictRequest, RetrainRequest, SessionCreate, StartRequest, ThresholdRequest, TrainRequest)

log = logging.getLogger("neuropilot")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def ok(data):
    return {"status": "ok", "data": data}


def create_app(db_url: Optional[str] = None, frontend_dir: Optional[str] = None) -> FastAPI:
    db_url = db_url or os.getenv("NEUROPILOT_DB", "sqlite:///./neuropilot.db")
    app = FastAPI(title="NeuroPilot AI", version="1.0", description=DISCLAIMER)
    st = app.state
    st.engine = make_engine(db_url)
    st.decoder = Decoder(seed=int(os.getenv("NEUROPILOT_SEED", "42")))
    st.decoder.train()
    st.rng = np.random.default_rng(7)
    st.source, st.pending, st.last_window = None, {}, {}
    st.stream = {"running": False, "session_id": None, "source": "simulation", "noise": 6.0, "windows": 0}
    st.stats = {"windows": 0, "commands": 0, "feedback": 0}
    st.last_bp = {}

    @app.exception_handler(HTTPException)
    async def _http(_, e):
        return JSONResponse({"status": "error", "error": e.detail}, status_code=e.status_code)

    @app.exception_handler(RequestValidationError)
    async def _val(_, e):
        return JSONResponse({"status": "error", "error": "validation failed", "details": jsonable(e.errors())}, status_code=422)

    def jsonable(errs):
        return [{"loc": list(x["loc"]), "msg": x["msg"]} for x in errs]

    def get_session(db, sid):
        row = db.get(SessionRow, sid)
        if not row:
            raise HTTPException(404, f"session {sid} not found")
        return row

    def run_window(intent=None, noise=6.0, session_id=None, include_signal=False):
        ci = CLASSES.index(intent) if intent else int(st.rng.integers(len(CLASSES)))
        t0 = time.perf_counter()
        raw = st.source.get_window() if st.source else generate_window(ci, noise, st.rng, 0.05)
        t1 = time.perf_counter()
        filt = preprocess(raw)
        feats, bp, psd = extract_features(filt)
        t2 = time.perf_counter()
        label, conf, accepted, proba = st.decoder.predict(feats)
        t3 = time.perf_counter()
        art = has_artifact(filt)
        res = {"ts": datetime.now(timezone.utc).isoformat(), "predicted": label, "confidence": round(conf, 4), "accepted": accepted,
               "command": label if accepted else None, "intent": None if st.source else CLASSES[ci], "probabilities": dict(zip(CLASSES, np.round(proba, 4).tolist())),
               "band_power": {k: round(float(v), 4) for k, v in bp.items()}, "artifact": art, "quality": signal_quality(noise, art),
               "latency_ms": {"acquisition": round((t1 - t0) * 1e3, 3), "processing": round((t2 - t1) * 1e3, 3),
                              "inference": round((t3 - t2) * 1e3, 3), "total": round((t3 - t0) * 1e3, 3)},
               "simulated": st.source is None, "prediction_id": None}
        if session_id:
            with session_scope(st.engine) as db:
                get_session(db, session_id)
                row = PredictionRow(session_id=session_id, predicted=label, confidence=conf, intent=res["intent"], accepted=accepted,
                                    processing_ms=res["latency_ms"]["processing"], inference_ms=res["latency_ms"]["inference"],
                                    total_ms=res["latency_ms"]["total"], quality=res["quality"])
                db.add(row)
                db.flush()
                res["prediction_id"] = row.id
            st.pending[row.id] = feats
            if len(st.pending) > 3000:
                st.pending.pop(next(iter(st.pending)))
            st.last_window[session_id] = (raw[:, -N:], filt)
        st.stats["windows"] += 1
        st.stats["commands"] += int(accepted)
        st.last_bp = res["band_power"]
        st.stream["windows"] += 1
        if include_signal:
            res["signal"] = {"channels": CH, "fs": FS, "raw": np.round(raw[:, -N:], 2).tolist(), "filtered": np.round(filt, 2).tolist()}
        return res

    # ---------------- REST ----------------
    @app.get("/api/health")
    def health():
        return ok({"service": "neuropilot-ai", "version": app.version, "decoder_ready": st.decoder.pipe is not None,
                   "simulated_default": True, "disclaimer": DISCLAIMER})

    @app.get("/api/sessions")
    def list_sessions():
        with session_scope(st.engine) as db:
            return ok([r.to_dict() | {"windows": len(r.predictions)} for r in db.query(SessionRow).order_by(SessionRow.created_at.desc())])

    @app.post("/api/sessions", status_code=201)
    def create_session(body: SessionCreate):
        with session_scope(st.engine) as db:
            row = SessionRow(id=uuid.uuid4().hex[:12], name=body.name)
            db.add(row)
            db.flush()
            log.info("session created %s", row.id)
            return ok(row.to_dict())

    @app.get("/api/sessions/{sid}")
    def read_session(sid: str):
        with session_scope(st.engine) as db:
            r = get_session(db, sid)
            return ok(r.to_dict() | {"predictions": [p.to_dict() for p in r.predictions], "feedback": [f.to_dict() for f in r.feedback]})

    @app.delete("/api/sessions/{sid}")
    def delete_session(sid: str):
        with session_scope(st.engine) as db:
            db.delete(get_session(db, sid))
        return ok({"deleted": sid})

    @app.post("/api/signals/start")
    def start(body: StartRequest):
        if st.source:
            st.source.close()
            st.source = None
        if body.source != "simulation":
            try:
                st.source = BrainFlowSource(-1 if body.source == "brainflow_synthetic" else 0, body.serial_port)
            except Exception as e:
                raise HTTPException(400, f"BrainFlow source unavailable: {e}")
        if body.session_id:
            with session_scope(st.engine) as db:
                r = get_session(db, body.session_id)
                r.status, r.started_at = "recording", datetime.now(timezone.utc)
        st.stream.update(running=True, session_id=body.session_id, source=body.source, noise=body.noise, windows=0)
        return ok(st.stream)

    @app.post("/api/signals/stop")
    def stop():
        sid = st.stream["session_id"]
        if sid:
            with session_scope(st.engine) as db:
                r = db.get(SessionRow, sid)
                if r:
                    r.status, r.ended_at = "completed", datetime.now(timezone.utc)
        if st.source:
            st.source.close()
            st.source = None
        st.stream.update(running=False)
        return ok(st.stream)

    @app.get("/api/signals/status")
    def status():
        return ok(st.stream | {"stats": st.stats})

    @app.post("/api/decoder/train")
    def train(body: TrainRequest):
        return ok(st.decoder.train(body.n_per_class, body.seed))

    @app.post("/api/decoder/predict")
    def predict(body: PredictRequest):
        return ok(run_window(body.intent, body.noise, body.session_id, body.include_signal))

    @app.get("/api/decoder/metrics")
    def metrics():
        d = st.decoder
        return ok(d.metrics | {"threshold": d.threshold, "verified_examples": len(d.verified), "last_retrain": d.last_retrain})

    @app.post("/api/decoder/threshold")
    def threshold(body: ThresholdRequest):
        st.decoder.threshold = body.threshold
        return ok({"threshold": body.threshold})

    @app.post("/api/decoder/retrain")
    def retrain(body: RetrainRequest):
        try:
            return ok(st.decoder.retrain(body.field_noise))
        except ValueError as e:
            raise HTTPException(409, f"{e}: submit feedback first")

    @app.post("/api/feedback")
    def feedback(body: FeedbackRequest):
        with session_scope(st.engine) as db:
            p = db.get(PredictionRow, body.prediction_id)
            if not p:
                raise HTTPException(404, "prediction not found")
            correct = p.predicted == body.label
            p.feedback_label, p.correct = body.label, correct
            db.add(FeedbackRow(session_id=p.session_id, prediction_id=p.id, predicted=p.predicted, label=body.label, correct=correct))
        feats = st.pending.get(body.prediction_id)
        if feats is not None:
            st.decoder.add_verified(feats, body.label)
        st.stats["feedback"] += 1
        return ok({"correct": correct, "verified_examples": len(st.decoder.verified), "used_for_training": feats is not None})

    @app.get("/api/analytics")
    def analytics():
        with session_scope(st.engine) as db:
            preds = db.query(PredictionRow).all()
            n_sess = db.query(SessionRow).count()
        n = len(preds)
        fb = [p for p in preds if p.correct is not None]
        return ok({"sessions": n_sess, "predictions": n, "distribution": dict(Counter(p.predicted for p in preds)),
                   "rejected": sum(not p.accepted for p in preds),
                   "mean_total_latency_ms": round(sum(p.total_ms for p in preds) / n, 3) if n else None,
                   "feedback_accuracy": round(sum(p.correct for p in fb) / len(fb), 4) if fb else None,
                   "last_band_power": st.last_bp, "model": {k: st.decoder.metrics[k] for k in ("accuracy", "macro_f1", "itr_bits_min")}})

    def load_all(sid):
        with session_scope(st.engine) as db:
            r = get_session(db, sid)
            return r.to_dict(), [p.to_dict() for p in r.predictions], [f.to_dict() for f in r.feedback]

    @app.get("/api/export/{sid}")
    def export(sid: str, kind: str = "json"):
        sess, preds, fb = load_all(sid)
        if kind == "json":
            return ok({"session": sess, "predictions": preds, "feedback": fb, "decoder": st.decoder.metrics, "simulated": True})
        if kind == "predictions":
            out = io.StringIO()
            w = csv.DictWriter(out, fieldnames=list(preds[0]) if preds else ["id"])
            w.writeheader()
            w.writerows(preds)
            return Response(out.getvalue(), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={sid}_predictions.csv"})
        if kind == "signal":
            if sid not in st.last_window:
                raise HTTPException(404, "no signal window recorded for this session in this server run")
            raw, filt = st.last_window[sid]
            out = io.StringIO()
            w = csv.writer(out)
            w.writerow(["sample"] + [f"{c}_{k}" for c in CH for k in ("raw", "filtered")])
            for i in range(N):
                w.writerow([i] + [round(float(x), 3) for c in range(len(CH)) for x in (raw[c, i], filt[c, i])])
            return Response(out.getvalue(), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={sid}_signal.csv"})
        raise HTTPException(400, "kind must be json, predictions or signal")

    @app.get("/api/report/{sid}")
    def report(sid: str):
        sess, preds, fb = load_all(sid)
        pdf = build_pdf(sess, preds, fb, st.decoder.metrics | {"itr_bits_min": st.decoder.metrics["itr_bits_min"]}, st.decoder.last_retrain)
        return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": f"inline; filename={sid}_report.pdf"})

    # ---------------- WebSocket ----------------
    @app.websocket("/ws/neural-stream")
    async def neural_stream(ws: WebSocket, session_id: Optional[str] = None, noise: float = 6.0, count: int = 0,
                            interval: float = 0.8, intent: Optional[str] = None):
        if intent is not None and intent not in CLASSES:
            await ws.close(code=4400)
            return
        if session_id:
            with session_scope(st.engine) as db:
                if not db.get(SessionRow, session_id):
                    await ws.close(code=4404)
                    return
        await ws.accept()
        n = 0
        try:
            while n < (count or 10000):
                await ws.send_json({"type": "window", **run_window(intent, noise, session_id, include_signal=True)})
                n += 1
                await asyncio.sleep(max(0.0, min(interval, 5.0)))
            await ws.send_json({"type": "done", "windows": n})
            await ws.close()
        except WebSocketDisconnect:
            log.info("stream client disconnected after %d windows", n)

    fe = Path(frontend_dir or os.getenv("FRONTEND_DIR", Path(__file__).resolve().parents[2] / "frontend"))
    if fe.is_dir():
        app.mount("/", StaticFiles(directory=fe, html=True), name="frontend")
    return app
