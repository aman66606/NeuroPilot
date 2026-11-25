import csv
import io

import pytest
from fastapi.testclient import TestClient
from app.main import create_app


def mk(client, name="s1"):
    return client.post("/api/sessions", json={"name": name}).json()["data"]["id"]


def test_health(client):
    d = client.get("/api/health").json()
    assert d["status"] == "ok" and d["data"]["decoder_ready"] and "educational" in d["data"]["disclaimer"]


def test_session_crud_and_validation(client):
    sid = mk(client)
    assert client.get(f"/api/sessions/{sid}").json()["data"]["name"] == "s1"
    assert len(client.get("/api/sessions").json()["data"]) == 1
    assert client.post("/api/sessions", json={"name": ""}).status_code == 422
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.delete(f"/api/sessions/{sid}").json()["data"]["deleted"] == sid
    assert client.get(f"/api/sessions/{sid}").status_code == 404


def test_session_persistence_across_restart(db_url):
    c1 = TestClient(create_app(db_url, frontend_dir="/x"))
    sid = mk(c1)
    c1.post("/api/decoder/predict", json={"session_id": sid, "intent": "RIGHT"})
    c2 = TestClient(create_app(db_url, frontend_dir="/x"))
    d = c2.get(f"/api/sessions/{sid}").json()["data"]
    assert len(d["predictions"]) == 1 and d["predictions"][0]["predicted"] in ("LEFT", "RIGHT", "FORWARD", "BACKWARD", "SELECT")


def test_signal_lifecycle(client):
    sid = mk(client)
    assert client.post("/api/signals/start", json={"session_id": sid}).json()["data"]["running"] is True
    assert client.get(f"/api/sessions/{sid}").json()["data"]["status"] == "recording"
    assert client.get("/api/signals/status").json()["data"]["running"] is True
    client.post("/api/signals/stop")
    d = client.get(f"/api/sessions/{sid}").json()["data"]
    assert d["status"] == "completed" and d["ended_at"]
    assert client.post("/api/signals/start", json={"noise": 99}).status_code == 422


def test_predict_real_latency_and_fields(client):
    r = client.post("/api/decoder/predict", json={"intent": "FORWARD", "noise": 4, "include_signal": True}).json()["data"]
    lat = r["latency_ms"]
    assert r["simulated"] is True and 0 < lat["processing"] <= lat["total"] and lat["inference"] > 0
    assert r["predicted"] == "FORWARD" and len(r["signal"]["raw"]) == 4 and abs(sum(r["probabilities"].values()) - 1) < 1e-3
    assert client.post("/api/decoder/predict", json={"intent": "JUMP"}).status_code == 422
    assert client.post("/api/decoder/predict", json={"session_id": "zzz"}).status_code == 404


def test_threshold_rejection_via_api(client):
    client.post("/api/decoder/threshold", json={"threshold": 0.99})
    rs = [client.post("/api/decoder/predict", json={"intent": "LEFT", "noise": 14}).json()["data"] for _ in range(10)]
    assert any(not r["accepted"] and r["command"] is None for r in rs)
    assert client.post("/api/decoder/threshold", json={"threshold": 5}).status_code == 422


def test_feedback_and_adaptive_retrain(client):
    sid = mk(client)
    assert client.post("/api/decoder/retrain", json={}).status_code == 409
    ids = []
    for i in range(40):
        p = client.post("/api/decoder/predict", json={"session_id": sid, "intent": ["LEFT", "RIGHT", "FORWARD", "BACKWARD", "SELECT"][i % 5], "noise": 12}).json()["data"]
        ids.append((p["prediction_id"], p["intent"]))
    for pid, lab in ids:
        f = client.post("/api/feedback", json={"prediction_id": pid, "label": lab}).json()["data"]
        assert f["used_for_training"]
    assert client.get("/api/decoder/metrics").json()["data"]["verified_examples"] == 40
    r = client.post("/api/decoder/retrain", json={"field_noise": 12}).json()["data"]
    assert r["after"]["accuracy"] >= r["before"]["accuracy"]
    assert client.post("/api/feedback", json={"prediction_id": 99999, "label": "LEFT"}).status_code == 404
    assert client.post("/api/feedback", json={"prediction_id": ids[0][0], "label": "X"}).status_code == 422
    assert len(client.get(f"/api/sessions/{sid}").json()["data"]["feedback"]) == 40


def test_unverified_predictions_never_trained(client):
    sid = mk(client)
    client.post("/api/decoder/predict", json={"session_id": sid})
    assert client.get("/api/decoder/metrics").json()["data"]["verified_examples"] == 0


def test_metrics_train_analytics(client):
    m = client.get("/api/decoder/metrics").json()["data"]
    assert m["accuracy"] > 0.85 and len(m["confusion_matrix"]) == 5 and m["synthetic"]
    t = client.post("/api/decoder/train", json={"n_per_class": 30, "seed": 1}).json()["data"]
    assert t["train_size"] == 120
    sid = mk(client)
    for _ in range(3):
        client.post("/api/decoder/predict", json={"session_id": sid})
    a = client.get("/api/analytics").json()["data"]
    assert a["predictions"] == 3 and sum(a["distribution"].values()) == 3 and a["sessions"] == 1


def test_exports(client):
    sid = mk(client)
    for _ in range(3):
        client.post("/api/decoder/predict", json={"session_id": sid})
    j = client.get(f"/api/export/{sid}").json()["data"]
    assert len(j["predictions"]) == 3 and j["simulated"]
    rows = list(csv.DictReader(io.StringIO(client.get(f"/api/export/{sid}?kind=predictions").text)))
    assert len(rows) == 3 and "total_ms" in rows[0]
    sig = list(csv.reader(io.StringIO(client.get(f"/api/export/{sid}?kind=signal").text)))
    assert len(sig) == 129 and sig[0][1] == "C3_raw"
    assert client.get(f"/api/export/{sid}?kind=bad").status_code == 400
    assert client.get("/api/export/nope").status_code == 404


def test_pdf_report(client):
    sid = mk(client)
    client.post("/api/decoder/predict", json={"session_id": sid})
    r = client.get(f"/api/report/{sid}")
    assert r.headers["content-type"] == "application/pdf" and r.content.startswith(b"%PDF") and len(r.content) > 2000


def test_websocket_stream(client):
    sid = mk(client)
    with client.websocket_connect(f"/ws/neural-stream?session_id={sid}&count=3&interval=0&intent=SELECT") as ws:
        msgs = [ws.receive_json() for _ in range(4)]
    assert [m["type"] for m in msgs] == ["window"] * 3 + ["done"]
    w = msgs[0]
    assert w["simulated"] and w["predicted"] and len(w["signal"]["filtered"]) == 4 and w["latency_ms"]["total"] > 0
    assert len(client.get(f"/api/sessions/{sid}").json()["data"]["predictions"]) == 3


def test_websocket_rejects_bad_session(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/neural-stream?session_id=nope"):
            pass


def test_brainflow_synthetic_board(client):
    pytest.importorskip("brainflow")
    r = client.post("/api/signals/start", json={"source": "brainflow_synthetic"})
    try:
        assert r.status_code == 200
        p = client.post("/api/decoder/predict", json={}).json()["data"]
        assert p["simulated"] is False and p["predicted"]
    finally:
        client.post("/api/signals/stop")
