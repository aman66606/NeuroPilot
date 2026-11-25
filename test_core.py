import numpy as np
import pytest
from app.ai.decoder import Decoder, itr
from app.core.signal import CLASSES, FS, N, PAD, extract_features, generate_window, has_artifact, preprocess, signal_quality


def test_generation_shape_and_reproducible():
    a = generate_window(0, 5, np.random.default_rng(1))
    b = generate_window(0, 5, np.random.default_rng(1))
    assert a.shape == (4, N + PAD) and np.allclose(a, b)


def test_classes_have_distinct_signatures():
    f = [extract_features(preprocess(generate_window(c, 2, np.random.default_rng(0))))[0] for c in range(5)]
    assert all(not np.allclose(f[i], f[j], atol=0.05) for i in range(5) for j in range(i + 1, 5))


def test_preprocess_removes_line_noise_and_attenuates_drift():
    t = np.arange(N + PAD) / FS
    line = preprocess(np.tile(10 * np.sin(2 * np.pi * 50 * t), (4, 1)))
    drift = preprocess(np.tile(20 * np.sin(2 * np.pi * 0.2 * t), (4, 1)))
    alpha = preprocess(np.tile(10 * np.sin(2 * np.pi * 10 * t), (4, 1)))
    rms = lambda x: float(np.sqrt((x ** 2).mean()))
    ref = 10 / np.sqrt(2)
    assert line.shape == (4, N)
    assert rms(line) < 0.2 * ref                    # 50 Hz notch: >14 dB attenuation
    assert rms(drift) < 0.1 * 20 / np.sqrt(2)       # 0.2 Hz drift removed
    assert 0.9 * ref < rms(alpha) < 1.1 * ref       # 10 Hz alpha preserved


def test_features_shape_and_bands():
    t = np.arange(N) / FS
    x = np.tile(10 * np.sin(2 * np.pi * 10 * t), (4, 1))
    feats, bp, psd = extract_features(x)
    assert feats.shape == (20,) and bp["alpha"] > 10 * bp["beta"] and psd.shape == (N // 2 + 1,)


def test_artifact_and_quality():
    assert has_artifact(np.full((4, N), 60.0)) and not has_artifact(np.zeros((4, N)))
    assert signal_quality(2, False) > signal_quality(10, True) >= 0


@pytest.fixture(scope="module")
def dec():
    d = Decoder(seed=42)
    d.train()
    return d


def test_training_metrics_heldout(dec):
    m = dec.metrics
    assert m["accuracy"] > 0.85 and m["n"] == len(dec.yte) and np.array(m["confusion_matrix"]).sum() == m["n"]
    assert m["train_size"] + m["n"] == 600 and m["synthetic"] is True


def test_training_is_reproducible():
    a, b = Decoder(seed=3), Decoder(seed=3)
    assert a.train(40)["accuracy"] == b.train(40)["accuracy"]


def test_classification_not_random(dec):
    rng = np.random.default_rng(5)
    hits = 0
    for i in range(50):
        c = i % 5
        f = extract_features(preprocess(generate_window(c, 4, rng)))[0]
        hits += dec.predict(f)[0] == CLASSES[c]
    assert hits >= 42


def test_confidence_threshold_rejects(dec):
    f = extract_features(preprocess(generate_window(0, 4, np.random.default_rng(2))))[0]
    dec.threshold = 0.3
    assert dec.predict(f)[2] is True
    dec.threshold = 0.999999
    assert dec.predict(f)[2] is False
    dec.threshold = 0.6


def test_retrain_needs_verified_then_improves():
    d = Decoder(seed=42)
    d.train(60)
    with pytest.raises(ValueError):
        d.retrain()
    rng = np.random.default_rng(11)
    for i in range(80):
        c = i % 5
        d.add_verified(extract_features(preprocess(generate_window(c, 12, rng)))[0], CLASSES[c])
    r = d.retrain(12.0)
    assert r["after"]["accuracy"] >= r["before"]["accuracy"] and r["verified_used"] == 80


def test_itr():
    assert itr(0.2) == 0 and itr(1.0) == pytest.approx(np.log2(5) * 60) and itr(0.9) > itr(0.7)
