"""Plumbing test for scripts/evaluate_bonn.py with a SYNTHETIC stand-in for the Bonn folder layout (not real EEG)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import evaluate_bonn as eb  # noqa: E402


def make_fake_bonn(root, per_set=10):
    rng = np.random.default_rng(0)
    t = np.arange(4097) / eb.BONN_FS
    amp = {"Z": 20, "O": 30, "N": 40, "F": 50, "S": 300}
    for s in eb.SETS:
        d = root / s
        d.mkdir()
        for i in range(per_set):
            x = amp[s] * np.sin(2 * np.pi * (8 + 3 * eb.SETS.index(s)) * t + rng.uniform(0, 6)) + 10 * rng.standard_normal(4097)
            np.savetxt(d / (f"{s}{i:03d}." + ("TXT" if s == "N" else "txt")), np.round(x), fmt="%d")


def test_pipeline_on_fake_bonn_layout(tmp_path):
    make_fake_bonn(tmp_path)
    segs = eb.load_segments(tmp_path)
    assert len(segs) == 50 and len(eb.windows_features(segs[0][2])) >= 20
    r = eb.run(tmp_path)
    m = next(iter(r["models"].values()))
    assert r["segments"] == 50 and np.array(m["cm"]).sum() == r["windows"]
    assert m["seizure_sensitivity"] > 0.9 and m["seizure_specificity"] > 0.9   # S is 6x larger by construction
    md = eb.markdown(r)
    assert "intracranial" in md and "not a medical device" in md
