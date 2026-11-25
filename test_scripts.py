"""Plumbing tests for the real-data scripts using a MOCK of the EEGMMIDB structure (not real EEG)."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

mne = pytest.importorskip("mne")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import evaluate_real as ev  # noqa: E402
import prepare_physionet as pp  # noqa: E402
from app.core.signal import FEATURE_NAMES  # noqa: E402


def mock_raw(seed, n_trials=12):
    rng = np.random.default_rng(seed)
    sf, T = 160, 8 * n_trials + 10
    data = rng.standard_normal((6, sf * T)) * 20e-6
    descs, ons = [], []
    for i in range(n_trials):
        d, o = ("T1" if i % 2 == 0 else "T2"), 2 + i * 8.0
        data[1 if d == "T2" else 3, int(o * sf):int((o + 4) * sf)] *= 0.4
        descs.append(d); ons.append(o)
    raw = mne.io.RawArray(data, mne.create_info(["Fc5.", "C3..", "Cz..", "C4..", "Pz..", "Oz.."], sf, "eeg"), verbose="ERROR")
    raw.set_annotations(mne.Annotations(ons, [4.1] * len(ons), descs))
    return raw


def test_raw_to_rows_shapes_labels_units():
    rows = pp.raw_to_rows(mock_raw(1), 1, 4)
    assert len(rows) == 12 * pp.WINDOWS_PER_TRIAL
    assert {r[1] for r in rows} == {"LEFT", "RIGHT"} and all(len(r[0]) == 20 and np.isfinite(r[0]).all() for r in rows)
    assert len({r[3] for r in rows}) == 12                      # trial ids kept for grouped CV
    assert np.mean([r[0][2] for r in rows]) > 0.5               # microvolt scaling gives non-trivial band power


def test_evaluate_pipeline_runs(tmp_path):
    path = tmp_path / "mock.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(FEATURE_NAMES + ["label", "source", "subject", "trial"])
        for s in (1, 2, 3):
            for f, lab, sub, trial in pp.raw_to_rows(mock_raw(s), s, 4):
                w.writerow([f"{v:.5f}" for v in f] + [lab, "MOCK", sub, trial])
    r = ev.evaluate(str(path))
    assert r["subjects"] == 3 and 0 <= r["leave_one_subject_out"] <= 1 and r["within_subject"]["mean"] is not None
    assert "Chance level" in ev.markdown(r)
