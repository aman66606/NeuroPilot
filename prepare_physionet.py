#!/usr/bin/env python
"""Build a REAL-EEG feature CSV from PhysioNet EEG Motor Movement/Imagery (EEGMMIDB).

Uses imagery runs 4, 8, 12 (T1 = imagine LEFT fist, T2 = imagine RIGHT fist), channels C3, Cz, C4, Pz,
resampled to 128 Hz, converted to microvolts, then the SAME preprocessing and 20 features as NeuroPilot.
Output is importable in the dashboard (Dataset Lab -> Import CSV) and used by evaluate_real.py.

    pip install -r backend/requirements.txt mne
    python scripts/prepare_physionet.py --subjects 1-10 --out data/physionet_lr.csv

Dataset: Schalk et al. 2004 (BCI2000), PhysioNet, Open Data Commons Attribution License 1.0.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.core.signal import CH, FEATURE_NAMES, FS, N, PAD, extract_features, preprocess  # noqa: E402

IMAGERY_RUNS = [4, 8, 12]
LABELS = {"T1": "LEFT", "T2": "RIGHT"}
WINDOWS_PER_TRIAL = 3  # 3 x 1 s windows inside the 4 s imagery period


def raw_to_rows(raw, subject, run=0):
    """Convert one MNE Raw (EEGMMIDB-style annotations T0/T1/T2) into feature rows."""
    import mne

    raw = raw.copy()
    mne.datasets.eegbci.standardize(raw)           # 'C3..' -> 'C3'
    raw.pick(CH).reorder_channels(CH)
    raw.resample(FS, verbose="ERROR")
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    inv = {v: k for k, v in event_id.items()}
    data = raw.get_data() * 1e6                    # volts -> microvolts
    rows = []
    for i, (onset, _, code) in enumerate(events):
        name = inv[code]
        if name not in LABELS:
            continue
        for k in range(WINDOWS_PER_TRIAL):
            seg = data[:, onset + k * N: onset + k * N + N + PAD]
            if seg.shape[1] < N + PAD:
                break
            rows.append((extract_features(preprocess(seg))[0], LABELS[name], subject, f"s{subject}_r{run}_t{i}"))
    return rows


def parse_subjects(s):
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1-10")
    ap.add_argument("--out", default="data/physionet_lr.csv")
    ap.add_argument("--data-dir", default=str(Path.home() / "mne_data"))
    a = ap.parse_args()
    import mne
    from mne.datasets import eegbci

    rows = []
    for s in parse_subjects(a.subjects):
        try:
            files = eegbci.load_data(s, IMAGERY_RUNS, path=a.data_dir, update_path=False)
            for run, f in zip(IMAGERY_RUNS, files):
                rows += raw_to_rows(mne.io.read_raw_edf(f, preload=True, verbose="ERROR"), s, run)
            print(f"subject {s}: ok ({len(rows)} windows so far)")
        except Exception as e:  # keep going; report at the end
            print(f"subject {s}: FAILED ({e})")
    if not rows:
        sys.exit("No data produced. Check your internet connection / MNE version.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(FEATURE_NAMES + ["label", "source", "subject", "trial"])
        for feats, lab, sub, trial in rows:
            w.writerow([f"{v:.5f}" for v in feats] + [lab, "PhysioNet EEGMMIDB (real EEG)", sub, trial])
    print(f"wrote {len(rows)} windows to {a.out}")


if __name__ == "__main__":
    main()
