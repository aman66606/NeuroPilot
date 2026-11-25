#!/usr/bin/env python
"""Honest evaluation of the NeuroPilot decoder on a real-EEG feature CSV.

    python scripts/evaluate_real.py data/physionet_lr.csv --out docs/real_data_results.md

Reports (a) within-subject cross-validation grouped by TRIAL (no leakage between neighbouring windows)
and (b) leave-one-subject-out. Chance level is printed so results are not over-read.
"""
import argparse
import csv
import json
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load(path):
    X, y, sub, trial, src = [], [], [], [], set()
    with open(path) as fh:
        r = csv.reader(fh)
        header = next(r)
        for row in r:
            X.append([float(v) for v in row[:20]]); y.append(row[20]); src.add(row[21])
            sub.append(row[22]); trial.append(row[23])
    return np.array(X), np.array(y), np.array(sub), np.array(trial), src


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))


def evaluate(path):
    X, y, sub, trial, src = load(path)
    counts = Counter(y)
    chance = max(counts.values()) / len(y)
    within = []
    for s in sorted(set(sub), key=lambda v: int(v)):
        m = sub == s
        k = min(5, len(set(trial[m])))
        if k < 3:
            continue
        pred = cross_val_predict(model(), X[m], y[m], groups=trial[m], cv=GroupKFold(k))
        within.append((s, float((pred == y[m]).mean())))
    loso = cross_val_predict(model(), X, y, groups=sub, cv=LeaveOneGroupOut()) if len(set(sub)) > 1 else None
    res = {"source": sorted(src), "windows": int(len(y)), "subjects": int(len(set(sub))), "classes": {str(k): int(v) for k, v in counts.items()},
           "chance": chance, "within_subject": {"per_subject": within, "mean": float(np.mean([a for _, a in within])) if within else None,
                                                  "sd": float(np.std([a for _, a in within])) if within else None},
           "leave_one_subject_out": float((loso == y).mean()) if loso is not None else None}
    return res


def markdown(r):
    w = r["within_subject"]
    return f"""# Real-EEG evaluation

**Data source:** {', '.join(r['source'])}
**Windows:** {r['windows']} (1 s each) from {r['subjects']} subjects; classes {r['classes']}
**Chance level (majority class):** {r['chance'] * 100:.1f}%

| Protocol | Accuracy |
|---|---|
| Within-subject, 5-fold grouped by trial (mean over subjects ± sd) | {w['mean'] * 100:.1f}% ± {w['sd'] * 100:.1f}% |
| Leave-one-subject-out (zero calibration) | {r['leave_one_subject_out'] * 100:.1f}% |

Notes: single 1-second windows, 4 channels (C3, Cz, C4, Pz), band-power features, linear classifier. Real motor-imagery decoding from
so little data is hard; values near chance are expected for some subjects. Per-subject accuracy: {', '.join(f"S{s}: {a * 100:.0f}%" for s, a in w['per_subject'])}.
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="docs/real_data_results.md")
    a = ap.parse_args()
    r = evaluate(a.csv)
    md = markdown(r)
    open(a.out, "w").write(md)
    print(md)
    print(json.dumps(r, indent=1))
