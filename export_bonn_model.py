#!/usr/bin/env python
"""Train a small browser-compatible model on the Bonn EEG dataset and export weights + a few example recordings.

Features are computed directly at the native 173.61 Hz (no resampling or filtering; the authors already band-limited to 0.53-40 Hz):
1 s windows of 174 samples, mean removed, DFT bins 1-3 delta, 4-7 theta, 8-12 alpha, 13-29 beta (log10(1+power)) + log variance.
The same maths is implemented in JavaScript in frontend/index.html (Real EEG page), so browser and Python agree.

    python scripts/export_bonn_model.py data/bonn --out data/bonn_browser_model.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SETS = ["Z", "O", "N", "F", "S"]
W = 174
BANDS = [(1, 3), (4, 7), (8, 12), (13, 29)]


def feats(x):
    out = []
    for st in range(0, len(x) - W + 1, W):
        w = np.asarray(x[st:st + W], float)
        w = w - w.mean()
        p = np.abs(np.fft.rfft(w)) ** 2 / W ** 2
        out.append([np.log10(1 + p[a:b + 1].sum()) for a, b in BANDS] + [np.log10(1 + np.mean(w ** 2))])
    return np.array(out)


def load(root):
    root, segs = Path(root), []
    for s in SETS:
        d = next(p for p in root.iterdir() if p.is_dir() and p.name.upper() == s)
        for f in sorted(d.iterdir()):
            if f.suffix.lower() == ".txt":
                segs.append((s, np.loadtxt(f).astype(int)))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="data/bonn_browser_model.json")
    a = ap.parse_args()
    segs = load(a.root)
    X, y, g = [], [], []
    for gi, (s, x) in enumerate(segs):
        f = feats(x)
        X += list(f); y += [s] * len(f); g += [gi] * len(f)
    X, y, g = np.array(X), np.array(y), np.array(g)
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=42))
    cv = GroupKFold(5)
    pred = cross_val_predict(mk(), X, y, groups=g, cv=cv)
    pb = cross_val_predict(mk(), X, (y == "S").astype(int), groups=g, cv=cv)
    bt = y == "S"
    seg_ok = np.mean([max(set(pred[g == k]), key=list(pred[g == k]).count) == y[g == k][0] for k in set(g)])
    m = mk().fit(X, y)
    sc, lr = m[0], m[1]
    assert list(lr.classes_) == sorted(SETS)           # alphabetical: F N O S Z
    order = [list(lr.classes_).index(s) for s in SETS]  # reorder to Z O N F S
    rng = np.random.default_rng(7)
    samples = {s: [] for s in SETS}
    for s in SETS:
        idx = [i for i, (t, _) in enumerate(segs) if t == s]
        for i in rng.choice(idx, 2, replace=False):
            samples[s].append(segs[i][1].tolist())
    ex = feats(np.array(samples["S"][0]))[:2]
    out = {"classes": SETS, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
           "coef": lr.coef_[order].tolist(), "intercept": lr.intercept_[order].tolist(),
           "metrics": {"windows": int(len(y)), "recordings": int(len(set(g))), "window_acc_5class": float((pred == y).mean()),
                       "recording_acc_5class": float(seg_ok), "seizure_sensitivity": float((pred == "S")[bt].mean()),
                       "seizure_specificity": float((pred != "S")[~bt].mean()), "binary_acc": float(((pb == 1) == bt).mean()),
                       "confusion": confusion_matrix(y, pred, labels=SETS).tolist()},
           "samples": samples,
           "parity": {"features": ex.tolist(), "probs": m.predict_proba(ex)[:, order].tolist()}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")))
    mt = out["metrics"]
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in mt.items() if k != "confusion"})
    print("size KB", Path(a.out).stat().st_size // 1024)


if __name__ == "__main__":
    main()
