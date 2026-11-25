#!/usr/bin/env python
"""Evaluate the NeuroPilot signal pipeline on the Bonn University EEG dataset (Andrzejak et al., 2001).

Sets (100 single-channel segments each, 4097 samples, 173.61 Hz, 23.6 s):
  Z = healthy, eyes open (scalp)   O = healthy, eyes closed (scalp)
  N = interictal, opposite hemisphere (intracranial)   F = interictal, epileptogenic zone (intracranial)
  S = seizure / ictal activity (intracranial)

    python scripts/evaluate_bonn.py data/bonn --out docs/bonn_results.md

The same 1-40 Hz band-pass + 50 Hz notch and band-power features as the BCI pipeline are used (1 channel -> 5 features).
Windows from one recording always stay in the same CV fold (grouped by file) so neighbouring windows cannot leak.
Please cite Andrzejak RG et al., Phys Rev E 64, 061907 (2001) and check the dataset terms of use.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.core.signal import FS, N, PAD, extract_features, preprocess  # noqa: E402

SETS = ["Z", "O", "N", "F", "S"]
BONN_FS = 173.61
DESC = {"Z": "healthy, eyes open", "O": "healthy, eyes closed", "N": "interictal (opposite hemisphere)",
        "F": "interictal (epileptogenic zone)", "S": "seizure"}


def load_segments(root):
    root, segs = Path(root), []
    for s in SETS:
        d = next((p for p in root.iterdir() if p.is_dir() and p.name.upper() == s), None)
        if d is None:
            raise SystemExit(f"folder for set {s} not found in {root}")
        for f in sorted(d.iterdir()):
            if f.suffix.lower() == ".txt":
                segs.append((s, f.name, np.loadtxt(f)))
    return segs


def windows_features(x):
    x = resample(x, int(round(len(x) * FS / BONN_FS)))
    out = []
    for st in range(0, len(x) - (N + PAD) + 1, N):
        out.append(extract_features(preprocess(x[st:st + N + PAD][None, :]))[0])
    return out


def build(segs):
    X, y, g = [], [], []
    for gi, (s, _, x) in enumerate(segs):
        for f in windows_features(x):
            X.append(f); y.append(s); g.append(gi)
    return np.array(X), np.array(y), np.array(g)


def models():
    return {"Logistic regression (NeuroPilot decoder)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)),
            "Random forest": RandomForestClassifier(200, random_state=42)}


def run(root):
    X, y, g = build(load_segments(root))
    cv = GroupKFold(5)
    res = {"windows": int(len(y)), "segments": int(len(set(g))), "models": {}}
    for name, m in models().items():
        pred = cross_val_predict(m, X, y, groups=g, cv=cv)
        seg_true = {gi: y[g == gi][0] for gi in set(g)}
        seg_pred = {gi: max(set(pred[g == gi]), key=list(pred[g == gi]).count) for gi in set(g)}
        seg_acc = float(np.mean([seg_true[k] == seg_pred[k] for k in seg_true]))
        bt, bp = y == "S", pred == "S"
        sens, spec = float((bp & bt).sum() / bt.sum()), float((~bp & ~bt).sum() / (~bt).sum())
        # seizure vs rest trained directly as a binary problem
        pb = cross_val_predict(m, X, (y == "S").astype(int), groups=g, cv=cv)
        bin_acc = float((pb == bt).mean())
        res["models"][name] = {"window_acc_5class": float((pred == y).mean()), "segment_acc_5class": seg_acc,
                                "cm": confusion_matrix(y, pred, labels=SETS).tolist(),
                                "seizure_sensitivity": sens, "seizure_specificity": spec, "binary_acc": bin_acc}
    return res


def markdown(r):
    L = ["# Bonn EEG dataset: results of the NeuroPilot pipeline (REAL EEG)", "",
         "Source: Andrzejak et al., Phys Rev E 64, 061907 (2001). Sets: " + "; ".join(f"{k} = {v}" for k, v in DESC.items()) + ".",
         f"Windows: {r['windows']} (1 s each, 128 Hz, band-pass 1-40 Hz + notch) from {r['segments']} recordings. 5-fold CV grouped by recording.", "",
         "| Model | 5-class window acc. | 5-class recording acc. (majority vote) | Seizure sensitivity | Seizure specificity | Seizure-vs-rest acc. |", "|---|---|---|---|---|---|"]
    for n, m in r["models"].items():
        L.append(f"| {n} | {m['window_acc_5class'] * 100:.1f}% | {m['segment_acc_5class'] * 100:.1f}% | {m['seizure_sensitivity'] * 100:.1f}% | {m['seizure_specificity'] * 100:.1f}% | {m['binary_acc'] * 100:.1f}% |")
    m = next(iter(r["models"].values()))
    L += ["", "Confusion matrix (first model; rows = true, columns = predicted, order Z O N F S):", "```"]
    L += [f"{SETS[i]}  " + " ".join(f"{v:4d}" for v in row) for i, row in enumerate(m["cm"])] + ["```", "",
          "## How to read this honestly",
          "- N, F and S are **intracranial** (depth electrode) recordings, not scalp EEG. A wearable scalp device would not record such signals.",
          "- The dataset has no patient IDs: recordings from the same person appear in training and test folds, so results are optimistic compared with testing on new patients.",
          "- Seizure segments have very large amplitude, so seizure-vs-rest is a comparatively easy problem on this dataset. High accuracy here is not evidence of clinical performance.",
          "- This is a research benchmark, not a medical device or diagnostic tool."]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="docs/bonn_results.md")
    a = ap.parse_args()
    md = markdown(run(a.root))
    Path(a.out).write_text(md)
    print(md)
