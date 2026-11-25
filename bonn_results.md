# Bonn EEG dataset: results of the NeuroPilot pipeline (REAL EEG)

Source: Andrzejak et al., Phys Rev E 64, 061907 (2001). Sets: Z = healthy, eyes open; O = healthy, eyes closed; N = interictal (opposite hemisphere); F = interictal (epileptogenic zone); S = seizure.
Windows: 11500 (1 s each, 128 Hz, band-pass 1-40 Hz + notch) from 500 recordings. 5-fold CV grouped by recording.

| Model | 5-class window acc. | 5-class recording acc. (majority vote) | Seizure sensitivity | Seizure specificity | Seizure-vs-rest acc. |
|---|---|---|---|---|---|
| Logistic regression (NeuroPilot decoder) | 68.1% | 76.2% | 94.0% | 98.2% | 96.9% |
| Random forest | 70.7% | 81.0% | 95.9% | 98.1% | 97.7% |

Confusion matrix (first model; rows = true, columns = predicted, order Z O N F S):
```
Z  1723  343  162   72    0
O   513 1719   11   56    1
N   200   52 1176  840   32
F   177   52  887 1055  129
S     0   47   21   71 2161
```

## How to read this honestly
- N, F and S are **intracranial** (depth electrode) recordings, not scalp EEG. A wearable scalp device would not record such signals.
- The dataset has no patient IDs: recordings from the same person appear in training and test folds, so results are optimistic compared with testing on new patients.
- Seizure segments have very large amplitude, so seizure-vs-rest is a comparatively easy problem on this dataset. High accuracy here is not evidence of clinical performance.
- This is a research benchmark, not a medical device or diagnostic tool.