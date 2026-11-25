"""Synthetic EEG-like generation, filtering and feature extraction (SIMULATED DATA)."""
import numpy as np
from scipy import signal as sp

FS, N, PAD = 128, 128, 64
CH = ["C3", "Cz", "C4", "Pz"]
CLASSES = ["LEFT", "RIGHT", "FORWARD", "BACKWARD", "SELECT"]
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}


def _signatures():
    out = []
    for i in range(len(CLASSES)):
        r = np.random.default_rng(100 + i * 17)
        out.append([[3 + 10 * r.random(), 2 + 7 * r.random(), 2 + 6 * r.random()] for _ in CH])
    return np.array(out)  # (class, channel, [alpha, beta, theta])


SIG = _signatures()


def generate_window(cls_idx, noise=6.0, rng=None, artifact_prob=0.0):
    """Return a (4, N+PAD) simulated window for a class. Not real brain data."""
    rng = rng or np.random.default_rng()
    t = np.arange(N + PAD) / FS
    out = np.zeros((len(CH), N + PAD))
    for c in range(len(CH)):
        a = SIG[cls_idx, c] * (1 + 0.25 * rng.standard_normal(3))
        ph = rng.uniform(0, 2 * np.pi, 3)
        out[c] = (a[0] * np.sin(2 * np.pi * 10 * t + ph[0]) + a[1] * np.sin(2 * np.pi * 20 * t + ph[1])
                  + a[2] * np.sin(2 * np.pi * 6 * t + ph[2]) + noise * rng.standard_normal(N + PAD)
                  + 3 * np.sin(2 * np.pi * 50 * t) + 6 * np.sin(2 * np.pi * 0.3 * t + ph[0]))
    if rng.random() < artifact_prob:
        out[0, 100:115] += 70 * np.sin(np.linspace(0, np.pi, 15))
    return out


_SOS = sp.butter(4, [1, 40], btype="band", fs=FS, output="sos")
_NB, _NA = sp.iirnotch(50, 30, fs=FS)


def preprocess(window):
    """1-40 Hz band-pass + 50 Hz notch (zero-phase); trims filtfilt edge transients from both ends."""
    f = sp.sosfiltfilt(_SOS, window, axis=1)
    f = sp.filtfilt(_NB, _NA, f, axis=1)
    return f[:, PAD // 2: PAD // 2 + N]


def has_artifact(filtered, thr=45.0):
    return bool(np.abs(filtered).max() > thr)


def signal_quality(noise, artifact):
    """Heuristic 0-100 score from the simulated noise level and artifact flag."""
    return int(max(0, round(100 - 3.5 * noise - (25 if artifact else 0))))


def extract_features(filtered):
    """20 features: log band power (delta, theta, alpha, beta) + log variance per channel.
    Returns (features, mean band power dict, mean PSD over channels)."""
    freqs = np.fft.rfftfreq(N, 1 / FS)
    psd = np.abs(np.fft.rfft(filtered, axis=1)) ** 2 / N ** 2
    feats, bp = [], {b: 0.0 for b in BANDS}
    for c in range(filtered.shape[0]):
        for b, (lo, hi) in BANDS.items():
            p = psd[c][(freqs >= lo) & (freqs < hi)].sum()
            feats.append(np.log10(1 + p))
            bp[b] += p / filtered.shape[0]
        feats.append(np.log10(1 + np.mean(filtered[c] ** 2)))
    return np.array(feats), bp, psd.mean(axis=0)


FEATURE_NAMES = [f"{c} {b}" for c in CH for b in ("Delta", "Theta", "Alpha", "Beta", "LogVar")]
