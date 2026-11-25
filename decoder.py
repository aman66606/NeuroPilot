"""Scikit-learn decoder trained on labelled SIMULATED windows; closed-loop retraining."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ..core.signal import CLASSES, extract_features, generate_window, preprocess


def itr(acc, n=5, window_s=1.0):
    """Wolpaw information transfer rate in bits/min (theoretical)."""
    if acc <= 1 / n:
        return 0.0
    bits = np.log2(n) + acc * np.log2(acc) + ((1 - acc) * np.log2((1 - acc) / (n - 1)) if acc < 1 else 0)
    return float(bits * 60 / window_s)


def make_dataset(n_per_class, noises, seed):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i in range(n_per_class * len(CLASSES)):
        c = i % len(CLASSES)
        noise = noises[(i // len(CLASSES)) % len(noises)] if isinstance(noises, (list, tuple)) else noises
        X.append(extract_features(preprocess(generate_window(c, noise, rng)))[0])
        y.append(c)
    return np.array(X), np.array(y)


class Decoder:
    def __init__(self, seed=42, threshold=0.6):
        self.seed, self.threshold = seed, threshold
        self.verified, self.metrics, self.last_retrain = [], None, None
        self.pipe = None

    def _fit(self, X, y):
        self.pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=self.seed))
        self.pipe.fit(X, y)

    def _evaluate(self, X, y):
        pred = self.pipe.predict(X)
        return {"accuracy": float((pred == y).mean()), "macro_f1": float(f1_score(y, pred, average="macro")),
                "confusion_matrix": confusion_matrix(y, pred, labels=range(len(CLASSES))).tolist(), "n": int(len(y))}

    def train(self, n_per_class=120, seed=None):
        self.seed = seed if seed is not None else self.seed
        X, y = make_dataset(n_per_class, [2, 4, 6], self.seed)
        self.Xtr, self.Xte, self.ytr, self.yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=self.seed)
        self._fit(self.Xtr, self.ytr)
        self.verified = []
        self.metrics = {**self._evaluate(self.Xte, self.yte), "train_size": int(len(self.ytr)), "classes": CLASSES,
                        "synthetic": True}
        self.metrics["itr_bits_min"] = itr(self.metrics["accuracy"])
        return self.metrics

    def predict(self, feats):
        proba = self.pipe.predict_proba([feats])[0]
        k = int(np.argmax(proba))
        conf = float(proba[k])
        return CLASSES[k], conf, conf >= self.threshold, proba.tolist()

    def add_verified(self, feats, label):
        """Only human- or ground-truth-verified labels may enter training."""
        self.verified.append((np.asarray(feats), CLASSES.index(label)))

    def retrain(self, field_noise=12.0, repeat=3):
        if not self.verified:
            raise ValueError("no verified examples")
        Xf, yf = make_dataset(40, field_noise, 999)
        before = self._evaluate(Xf, yf)
        X = np.vstack([self.Xtr] + [np.array([v[0] for v in self.verified])] * repeat)
        y = np.concatenate([self.ytr] + [np.array([v[1] for v in self.verified])] * repeat)
        self._fit(X, y)
        after = self._evaluate(Xf, yf)
        self.metrics.update(self._evaluate(self.Xte, self.yte))
        self.last_retrain = {"field_noise": field_noise, "verified_used": len(self.verified), "before": before, "after": after}
        return self.last_retrain
