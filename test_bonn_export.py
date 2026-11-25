import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import export_bonn_model as ex  # noqa: E402


def test_browser_features_shape_and_bands():
    t = np.arange(4097) / 173.61
    f = ex.feats(10 * np.sin(2 * np.pi * 10 * t))
    assert f.shape == (23, 5) and np.isfinite(f).all()
    assert f[:, 2].mean() > 5 * f[:, 3].mean()          # a 10 Hz tone lands in alpha, not beta
