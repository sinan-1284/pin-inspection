"""One-class model of good pins: Mahalanobis distance from their mean.

Saved as JSON (mean, inverse covariance, threshold, feature names) so it is
readable, diffable and safe to load.
"""
import json
from pathlib import Path

import numpy as np


def value(res, name):
    """Look a measurement up by name: mm first, then ratios, then px."""
    for section in ("mm", "ratios", "px"):
        if name in res.get(section, {}):
            return res[section][name]
    if name in res:
        return res[name]
    raise KeyError(name)


def units(res):
    return "mm" if "mm" in res else "px"


def vector(res, names):
    return np.array([value(res, n) for n in names], dtype=float)


class PinModel:
    def __init__(self, names, mean, inv_cov, threshold, units, mm_per_px, n_train):
        self.names = names
        self.mean = np.asarray(mean, float)
        self.inv_cov = np.asarray(inv_cov, float)
        self.threshold = threshold
        self.units = units
        self.mm_per_px = mm_per_px
        self.n_train = n_train

    @classmethod
    def fit(cls, results, cfg_model, mm_per_px):
        names = list(cfg_model["features"])
        X = np.array([vector(r, names) for r in results])
        n, d = X.shape
        mean = X.mean(axis=0)
        cov = np.cov(X.T) if n > 1 else np.zeros((d, d))
        # Shrink towards the diagonal when there are few samples, and never
        # let a feature's spread drop below its noise floor.
        shrink = min(1.0, d / max(n, 1))
        cov = (1 - shrink) * cov + shrink * np.diag(np.diag(cov))
        floor = np.array([cfg_model["features"][k] for k in names]) ** 2
        cov += np.diag(np.maximum(floor - np.diag(cov), 0))
        model = cls(names, mean, np.linalg.inv(cov), None, units(results[0]), mm_per_px, n)
        worst = max(model.score(r) for r in results)
        model.threshold = max(cfg_model["min_threshold"], cfg_model["threshold_multiplier"] * worst)
        return model

    def score(self, res):
        d = vector(res, self.names) - self.mean
        return float(np.sqrt(d @ self.inv_cov @ d))

    def contributions(self, res):
        """Per-feature deviation in sigmas (sign shows direction)."""
        sigma = np.sqrt(np.diag(np.linalg.inv(self.inv_cov)))
        return dict(zip(self.names, ((vector(res, self.names) - self.mean) / sigma).round(1).tolist()))

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "features": self.names, "mean": self.mean.tolist(), "inv_cov": self.inv_cov.tolist(),
            "threshold": self.threshold, "units": self.units, "mm_per_px": self.mm_per_px,
            "n_train": self.n_train,
        }, indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["features"], d["mean"], d["inv_cov"], d["threshold"], d["units"],
                   d["mm_per_px"], d["n_train"])
