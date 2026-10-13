from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def ensure_edaspy_importable() -> None:
    """Add common local EDAspy paths to sys.path when available."""
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "EDAspy",
        Path("/app/EDAspy"),
    ]
    for candidate in candidates:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)


def enable_adaptive_bn_folds_patch() -> bool:
    """
    Patch BN-based learn() methods so num_folds never exceeds a safe small-sample limit.

    This avoids brittle failures in small smoke runs and in early generations
    where the elite/archive is still small.
    """
    ensure_edaspy_importable()
    try:
        from EDAspy.optimization.custom.probabilistic_models.semiparametric_bayesian_network import SPBN
        from EDAspy.optimization.custom.probabilistic_models.kde_bayesian_network import KDEBN
    except Exception:
        return False

    patched_any = False

    for cls in (SPBN, KDEBN):
        if getattr(cls, "_adaptive_folds_patch", False):
            patched_any = True
            continue

        original_learn = cls.learn

        def patched_learn(self, dataset: np.ndarray, num_folds: int = 10, *args, _original_learn=original_learn, **kwargs):
            n_samples = int(np.asarray(dataset).shape[0])
            safe_folds = int(num_folds)
            if n_samples >= 3:
                safe_folds = max(2, min(int(num_folds), n_samples - 1))
            return _original_learn(self, dataset, num_folds=safe_folds, *args, **kwargs)

        cls.learn = patched_learn
        cls._adaptive_folds_patch = True
        patched_any = True

    return patched_any
