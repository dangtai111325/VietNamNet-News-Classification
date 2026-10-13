"""Probability calibration and class-specific decision adjustments."""
import numpy as np


def temperature_probabilities(logits, temperature: float) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(temperature) or temperature <= 0 or not np.isfinite(z).all():
        raise ValueError("Logits and temperature must be finite; temperature must be positive")
    z = z / temperature
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def calibrated_prediction(logits, classes: list[str], temperature: float, thresholds, top_k: int = 3) -> dict:
    p = temperature_probabilities(logits, temperature)
    t = np.asarray(thresholds, dtype=np.float64)
    if p.shape != (len(classes),) or t.shape != p.shape:
        raise ValueError("Logits, thresholds and taxonomy must have the same number of classes")
    if not np.isfinite(t).all() or np.any(t <= 0):
        raise ValueError("All class thresholds must be positive and finite")
    scores = p / t
    # Stable order also matches numpy.argmax when decision scores are tied.
    order = np.argsort(-scores, kind="stable")
    q = scores / scores.sum()
    selected = int(order[0])
    margin = float(q[order[0]] - q[order[1]]) if len(order) > 1 else 1.0
    return {
        "label": classes[selected],
        "confidence": float(p[selected]),
        "margin": margin,
        "candidates": [
            {"label": classes[int(i)], "score": float(p[i]), "decision_score": float(scores[i])}
            for i in order[:max(1, min(int(top_k), 5))]
        ],
    }


def expected_calibration_error(logits, labels, temperature=1.0, n_bins=15) -> float:
    """The bin boundaries intentionally reproduce the notebook's [lo, hi) rule."""
    p = temperature_probabilities(logits, temperature)
    confidence, prediction = p.max(axis=1), p.argmax(axis=1)
    labels = np.asarray(labels)
    total, error = 0, 0.0
    for b in range(n_bins):
        mask = (confidence >= b / n_bins) & (confidence < (b + 1) / n_bins)
        count = int(mask.sum())
        if count:
            error += count * abs(float((prediction[mask] == labels[mask]).mean()) - float(confidence[mask].mean()))
            total += count
    return error / total if total else 0.0
