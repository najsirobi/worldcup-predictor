"""Classification / probability metrics for W/D/L backtests.

Primary metrics are log loss, Brier score, calibration and accuracy — NOT
R-squared (R^2 is only a secondary metric for goal-count regressions).
"""
import numpy as np
import pandas as pd

CLASSES = ["home_win", "draw", "away_win"]


def _onehot(y_true):
    y = np.asarray(y_true)
    return np.stack([(y == c).astype(float) for c in CLASSES], axis=1)


def multiclass_log_loss(y_true, proba, eps=1e-15):
    P = np.clip(np.asarray(proba), eps, 1 - eps)
    P = P / P.sum(axis=1, keepdims=True)
    Y = _onehot(y_true)
    return float(-(Y * np.log(P)).sum(axis=1).mean())


def multiclass_brier(y_true, proba):
    Y = _onehot(y_true)
    return float(((np.asarray(proba) - Y) ** 2).sum(axis=1).mean())


def accuracy(y_true, proba):
    pred = np.asarray(proba).argmax(axis=1)
    true = np.array([CLASSES.index(c) for c in y_true])
    return float((pred == true).mean())


def avg_prob_on_actual(y_true, proba):
    Y = _onehot(y_true)
    return float((np.asarray(proba) * Y).sum(axis=1).mean())


def confusion(y_true, proba):
    pred = np.asarray(proba).argmax(axis=1)
    true = np.array([CLASSES.index(c) for c in y_true])
    M = np.zeros((3, 3), dtype=int)
    for t, p in zip(true, pred):
        M[t, p] += 1
    return pd.DataFrame(M, index=[f"true_{c}" for c in CLASSES],
                        columns=[f"pred_{c}" for c in CLASSES])


def calibration_table(y_true, proba, n_bins=10):
    """Reliability table over the max predicted probability."""
    p = np.asarray(proba)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    true = np.array([CLASSES.index(c) for c in y_true])
    correct = (pred == true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({
            "bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
            "n": int(m.sum()),
            "mean_pred_conf": round(float(conf[m].mean()), 3),
            "empirical_acc": round(float(correct[m].mean()), 3),
        })
    return pd.DataFrame(rows)


def all_wdl_metrics(y_true, proba):
    return {
        "log_loss": round(multiclass_log_loss(y_true, proba), 4),
        "brier": round(multiclass_brier(y_true, proba), 4),
        "accuracy": round(accuracy(y_true, proba), 4),
        "avg_prob_on_actual": round(avg_prob_on_actual(y_true, proba), 4),
    }
