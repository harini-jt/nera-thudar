import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)

METRIC_NAMES = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
    "mcc",
    "cohen_kappa",
    "log_loss",
    "auc",
    "brier",
    "ece",
]
HIGHER_IS_BETTER = {
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
    "mcc",
    "cohen_kappa",
    "auc",
}
LOWER_IS_BETTER = {
    "log_loss",
    "brier",
    "ece",
    "fit_time_s",
    "predict_time_s",
    "predict_time_per_series_ms",
}


def _nan_on_error(fn, *args, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except ValueError:  # e.g. a class is missing from the test set
        return float("nan")


def brier_score(y_true, proba, classes) -> float:
    """Multiclass Brier score: mean over samples of the squared error summed over classes."""
    onehot = (np.asarray(y_true)[:, None] == classes[None, :]).astype(float)
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def expected_calibration_error(y_true, proba, classes, n_bins: int = 15) -> float:
    """Gap between confidence (max probability) and accuracy, averaged over confidence bins."""
    conf = proba.max(axis=1)
    correct = (classes[proba.argmax(axis=1)] == np.asarray(y_true)).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def compute_metrics(y_true, proba, classes) -> dict:
    y_true, classes = np.asarray(y_true), np.asarray(classes)
    preds = classes[proba.argmax(axis=1)]
    if len(classes) == 2:
        auc = _nan_on_error(roc_auc_score, y_true, proba[:, 1])
    else:
        auc = _nan_on_error(
            roc_auc_score, y_true, proba, multi_class="ovr", labels=classes
        )
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "f1_macro": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, preds, average="weighted", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, preds)),
        "cohen_kappa": float(cohen_kappa_score(y_true, preds)),
        "log_loss": _nan_on_error(log_loss, y_true, proba, labels=classes),
        "auc": auc,
        "brier": brier_score(y_true, proba, classes),
        "ece": expected_calibration_error(y_true, proba, classes),
    }
