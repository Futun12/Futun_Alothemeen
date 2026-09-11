"""Lab 6: sliced evaluation report."""

from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score


SMALL_SLICE_THRESHOLD = 30


def _length_bucket(text):
    length = len(str(text).split())

    if length <= 10:
        return "short"

    if length <= 25:
        return "medium"

    return "long"


def _evaluate_slice(y_true, y_pred, indices):
    true_values = [
        y_true[i]
        for i in indices
    ]

    pred_values = [
        y_pred[i]
        for i in indices
    ]

    score = f1_score(
        true_values,
        pred_values,
        average="macro",
        zero_division=0,
    )

    n = len(indices)

    return {
        "n": n,
        "macro_f1": float(score),
        "small_slice": n < SMALL_SLICE_THRESHOLD,
    }


def sliced_report(
    y_true,
    y_pred,
    *,
    languages=None,
    dialects=None,
    texts=None,
):
    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length"
        )

    n = len(y_true)

    if n == 0:
        raise ValueError(
            "evaluation data must not be empty"
        )

    if languages is not None and len(languages) != n:
        raise ValueError(
            "languages must match y_true length"
        )

    if dialects is not None and len(dialects) != n:
        raise ValueError(
            "dialects must match y_true length"
        )

    if texts is not None and len(texts) != n:
        raise ValueError(
            "texts must match y_true length"
        )

    report = {
        "overall": {
            "n": n,
            "macro_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            "small_slice": (
                n < SMALL_SLICE_THRESHOLD
            ),
        },
        "language": {},
        "dialect": {},
        "class": {},
        "length": {},
    }

    # -----------------------------------------------------
    # Language slices
    # -----------------------------------------------------

    if languages is not None:
        groups = defaultdict(list)

        for i, value in enumerate(languages):
            groups[str(value)].append(i)

        for value, indices in groups.items():
            report["language"][value] = (
                _evaluate_slice(
                    y_true,
                    y_pred,
                    indices,
                )
            )

    # -----------------------------------------------------
    # Dialect slices
    # -----------------------------------------------------

    if dialects is not None:
        groups = defaultdict(list)

        for i, value in enumerate(dialects):
            groups[str(value)].append(i)

        for value, indices in groups.items():
            report["dialect"][value] = (
                _evaluate_slice(
                    y_true,
                    y_pred,
                    indices,
                )
            )

    # -----------------------------------------------------
    # Class slices
    # -----------------------------------------------------

    groups = defaultdict(list)

    for i, label in enumerate(y_true):
        groups[str(label)].append(i)

    for label, indices in groups.items():
        report["class"][label] = (
            _evaluate_slice(
                y_true,
                y_pred,
                indices,
            )
        )

    # -----------------------------------------------------
    # Length slices
    # -----------------------------------------------------

    if texts is not None:
        groups = defaultdict(list)

        for i, text in enumerate(texts):
            bucket = _length_bucket(text)
            groups[bucket].append(i)

        for bucket, indices in groups.items():
            report["length"][bucket] = (
                _evaluate_slice(
                    y_true,
                    y_pred,
                    indices,
                )
            )

    return report