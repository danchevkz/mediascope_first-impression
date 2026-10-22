import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .data import LABELS


def score(task, truth, predicted):
    labels = LABELS[task]
    if not len(truth) or len(truth) != len(predicted):
        raise ValueError('Expected equally sized, non-empty predictions and targets')
    if set(truth) - set(labels) or set(predicted) - set(labels):
        raise ValueError('Unknown label; failed API requests must not be silently dropped')
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0)
    order = {label: i for i, label in enumerate(labels)}
    delta = np.array([order[p] - order[t] for t, p in zip(truth, predicted, strict=True)])
    return {
        'n': len(truth), 'accuracy': float(accuracy_score(truth, predicted)),
        'macro_precision': float(precision.mean()), 'macro_recall': float(recall.mean()),
        'macro_f1': float(f1.mean()), 'mae': float(np.abs(delta).mean()),
        'mse': float(np.square(delta).mean()),
        'per_class': [dict(label=label, precision=float(precision[i]), recall=float(recall[i]),
                           f1=float(f1[i]), support=int(support[i])) for i, label in enumerate(labels)],
        'confusion_matrix': confusion_matrix(truth, predicted, labels=labels).tolist(),
    }
