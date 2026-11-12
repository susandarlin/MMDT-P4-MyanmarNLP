"""
Task 3 - Evaluation metrics (built from scratch).

    Accuracy  = correct / total
    Precision = TP / (TP + FP)      "when we predicted class k, how often right?"
    Recall    = TP / (TP + FN)      "of the real class k, how many did we find?"
    F1        = 2 * P * R / (P + R)
    Macro-F1  = average of the per-class F1 (each class counts equally)
    Confusion matrix: rows = true label, columns = predicted label
"""

import numpy as np

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def normalize_labels(values):
    """Clean labels so small typos don't silently drop rows:
    ' supports ' -> 'SUPPORTS', 'not  enough info' -> 'NOT ENOUGH INFO'."""
    return [" ".join(str(v).split()).upper() for v in values]


def confusion_matrix(y_true, y_pred, labels=LABELS):
    y_true, y_pred = normalize_labels(y_true), normalize_labels(y_pred)
    idx = {c: i for i, c in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm


def classification_report(y_true, y_pred, labels=LABELS):
    y_true, y_pred = normalize_labels(y_true), normalize_labels(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels)
    tp = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0)          # column sums
    actual = cm.sum(axis=1)             # row sums
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    return {
        "labels": list(labels),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "precision": precision, "recall": recall, "f1": f1, "support": actual,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": cm,
    }


def print_report(r):
    print(f"\n{'label':<17}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}")
    for i, c in enumerate(r["labels"]):
        print(f"{c:<17}{r['precision'][i]:>10.3f}{r['recall'][i]:>9.3f}"
              f"{r['f1'][i]:>8.3f}{r['support'][i]:>9d}")
    print(f"{'macro avg':<17}{r['macro_precision']:>10.3f}{r['macro_recall']:>9.3f}"
          f"{r['macro_f1']:>8.3f}")
    print(f"\nAccuracy : {r['accuracy']:.3f}")
    print(f"Macro-F1 : {r['macro_f1']:.3f}")
    short = {"SUPPORTS": "SUP", "REFUTES": "REF", "NOT ENOUGH INFO": "NEI"}
    print("\nConfusion matrix (rows = true, columns = predicted)")
    print(" " * 6 + "".join(f"{short.get(c, c):>6}" for c in r["labels"]))
    for i, c in enumerate(r["labels"]):
        print(f"{short.get(c, c):<6}" + "".join(f"{v:>6d}" for v in r["confusion_matrix"][i]))
