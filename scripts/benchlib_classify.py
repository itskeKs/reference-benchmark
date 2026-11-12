"""Single-label classification metrics (stdlib only): accuracy, macro-F1, per-class P/R/F1, confusion matrix.

Input is a list of (gold, predicted) label pairs. Conventions (the same as scikit-learn's defaults):
  - labels are compared as strings; a prediction of None (the system produced nothing for the item) counts as
    wrong, shows up as the pseudo-class "(missing)" in the confusion matrix and is not a class of its own in the
    macro averages (the gold item still counts against its class's recall);
  - macro averages run over the classes that occur in the gold labels or in the predictions;
  - precision / recall of a class without predictions / without gold items is 0.0 (not undefined), F1 0.0 when
    precision + recall is 0;
  - for single-label data micro-F1 equals accuracy, so there is no separate micro metric.
Used by the bench.py init `items` template (bench_stub_metrics.py) when the gold values are plain strings.
"""
from __future__ import annotations

import json

MISSING = "(missing)"
OTHER = "(other)"


def label_of(value):
    """Gold / predicted value -> label string (None stays None; non-strings are JSON-encoded)."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def normalize(pairs):
    return [(label_of(g), label_of(p)) for g, p in pairs]


def class_labels(pairs):
    """Sorted classes that occur as gold or as (non-missing) prediction."""
    return sorted({g for g, _ in pairs if g is not None} | {p for _, p in pairs if p is not None})


def per_class(pairs, labels=None):
    """{label: {"tp", "fp", "fn", "support", "precision", "recall", "f1"}} for every class in `labels`
    (default: every class occurring in the pairs)."""
    pairs = normalize(pairs)
    out = {}
    for label in labels if labels is not None else class_labels(pairs):
        tp = sum(1 for g, p in pairs if g == label and p == label)
        fp = sum(1 for g, p in pairs if p == label and g != label)
        fn = sum(1 for g, p in pairs if g == label and p != label)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[label] = {"tp": tp, "fp": fp, "fn": fn, "support": tp + fn, "precision": prec, "recall": rec,
                      "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    return out


def accuracy(pairs):
    """Share of items whose prediction equals the gold label (None for an empty list)."""
    pairs = normalize(pairs)
    return sum(1 for g, p in pairs if g == p) / len(pairs) if pairs else None


def summarize(pairs, labels=None):
    """{"n", "accuracy", "macro_precision", "macro_recall", "macro_f1", "labels", "per_class"} (None averages without
    classes)."""
    pairs = normalize(pairs)
    classes = per_class(pairs, labels)

    def macro(key):
        return sum(c[key] for c in classes.values()) / len(classes) if classes else None
    return {"n": len(pairs), "accuracy": accuracy(pairs), "macro_precision": macro("precision"),
            "macro_recall": macro("recall"), "macro_f1": macro("f1"), "labels": list(classes), "per_class": classes}


def confusion(pairs, labels=None, max_labels=25):
    """{"rows": [gold labels], "cols": [predicted labels, "(missing)" last when there are missing predictions],
    "matrix": {gold: {pred: count}}, "folded": bool}. With more than max_labels classes the rarest ones are folded
    into "(other)"."""
    pairs = normalize(pairs)
    labels = list(labels) if labels is not None else class_labels(pairs)
    folded = len(labels) > max_labels
    if folded:
        freq = {l: sum(1 for g, p in pairs if l in (g, p)) for l in labels}
        keep = set(sorted(labels, key=lambda l: (-freq[l], l))[:max_labels])
        pairs = [(g if g in keep or g is None else OTHER, p if p in keep or p is None else OTHER) for g, p in pairs]
        labels = sorted(keep) + [OTHER]
    matrix = {l: {} for l in labels}
    for g, p in pairs:
        row = matrix.setdefault(g, {})
        row[p if p is not None else MISSING] = row.get(p if p is not None else MISSING, 0) + 1
    cols = list(labels) + ([MISSING] if any(p is None for _, p in pairs) else [])
    return {"rows": labels, "cols": cols, "matrix": matrix, "folded": folded}


def _table(headers, body):
    def esc(v):
        return str(v).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(esc(h) for h in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    return "\n".join(lines + ["| " + " | ".join(esc(v) for v in row) + " |" for row in body])


def confusion_markdown(pairs, labels=None, max_labels=25):
    """Markdown confusion matrix: rows = gold, columns = predicted, diagonal = correct."""
    c = confusion(pairs, labels, max_labels)
    body = [[g] + [c["matrix"][g].get(p, 0) or "" for p in c["cols"]] for g in c["rows"]]
    text = _table(["gold \\ predicted"] + c["cols"], body)
    if c["folded"]:
        text += "\n\nThe rarest classes are folded into `%s` (max %d classes shown)." % (OTHER, max_labels)
    return text


def per_class_markdown(summary):
    """Markdown table of per-class precision / recall / F1 / support from summarize()."""
    body = [[l, c["support"], "%.3f" % c["precision"], "%.3f" % c["recall"], "%.3f" % c["f1"]]
            for l, c in summary["per_class"].items()]
    return _table(["class", "support", "precision", "recall", "f1"], body)
