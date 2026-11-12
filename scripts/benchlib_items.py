"""Per-item results: the items.json contract, per-split aggregates and the per-item run comparison.

items.json (written by the metrics command, optional):
  {"schema_version": 1, "items": {"<item_id>": {"split": "train" | "holdout" | ... | null,
                                                 "score": <number | null>,   # higher is better
                                                 "pass": <true | false | null>,
                                                 "detail": {...anything...}}}}
"""
from __future__ import annotations

import math

from benchlib_util import UsageError, is_num, md_table, read_json


def validate_items(obj):
    """-> list of problems (empty = valid)."""
    if not isinstance(obj, dict) or obj.get("schema_version") != 1 or not isinstance(obj.get("items"), dict):
        return ['top level must be {"schema_version": 1, "items": {...}}']
    problems = []
    for iid, item in obj["items"].items():
        if not isinstance(item, dict):
            problems.append("item %r must be an object" % iid)
            continue
        if item.get("split") is not None and not (isinstance(item["split"], str) and item["split"]):
            problems.append("item %r: split must be a non-empty string or null" % iid)
        if item.get("score") is not None and not (is_num(item["score"]) and math.isfinite(item["score"])):
            problems.append("item %r: score must be a finite number or null" % iid)
        if item.get("pass") is not None and not isinstance(item["pass"], bool):
            problems.append("item %r: pass must be true, false or null" % iid)
    return problems[:5]


def load_items(run_dir):
    """items dict of a run dir, or None when the run has no items.json; invalid -> UsageError."""
    path = run_dir / "items.json"
    if not path.exists():
        return None
    obj = read_json(path)
    problems = validate_items(obj)
    if problems:
        raise UsageError("%s is invalid: %s" % (path, "; ".join(problems)))
    return obj["items"]


def split_aggregates(items):
    """{<split>.score_mean, <split>.pass_rate, <split>.n} for every split name that occurs (null splits are skipped)."""
    out = {}
    for split in sorted({i.get("split") for i in items.values() if i.get("split")}):
        members = [i for i in items.values() if i.get("split") == split]
        scores = [i["score"] for i in members if is_num(i.get("score"))]
        passes = [i["pass"] for i in members if isinstance(i.get("pass"), bool)]
        out["%s.score_mean" % split] = math.fsum(scores) / len(scores) if scores else None
        out["%s.pass_rate" % split] = sum(passes) / len(passes) if passes else None
        out["%s.n" % split] = len(members)
    return out


def classify(a, b, threshold):
    """-> (verdict, delta) for one item present in both runs. A pass flip decides; else the score delta vs threshold."""
    delta = b["score"] - a["score"] if is_num(a.get("score")) and is_num(b.get("score")) else None
    pa, pb = a.get("pass"), b.get("pass")
    if isinstance(pa, bool) and isinstance(pb, bool) and pa != pb:
        return ("improved" if pb else "regressed"), delta
    if delta is None:
        return "unchanged", None
    return ("regressed" if delta < -threshold else "improved" if delta > threshold else "unchanged"), delta


def compare_items(a_items, b_items, *, split=None, threshold=0.0, hide_split=None):
    """Per item: regressed / improved / unchanged / only_in_a / only_in_b (B relative to A, score deltas).

    split keeps only items of that split; hide_split (the holdout guard) drops that split and counts the items,
    unless split names it explicitly."""
    rows = {"regressed": [], "improved": [], "unchanged": [], "only_in_a": [], "only_in_b": []}
    hidden = 0
    for iid in sorted(set(a_items) | set(b_items)):
        a, b = a_items.get(iid), b_items.get(iid)
        item_split = (a or {}).get("split") or (b or {}).get("split")
        if hide_split and item_split == hide_split and split != hide_split:
            hidden += 1
            continue
        if split is not None and item_split != split:
            continue
        entry = {"item": iid, "split": item_split}
        if a is None or b is None:
            only = a if b is None else b
            entry.update(score=only.get("score"), **{"pass": only.get("pass")})
            rows["only_in_a" if b is None else "only_in_b"].append(entry)
            continue
        verdict, delta = classify(a, b, threshold)
        entry.update(score_a=a.get("score"), score_b=b.get("score"), pass_a=a.get("pass"), pass_b=b.get("pass"), delta=delta)
        rows[verdict].append(entry)
    worst_first = lambda e: (e["delta"] is None, e["delta"] if e["delta"] is not None else 0.0, e["item"])
    rows["regressed"].sort(key=worst_first)
    rows["unchanged"].sort(key=worst_first)
    rows["improved"].sort(key=lambda e: (e["delta"] is None, -(e["delta"] or 0.0), e["item"]))
    return {"counts": {k: len(v) for k, v in rows.items()}, "threshold": threshold, "split": split,
            "hidden_holdout_items": hidden, **rows}


def render_items_md(res, a_label, b_label, hide_split=None):
    lines = ["# Items: %s -> %s" % (a_label, b_label), "",
             "Deltas are B minus A (score: higher is better); a pass flip decides over the score; threshold %g%s." % (
                 res["threshold"], "; split `%s` only" % res["split"] if res["split"] else "")]
    c = res["counts"]
    lines.append("regressed %d | improved %d | unchanged %d | only in A %d | only in B %d" % (
        c["regressed"], c["improved"], c["unchanged"], c["only_in_a"], c["only_in_b"]))
    if res["hidden_holdout_items"]:
        lines.append("%d item(s) of the holdout split `%s` are hidden (--show-holdout for the final check)." % (
            res["hidden_holdout_items"], hide_split))
    for title, key in (("Regressed", "regressed"), ("Improved", "improved")):
        if res[key]:
            lines += ["", "## %s" % title, "", md_table(["item", "split", "score A", "score B", "delta", "pass A", "pass B"],
                      [[e["item"], e["split"], e["score_a"], e["score_b"], e["delta"], e["pass_a"], e["pass_b"]] for e in res[key]])]
    for title, key in (("Only in A", "only_in_a"), ("Only in B", "only_in_b")):
        if res[key]:
            lines += ["", "## %s" % title, "", md_table(["item", "split", "score", "pass"],
                      [[e["item"], e["split"], e["score"], e["pass"]] for e in res[key]])]
    if res["unchanged"]:
        lines += ["", "## Unchanged (%d)" % len(res["unchanged"]), "", ", ".join(e["item"] for e in res["unchanged"])]
    return "\n".join(lines) + "\n"
