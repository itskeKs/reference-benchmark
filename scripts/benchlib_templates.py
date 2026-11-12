"""`bench.py init` templates: a text/WER stub and a generic multi-item template (stdlib only).

The items template's metrics stub emits split-prefixed metrics only (holdout guard) and uses benchlib_classify.py for
single-label classification gold."""
from __future__ import annotations

import json

TEMPLATES = ("text", "items")

# ---------------------------------------------------------------- text template

TEXT_DRIVER = '''#!/usr/bin/env python3
"""Stub driver for bench.py: copies fixture 'input' to result.json["output"].

Config keys: drop_every=N drops every Nth word (in bench.json defaults). Fault injection (new keys: pass
--allow-new-keys): fail="error" (result with status error) | "crash" (exit 3, no result) | "killed" (exit 75)
| "sleep:<seconds>"; fail_attempts=N makes only the first N invocations fail (counter file in the history root)."""
import json
import os
import sys
import time
from pathlib import Path

run_dir = Path(os.environ["BENCH_RUN_DIR"])
cfg = json.loads(Path(os.environ["BENCH_CONFIG"]).read_text())
(run_dir / "pid.txt").write_text(str(os.getpid()))
counter = run_dir.parent.parent / "stub_counter.txt"
calls = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(calls))
result = {"schema_version": 1, "status": "success"}
fail = cfg.get("fail")
if fail and calls <= cfg.get("fail_attempts", calls):
    if fail == "killed":
        sys.exit(75)
    if fail == "crash":
        sys.exit(3)
    if fail.startswith("sleep:"):
        time.sleep(float(fail[6:]))
    if fail == "error":
        result = {"schema_version": 1, "status": "error", "error": {"code": "stub_error", "message": "requested by config"}}
if result["status"] == "success":
    words = Path(os.environ["BENCH_FIXTURE_INPUT"]).read_text(encoding="utf-8").split()
    if cfg.get("drop_every"):
        words = [w for i, w in enumerate(words, 1) if i % cfg["drop_every"]]
    result.update(output=" ".join(words), timings_ms={"total": 1})
tmp = run_dir / ".result.json.tmp"
tmp.write_text(json.dumps(result))
os.replace(tmp, run_dir / "result.json")
'''

TEXT_METRICS = '''#!/usr/bin/env python3
"""Stub metrics for bench.py: WER of result.json["output"] vs fixture 'reference' (one segment per line).

Needs textdiff.py next to this file (bench.py init copies it)."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import textdiff

run_dir = Path(os.environ["BENCH_RUN_DIR"])


def write(name, text):  # tmp + rename: readers never see a half-written file
    tmp = run_dir / ("." + name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, run_dir / name)


output = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))["output"]
reference = Path(os.environ["BENCH_FIXTURE_REFERENCE"]).read_text(encoding="utf-8")
segments = [{"id": str(n), "text": t} for n, t in enumerate(reference.splitlines(), 1) if t.strip()]
report = textdiff.build_report(segments, output)
s = report["summary"]
write("metrics.json", json.dumps({k: s[k] for k in ("wer", "S", "D", "I", "N_ref", "N_hyp")}))
write("metrics.md", "WER %.2f %% (S %d, D %d, I %d)\\n" % (100 * s["wer"], s["S"], s["D"], s["I"]))
write("diff.md", textdiff.render_markdown(report, "Diff"))
'''

TEXT_CONFIG = {
    "suite": "starter", "history_root": "history",
    "fixtures": {"input": "fixtures/input.txt", "reference": "fixtures/reference.txt"},
    "driver": ["python3", "bench_stub_driver.py"], "metrics": ["python3", "bench_stub_metrics.py"],
    "defaults": {"drop_every": 0}, "timeout_seconds": 600, "retries": 0,
    "index_columns": {"config": ["drop_every"], "metrics": ["wer", "S", "D", "I"]},
    "primary_metric": {"name": "wer", "goal": "min"}, "criteria": {"wer": "<=0.5"},
}

# ---------------------------------------------------------------- items template

ITEM_ROWS = [  # (id, split, input text, gold)
    ("item-01", "train", "Order 10234 from Alice Meyer, total 19.99 EUR, date 2026-01-03",
     {"order_id": "10234", "customer": "Alice Meyer", "total": "19.99", "date": "2026-01-03"}),
    ("item-02", "train", "Order 10235 from Bob Stone, total 5.50 EUR, date 2026-01-04",
     {"order_id": "10235", "customer": "Bob Stone", "total": "5.50", "date": "2026-01-04"}),
    ("item-03", "train", "Order 10236 from Carla Diaz, total 120.00 EUR, date 2026-01-05",
     {"order_id": "10236", "customer": "Carla Diaz", "total": "120.00", "date": "2026-01-05"}),
    ("item-04", "train", "Order 10237 from Dmitri Petrov, total 42.10 EUR",
     {"order_id": "10237", "customer": "Dmitri Petrov", "total": "42.10"}),
    ("item-05", "train", "Order 10238 from Anna von Berg, total 7.25 EUR, date 2026-02-01",
     {"order_id": "10238", "customer": "Anna von Berg", "total": "7.25", "date": "2026-02-01"}),
    ("item-06", "holdout", "Order 10239 from Eve Adams, total 99.99 EUR, date 2026-02-02",
     {"order_id": "10239", "customer": "Eve Adams", "total": "99.99", "date": "2026-02-02"}),
    ("item-07", "holdout", "Order 10240 from Frank Li, total 1.00 EUR, date 2026-02-03",
     {"order_id": "10240", "customer": "Frank Li", "total": "1.00", "date": "2026-02-03"}),
    ("item-08", "holdout", "Order 10241 from Grace O'Hara, total 15.75 EUR, date 2026-02-04",
     {"order_id": "10241", "customer": "Grace O'Hara", "total": "15.75", "date": "2026-02-04"}),
]

ITEMS_DRIVER = '''#!/usr/bin/env python3
"""Stub driver for bench.py (items template): a toy rule-based extractor run over every item of fixture 'items'.

Replace extract() with a call into the real system under test (same code path as production), keep the
rest: read the items, write result.json atomically.

Config keys (bench.json "defaults"): extract_date (bool) also extract the date; lowercase_customer (bool)
degrades the customer field on purpose. Fault injection (new key: pass --allow-new-keys):
fail="error" (result with status error) | "crash" (exit 3, no result) | "killed" (exit 75).

Systems that FIT on gold (classifiers, learned thresholds, few-shot examples picked from gold): never read the gold
of holdout items. bench.py hands every fixture to the driver, so the split assignment is available here too
(fixture 'splits' -> env BENCH_FIXTURE_SPLITS, {"train": [ids], "dev": [ids], "holdout": [ids]}; env
BENCH_HOLDOUT_SPLIT names the holdout split):
    splits = json.loads(Path(os.environ["BENCH_FIXTURE_SPLITS"]).read_text())
    training_ids = set(splits.get("train", []))   # read gold (env BENCH_FIXTURE_GOLD) ONLY for these ids, fit on them
    # then predict every item, holdout included, without ever having seen its gold
Report scores on items the system did not train on (a `dev` split, or cross-validation inside train; the metrics
script emits them as <split>.<metric>) and set "rank_split": "dev" (or "cv") in bench.json: sweeps and criteria then
rank on those numbers, never on scores of items the system was fitted to."""
import json
import os
import re
import sys
import time
from pathlib import Path

run_dir = Path(os.environ["BENCH_RUN_DIR"])
cfg = json.loads(Path(os.environ["BENCH_CONFIG"]).read_text())
items_dir = Path(os.environ["BENCH_FIXTURE_ITEMS"])


def write_result(result):
    tmp = run_dir / ".result.json.tmp"
    tmp.write_text(json.dumps(result), encoding="utf-8")
    os.replace(tmp, run_dir / "result.json")


def extract(text):
    out = {}
    patterns = {"order_id": r"Order (\\d+)", "customer": r"from ([A-Z][a-z]+(?: [A-Z][a-z]+)*)", "total": r"total ([\\d.]+)"}
    if cfg.get("extract_date", True):
        patterns["date"] = r"date (\\d{4}-\\d{2}-\\d{2})"
    for field, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            out[field] = m.group(1)
    if cfg.get("lowercase_customer") and "customer" in out:
        out["customer"] = out["customer"].lower()
    return out


fail = cfg.get("fail")
if fail == "killed":
    sys.exit(75)
if fail == "crash":
    sys.exit(3)
if fail == "error":
    write_result({"schema_version": 1, "status": "error", "error": {"code": "stub_error", "message": "requested by config"}})
    sys.exit(0)
outputs, timings = {}, {}
for path in sorted(items_dir.glob("*.json")):
    item = json.loads(path.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    outputs[item["id"]] = extract(item["input"])
    timings[item["id"]] = round((time.perf_counter() - t0) * 1000, 4)
write_result({"schema_version": 1, "status": "success", "outputs": outputs, "timings_ms": timings,
              "latency_ms": round(sum(timings.values()), 4)})
'''

ITEMS_METRICS = r'''#!/usr/bin/env python3
"""Stub metrics for bench.py (items template): per-item scoring against fixture 'gold'.

Gold values that are dicts are scored field by field (a field is correct when key and value match gold); gold values
that are plain strings are single-label classification (equality; benchlib_classify.py adds accuracy, macro-F1,
per-class P/R/F1 and a confusion matrix). Item score = field F1 (classification: 1 or 0), pass = exact equality.

Metric names follow the holdout guard: every quality metric is SPLIT-PREFIXED (train.f1, dev.f1, holdout.f1, ...).
An unprefixed aggregate (accuracy, f1) would mix the holdout items into the number you tune on; bench.py hides such
keys, and its criteria / rankings use <rank_split>.<metric>. Writes
  metrics.json  per split <split>.precision / .recall / .f1 / .exact_rate (extraction: micro over fields;
                classification: macro P / R / F1, exact_rate = accuracy) and, for classification, .accuracy and
                .macro_f1; plus latency_ms (a timing key). bench.py adds <split>.score_mean / .pass_rate / .n from
                items.json. Items that are in no split are left out of the split metrics. all.* (over EVERY item,
                holdout included) is written only when BENCH_SHOW_HOLDOUT=1. When no item has a split at all, plain
                unprefixed metrics are written instead (there is no holdout to protect).
  items.json    per item: split, score, pass, detail. Raw per-item data, unfiltered: `bench.py items` hides the
                holdout items itself, so keep them in the file.
  metrics.md, diff.md  the human reports. They leave the holdout split out unless BENCH_SHOW_HOLDOUT=1 (bench.py sets
                it only for `bench.py metrics RUN --show-holdout`; BENCH_HOLDOUT_SPLIT names the holdout split).
                Do the same in your own metrics script: what you print into a run's files is what people open.
Replace score_item() with the task's scorer."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchlib_classify as clf
from benchlib_util import md_table

run_dir = Path(os.environ["BENCH_RUN_DIR"])
SHOW_HOLDOUT = os.environ.get("BENCH_SHOW_HOLDOUT") == "1"
HOLDOUT = os.environ.get("BENCH_HOLDOUT_SPLIT") or None  # set by bench.py when bench.json has "holdout_split"


def write(name, text):  # tmp + rename: readers never see a half-written file
    tmp = run_dir / ("." + name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, run_dir / name)


def counts(expected, actual):
    """(tp, fp, fn) of one item."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        tp = sum(1 for k, v in actual.items() if k in expected and expected[k] == v)
        return tp, len(actual) - tp, len(expected) - tp
    if actual is None:
        return 0, 0, 1
    return (1, 0, 0) if expected == actual else (0, 1, 1)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else (1.0 if not fn else 0.0)
    r = tp / (tp + fn) if tp + fn else 1.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def score_item(expected, actual):
    tp, fp, fn = counts(expected, actual)
    return {"score": prf(tp, fp, fn)[2], "pass": expected == actual, "counts": (tp, fp, fn)}


result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
outputs = result["outputs"]
gold = {}
for path in sorted(Path(os.environ["BENCH_FIXTURE_GOLD"]).glob("*.json")):
    g = json.loads(path.read_text(encoding="utf-8"))
    gold[g["id"]] = g["expected"]
splits = json.loads(Path(os.environ["BENCH_FIXTURE_SPLITS"]).read_text(encoding="utf-8"))
split_of = {iid: name for name, ids in splits.items() for iid in ids}
classification = bool(gold) and all(isinstance(v, str) for v in gold.values())  # plain-string gold = class labels

items, item_counts, mismatches, ids_by_split = {}, {}, {}, {}
for iid, expected in gold.items():
    actual = outputs.get(iid)
    s = score_item(expected, actual)
    item_counts[iid] = s["counts"]
    items[iid] = {"split": split_of.get(iid), "score": s["score"], "pass": s["pass"],
                  "detail": {"expected": expected, "actual": actual}}
    ids_by_split.setdefault(split_of.get(iid), []).append(iid)
    if isinstance(expected, dict) and isinstance(actual, dict):
        mismatches[iid] = [(k, expected.get(k), actual.get(k)) for k in sorted(set(expected) | set(actual)) if expected.get(k) != actual.get(k)]
    elif not s["pass"]:
        mismatches[iid] = [("(whole item)", expected, actual)]


def aggregate(ids):
    """{precision, recall, f1, exact_rate[, accuracy, macro_f1]} over these item ids."""
    exact = sum(items[i]["pass"] for i in ids) / max(len(ids), 1)
    if classification:
        s = clf.summarize([(gold[i], outputs.get(i)) for i in ids])
        return {"precision": s["macro_precision"], "recall": s["macro_recall"], "f1": s["macro_f1"], "exact_rate": exact,
                "accuracy": s["accuracy"], "macro_f1": s["macro_f1"]}
    p, r, f1 = prf(*(sum(item_counts[i][n] for i in ids) for n in range(3)))
    return {"precision": p, "recall": r, "f1": f1, "exact_rate": exact}


named = sorted(n for n in ids_by_split if n)
metrics = {"latency_ms": result.get("latency_ms")}
for name in named:
    metrics.update({"%s.%s" % (name, k): v for k, v in aggregate(ids_by_split[name]).items()})
if not named:  # no splits: nothing to protect, plain metrics
    metrics.update(aggregate(list(items)))
elif SHOW_HOLDOUT:  # `bench.py metrics RUN --show-holdout`: the final check over every item
    metrics.update({"all.%s" % k: v for k, v in aggregate(list(items)).items()})
write("metrics.json", json.dumps(metrics))
write("items.json", json.dumps({"schema_version": 1, "items": items}))

# ---- human reports: the holdout split is left out unless BENCH_SHOW_HOLDOUT=1
left_out = bool(HOLDOUT) and not SHOW_HOLDOUT and HOLDOUT in ids_by_split
shown_ids = [i for i in items if SHOW_HOLDOUT or not HOLDOUT or items[i]["split"] != HOLDOUT]
groups = [(n, ids_by_split[n]) for n in named if SHOW_HOLDOUT or n != HOLDOUT] if named else [("all", list(items))]
if named and SHOW_HOLDOUT:
    groups.append(("all", list(items)))
lines = ["# Metrics", ""]
if left_out:
    lines += ["Holdout split `%s` is left out of this report (final check: `bench.py metrics RUN --show-holdout`)." % HOLDOUT, ""]
lines.append(md_table(["split", "n", "precision", "recall", "f1", "exact_rate"],
                      [[n, len(ids)] + [aggregate(ids)[k] for k in ("precision", "recall", "f1", "exact_rate")] for n, ids in groups]))
if classification:
    for name, ids in groups:
        pairs = [(gold[i], outputs.get(i)) for i in ids]
        lines += ["", "## %s: per class" % name, "", clf.per_class_markdown(clf.summarize(pairs)),
                  "", "Confusion matrix (rows gold, columns predicted):", "", clf.confusion_markdown(pairs)]
write("metrics.md", "\n".join(lines) + "\n")
lines = ["# Per-item diff", "", "Failing items first.%s" % (" Holdout items are left out." if left_out else ""), ""]
for iid in sorted(shown_ids, key=lambda i: (items[i]["pass"], i)):
    it = items[iid]
    lines.append("## %s (%s) score %.3f %s" % (iid, it["split"], it["score"], "ok" if it["pass"] else "FAIL"))
    lines += ["- %s: expected %r, got %r" % m for m in mismatches.get(iid, [])] + [""]
write("diff.md", "\n".join(lines))
'''

ITEMS_CONFIG = {
    "suite": "starter-items", "history_root": "history",
    "fixtures": {"items": "fixtures/items", "gold": "fixtures/gold", "splits": "fixtures/splits.json"},
    "driver": ["python3", "bench_stub_driver.py"], "metrics": ["python3", "bench_stub_metrics.py"],
    "defaults": {"extract_date": True, "lowercase_customer": False}, "timeout_seconds": 600, "retries": 0,
    "provenance": ["bench_stub_driver.py", "benchlib_classify.py"],
    "index_columns": {"config": ["extract_date", "lowercase_customer"],
                      "metrics": ["train.precision", "train.recall", "train.f1", "train.exact_rate", "latency_ms"]},
    "primary_metric": {"name": "f1", "goal": "max"},
    "criteria": {"f1": ">=0.8", "exact_rate": ">=0.5"},
    "timing_keys": ["latency_ms"],
    "holdout_split": "holdout",
    "rank_split": "train",
    "public_metrics": [],
}


def template_files(name):
    """{relative path: text} of one template (without the copied toolkit scripts)."""
    if name == "text":
        return {
            "bench.json": json.dumps(TEXT_CONFIG, indent=2) + "\n",
            "bench_stub_driver.py": TEXT_DRIVER, "bench_stub_metrics.py": TEXT_METRICS,
            "fixtures/input.txt": "the quick brown fox jumps over the lazy dog\n",
            "fixtures/reference.txt": "the quick brown fox jumped over the lazy dog today\n",
        }
    files = {"bench.json": json.dumps(ITEMS_CONFIG, indent=2) + "\n",
             "bench_stub_driver.py": ITEMS_DRIVER, "bench_stub_metrics.py": ITEMS_METRICS}
    splits = {}
    for iid, split, text, gold in ITEM_ROWS:
        files["fixtures/items/%s.json" % iid] = json.dumps({"id": iid, "input": text}, indent=2) + "\n"
        files["fixtures/gold/%s.json" % iid] = json.dumps({"id": iid, "expected": gold}, indent=2) + "\n"
        splits.setdefault(split, []).append(iid)
    files["fixtures/splits.json"] = json.dumps(splits, indent=2) + "\n"
    return files


def next_steps(name, bench, config):
    """Copy-paste commands for the printed hint; `bench` / `config` are already shell-quoted paths."""
    base = "python3 %s" % bench
    steps = ["%s run --label baseline --config %s" % (base, config)]
    if name == "items":
        steps += ["%s run --label no-date --set extract_date=false --config %s" % (base, config),
                  "%s compare baseline no-date --config %s" % (base, config),
                  "%s items baseline no-date --config %s" % (base, config),
                  "%s sweep --explain" % base]
    else:
        steps += ["%s run --label dropped --set drop_every=3 --config %s" % (base, config),
                  "%s compare baseline dropped --config %s" % (base, config)]
    return steps
