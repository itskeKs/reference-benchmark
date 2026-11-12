"""Repeat statistics, determinism check and per-config aggregation of repeated runs."""
from __future__ import annotations

import json
import math

from benchlib_util import UsageError, is_num, md_table

_MISSING = "<missing>"


def metric_stats(values):
    n = len(values)
    mean = math.fsum(values) / n
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else None
    return {"n": n, "mean": mean, "sd": sd, "min": min(values), "max": max(values),
            "cv_pct": 100.0 * sd / abs(mean) if sd is not None and mean else None}


def fmt_mean_sd(st):
    return "%.6g" % st["mean"] if st["sd"] is None else "%.6g±%.3g" % (st["mean"], st["sd"])


def numeric_keys(rows):
    return sorted({k for r in rows for k, v in r["metrics"].items() if is_num(v)})


def compute_stats(rows, timing_keys):
    """Per numeric metric: n, mean, sd (sample, n-1), min, max, cv% over the runs that have it."""
    out = {}
    for key in numeric_keys(rows):
        st = metric_stats([r["metrics"][key] for r in rows if is_num(r["metrics"].get(key))])
        st["timing"] = key in timing_keys
        out[key] = st
    return {"runs": [{"run_id": r["run_id"], "label": r["label"], "status": r["status"]} for r in rows],
            "runs_without_metrics": [r["run_id"] for r in rows if not r["metrics"]], "metrics": out}


def render_stats_md(stats, title="Repeat statistics"):
    body = [[k, s["n"], s["mean"], s["sd"], s["min"], s["max"], s["cv_pct"], "yes" if s["timing"] else ""]
            for k, s in stats["metrics"].items()]
    text = "# %s\n\n%d run(s): %s\n\n" % (title, len(stats["runs"]), ", ".join(r["label"] for r in stats["runs"]))
    if stats["runs_without_metrics"]:
        text += "No metrics (ignored): %s\n\n" % ", ".join(stats["runs_without_metrics"])
    text += md_table(["metric", "n", "mean", "sd", "min", "max", "cv%", "timing"], body)
    return text + "\n\nsd is the sample standard deviation (n-1; empty for n=1); cv% = 100 * sd / |mean|.\n"


def check_determinism(rows, timing_keys):
    """Non-timing metrics must be identical across all runs; missing in some runs counts as different."""
    if len(rows) < 2:
        raise UsageError("determinism needs at least 2 runs")
    empty = [r["run_id"] for r in rows if not r["metrics"]]
    if empty:
        raise UsageError("runs without metrics.json cannot be checked: %s" % ", ".join(empty))
    all_keys = sorted({k for r in rows for k in r["metrics"]})
    keys = [k for k in all_keys if k not in timing_keys]
    differing = {}
    for k in keys:
        values = {r["run_id"]: r["metrics"].get(k, _MISSING) for r in rows}
        if len({json.dumps(v, sort_keys=True) for v in values.values()}) > 1:
            differing[k] = {i: None if v == _MISSING else v for i, v in values.items()}
    return {"identical": not differing, "runs": [r["run_id"] for r in rows], "compared_keys": keys,
            "ignored_timing_keys": [k for k in all_keys if k in timing_keys], "differing": differing}


def render_determinism_md(res):
    head = "%d run(s), %d non-timing metric(s) compared" % (len(res["runs"]), len(res["compared_keys"]))
    if res["ignored_timing_keys"]:
        head += "; timing keys ignored: %s" % ", ".join(res["ignored_timing_keys"])
    if res["identical"]:
        return "DETERMINISTIC: %s, all identical.\n" % head
    body = [[k] + [v.get(i) for i in res["runs"]] for k, v in res["differing"].items()]
    return "NOT DETERMINISTIC: %s; differing: %s\n\n%s\n" % (head, ", ".join(res["differing"]), md_table(["metric"] + res["runs"], body))


# ---------------------------------------------------------------- repeats inside a sweep

def aggregate_repeats(label, rows, timing_keys):
    """Runs of one config (its repeats) -> (pseudo run row for ranking/deltas, display cells, non-deterministic keys).

    The pseudo row's metrics are means over the runs that have the key; cells show `mean±sd` for timing keys and
    for quality keys that vary between repeats (those are also returned as non-deterministic)."""
    ok = [r for r in rows if r["status"] == "success"]
    metrics, cells, varying = {}, {}, []
    for key in sorted({k for r in ok for k in r["metrics"]}):
        vals = [r["metrics"][key] for r in ok if key in r["metrics"]]
        if all(is_num(v) for v in vals):
            st = metric_stats(vals)
            metrics[key] = st["mean"]
            constant = st["min"] == st["max"]
            cells[key] = vals[0] if constant and key not in timing_keys else fmt_mean_sd(st)
        else:
            metrics[key] = vals[0]
            constant = len({json.dumps(v) for v in vals}) == 1
            cells[key] = vals[0] if constant else "varies"
        if not constant and key not in timing_keys:
            varying.append(key)
            cells[key] = "%s (!)" % cells[key]
    verdicts = [r["pass"] for r in rows]
    passed = False if False in verdicts else None if None in verdicts else True
    first = rows[0]
    pseudo = {"run_id": label, "label": label, "status": "success" if len(ok) == len(rows) else "%d/%d ok" % (len(ok), len(rows)),
              "attempt": len(rows), "error_code": next((r["error_code"] for r in rows if r["error_code"]), None),
              "fixtures": first["fixtures"], "pass": passed if any(r["has_criteria"] for r in rows) else None,
              "has_criteria": any(r["has_criteria"] for r in rows), "config": first["config"], "metrics": metrics,
              "started_at": first["started_at"], "duration_ms": None, "group": first["group"]}
    return pseudo, cells, varying
