"""Reading history: run loading, run / group id resolution, the holdout guard (hidden metrics), criteria, ranking, code
and fixture warnings, index and comparison tables."""
from __future__ import annotations

import fnmatch
import json
import re

from benchlib_config import CRITERION_OPS, TUNING_SPLIT, code_identity, fingerprint, tuning_split
from benchlib_util import (UsageError, close_matches, did_you_mean, ensure_dir, is_num, md_table, read_json,
                           warn_once, write_atomic, write_json)

CORE = ("run_id", "label", "status", "attempt")  # base columns; a config/metric key with such a name is prefixed
TABLE_CORE = CORE + ("error_code", "fixtures", "pass")
CORE_NAMES = set(TABLE_CORE) | {"rank", "duration_ms", "started_at", "group"}


# ---------------------------------------------------------------- loading

def run_label(run_dir, run=None):
    run = run if run is not None else read_json(run_dir / "run.json")
    run = run if isinstance(run, dict) else {}
    return run.get("label") or run_dir.name.split("_", 1)[-1]


def fixtures_display(run):
    """Fingerprint of the fixtures the run was produced against; `a>b` when it was re-scored (--force) against others."""
    base = run.get("fixture_fingerprint") or fingerprint(run.get("fixtures"))
    if run.get("fixture_drift"):
        new = fingerprint(run.get("rescored_fixtures"))
        if new and new != base:
            return "%s>%s" % (base or "?", new)
    return base


def load_run(run_dir):
    """Tolerant loader: a dir may lack run.json and/or metrics.json."""
    run, metrics = read_json(run_dir / "run.json"), read_json(run_dir / "metrics.json")
    run = run if isinstance(run, dict) else {}
    config = run["config"] if "config" in run else read_json(run_dir / "config.json")
    cr = run.get("criteria_result")
    return {
        "run_id": run_dir.name, "label": run_label(run_dir, run), "status": run.get("status") or "unknown",
        "attempt": run.get("attempt"), "error_code": run.get("error_code"), "started_at": run.get("started_at"),
        "duration_ms": run.get("duration_ms"), "group": run.get("group"), "fixtures": fixtures_display(run),
        "pass": cr.get("pass") if isinstance(cr, dict) else None, "has_criteria": isinstance(cr, dict),
        "config": config if isinstance(config, dict) else {}, "metrics": metrics if isinstance(metrics, dict) else {},
        "code": code_identity(run),
    }


def list_run_ids(runs):
    return sorted(p.name for p in runs.iterdir() if p.is_dir() and not p.name.startswith(".")) if runs.is_dir() else []


def load_runs(root):
    runs = root / "runs"
    return [load_run(runs / rid) for rid in list_run_ids(runs)]


# ---------------------------------------------------------------- run-id resolution

def _candidates(ids, labels):
    shown = ["%s (label %s)" % (i, labels[i]) for i in ids[:10]]
    return "\n  ".join(shown) + ("\n  ... and %d more" % (len(ids) - 10) if len(ids) > 10 else "")


def resolve_ids(cfg, tokens, latest=False, what="run id"):
    """Run references -> exact run ids. A token is an exact id, else a unique id prefix, else a unique label
    (the newest one with latest=True); anything else exits 2 listing the candidates."""
    runs = cfg["history_root"] / "runs"
    ids = list_run_ids(runs)
    labels = {i: run_label(runs / i) for i in ids}
    out = []
    for token in tokens:
        if token in labels:
            out.append(token)
            continue
        prefix = [i for i in ids if i.startswith(token)]
        if len(prefix) == 1:
            out.append(prefix[0])
            continue
        by_label = [i for i in ids if labels[i] == token]
        if len(by_label) == 1:
            out.append(by_label[0])
        elif len(by_label) > 1 and latest:
            out.append(by_label[-1])
        elif by_label or prefix:
            found = by_label or prefix
            hint = "pass --latest to take the newest label match, or" if by_label else "use"
            raise UsageError("ambiguous %s %r matches %d runs; %s a longer prefix / the full id:\n  %s"
                             % (what, token, len(found), hint, _candidates(found, labels)))
        else:
            near = close_matches(token, list(labels.values()))
            raise UsageError("unknown %s %r%s" % (what, token, " (similar labels: %s)" % ", ".join(near) if near else ""))
    return out


def sweep_run_ids(cfg, name):
    """Run ids of the newest sweep summary named `name` (sweep name or the summary directory name)."""
    sweeps = cfg["history_root"] / "sweeps"
    found = []
    for d in sorted(p for p in sweeps.iterdir() if p.is_dir()) if sweeps.is_dir() else []:
        summary = read_json(d / "summary.json")
        if isinstance(summary, dict) and name in (d.name, summary.get("sweep")):
            found.append(summary)
    if not found:
        raise UsageError("no sweep named %r in %s" % (name, sweeps))
    return [r["run_id"] for r in found[-1].get("results", []) if isinstance(r, dict) and r.get("run_id")]


_STAMP_RE = re.compile(r"^\d{8}T\d{9}Z_")


def resolve_group(known, token, latest=False):
    """A group reference -> group id: an exact id, else a unique id prefix, else the base label (the part after the
    timestamp, e.g. `det` for `run --label det --repeat 2`; the newest group with latest=True), like run ids."""
    if token in known:
        return token
    prefix = [g for g in known if g.startswith(token)]
    if len(prefix) == 1:
        return prefix[0]
    by_label = [g for g in known if _STAMP_RE.sub("", g) == token]
    if len(by_label) == 1:
        return by_label[0]
    if len(by_label) > 1 and latest:
        return by_label[-1]
    if by_label or prefix:
        found = by_label or prefix
        hint = "pass --latest to take the newest label match, or use" if by_label else "use"
        raise UsageError("ambiguous group %r matches %d groups; %s a longer prefix / the full group id:\n  %s" % (
            token, len(found), hint, "\n  ".join(found[:10])))
    raise UsageError("unknown group %r%s" % (token, did_you_mean(token, [_STAMP_RE.sub("", g) for g in known] + known)))


def select_runs(cfg, tokens=(), *, group=None, label_glob=None, all_runs=False, sweep=None, latest=False):
    """Union of explicit run references, --all, --label-glob, --group and --sweep -> run ids in history order."""
    runs = cfg["history_root"] / "runs"
    ids = list_run_ids(runs)
    chosen = set(resolve_ids(cfg, list(tokens), latest))
    if all_runs:
        chosen |= set(ids)
    if label_glob:
        hit = {i for i in ids if fnmatch.fnmatchcase(run_label(runs / i), label_glob)}
        if not hit:
            raise UsageError("no run label matches %r" % label_glob)
        chosen |= hit
    if group:
        groups = {i: (read_json(runs / i / "run.json") or {}).get("group") for i in ids}
        match = resolve_group(sorted({g for g in groups.values() if g}), group, latest)
        chosen |= {i for i, g in groups.items() if g == match}
    if sweep:
        members = sweep_run_ids(cfg, sweep)
        missing = [i for i in members if i not in ids]
        if missing:
            raise UsageError("sweep %r references runs missing from history: %s" % (sweep, ", ".join(missing)))
        chosen |= set(members)
    return [i for i in ids if i in chosen]


# ---------------------------------------------------------------- holdout guard

RESERVED_UNSPLIT = {"all"}  # `all.*` = aggregates over every item (holdout included): never a split name


class Guard:
    """Which metric keys the holdout guard keeps out of index.md / compare / sweep summaries / stats.

    Active only with bench.json "holdout_split". Hidden: every `<holdout_split>.*` key, and every metric key that is
    not namespaced to a non-holdout split, not a timing key and not allowlisted in "public_metrics" (exact names or
    glob patterns): an unprefixed aggregate such as `accuracy` is computed over all items, holdout included.
    Non-holdout splits are `train`, the rank_split plus every split bench.py aggregated from items.json (the
    `<split>.n` / `.score_mean` / `.pass_rate` triple); other split-like prefixes need a public_metrics pattern such
    as "dev.*". index.json is raw data and keeps every key."""

    def __init__(self, cfg, rows):
        self.cfg = cfg
        self.holdout = cfg["holdout_split"]
        self.active = bool(self.holdout)
        keys = {k for r in rows for k in r["metrics"]}
        splits = {k[:-2] for k in keys if k.endswith(".n") and k[:-2] + ".score_mean" in keys and k[:-2] + ".pass_rate" in keys}
        self.splits = (splits | {cfg["rank_split"], TUNING_SPLIT}) - RESERVED_UNSPLIT - {self.holdout}

    def exempt(self, key):
        """Timing keys and public_metrics are never hidden (they are not holdout-contaminated quality numbers)."""
        return key in self.cfg["timing_keys"] or any(fnmatch.fnmatchcase(key, p) for p in self.cfg["public_metrics"])

    def in_split(self, key):
        """True when key is `<split>.<metric>` for a known split (the holdout split included)."""
        return any(str(key).startswith(s + ".") for s in self.splits | ({self.holdout} if self.holdout else set()))

    def reason(self, key):
        """None (shown) | "holdout" (`<holdout_split>.*`) | "unsplit" (not namespaced to a non-holdout split)."""
        key = str(key)
        if not self.active:
            return None
        if key.startswith(self.holdout + "."):
            return "holdout"
        if self.exempt(key) or any(key.startswith(s + ".") for s in self.splits):
            return None
        return "unsplit"

    def hidden(self, key):
        return self.reason(key) is not None

    def hidden_unsplit(self, keys):
        return sorted(k for k in set(keys) if self.reason(k) == "unsplit")

    def hidden_note(self, rows):
        """Report note naming the hidden unsplit keys of these runs ([] when there are none)."""
        keys = self.hidden_unsplit(k for r in rows for k in r["metrics"])
        return ["NOTE: unsplit metric(s) %s may include holdout items and are hidden (`--show-holdout` shows them for the "
                "final check; `public_metrics` in bench.json allowlists safe ones)." % ", ".join("`%s`" % k for k in keys)] if keys else []

    def announce(self, rows):
        """One-time stderr warning naming the hidden unsplit keys, so the user knows why they disappeared."""
        keys = self.hidden_unsplit(k for r in rows for k in r["metrics"])
        if keys:
            warn_once("holdout_split %r is set: metric(s) %s are not prefixed with a non-holdout split, so they may include "
                      "holdout items. They are hidden from index.md, compare, sweep summaries and stats (index.json keeps "
                      "them). Emit split-prefixed metrics (%s.<metric>), allowlist harmless keys in bench.json "
                      "\"public_metrics\", or pass --show-holdout for the final check." % (
                          self.holdout, ", ".join(keys), self.cfg["rank_split"]))


# ---------------------------------------------------------------- criteria

def evaluate_criteria(criteria, metrics, cfg=None):
    """-> None without criteria, else {"pass": True|False|None, "details": {metric: {...}}, "warnings"?: [...]}.

    pass is False if any criterion fails, None if none fails but a metric is missing / not numeric.
    With a cfg that has a tuning split (holdout_split or rank_split), an unprefixed criterion `f1` is evaluated against
    `<rank_split>.f1` when that metric exists (details[...]["metric"] names the key used); otherwise it falls back to
    the plain key and records a warning (the plain value may include holdout items)."""
    if not criteria:
        return None
    metrics = metrics if isinstance(metrics, dict) else {}
    split = tuning_split(cfg) if cfg else None
    guard = Guard(cfg, [{"metrics": metrics}]) if split else None
    details, verdicts, warnings = {}, [], []
    for c in criteria:
        key, note = c["name"], None
        if split and not guard.in_split(key):
            tuned = "%s.%s" % (split, key)
            if is_num(metrics.get(tuned)):
                key = tuned
            elif key in metrics and not guard.exempt(key):
                note = "criterion `%s`: no `%s` metric, evaluated against the unsplit `%s`%s" % (
                    key, tuned, key, ", which may include holdout items" if cfg["holdout_split"] else "")
                warnings.append(note)
        value = metrics.get(key)
        ok = CRITERION_OPS[c["op"]](value, c["threshold"]) if is_num(value) else None
        details[c["name"]] = {"criterion": c["text"], "value": value if is_num(value) else None, "pass": ok}
        if key != c["name"]:
            details[c["name"]]["metric"] = key
        if note:
            details[c["name"]]["warning"] = note
        verdicts.append(ok)
    out = {"pass": False if False in verdicts else None if None in verdicts else True, "details": details}
    if warnings:
        out["warnings"] = warnings
    return out


# ---------------------------------------------------------------- columns

def relabel(cols):
    """Headers: the bare key, or kind.key when a core column or the other kind has the same key (or the user asked
    for the prefixed form)."""
    for c in cols:
        clash = c["key"] in TABLE_CORE or c.get("explicit") or any(o["key"] == c["key"] and o["kind"] != c["kind"] for o in cols)
        c["header"] = "%s.%s" % (c["kind"], c["key"]) if clash else c["key"]
    return cols


def make_columns(config_keys, metric_keys, explicit=()):
    return relabel([{"key": str(k), "kind": kind, "explicit": (kind, str(k)) in explicit}
                    for kind, keys in (("config", config_keys), ("metrics", metric_keys)) for k in keys])


def ensure_metric_column(cols, key):
    """Append a metrics column for `key` unless one exists (a ranking key must be visible)."""
    if any(c["kind"] == "metrics" and c["key"] == key for c in cols):
        return cols
    return relabel(cols + [{"key": key, "kind": "metrics", "explicit": False}])


def resolve_columns(cfg, rows):
    ic = cfg["index_columns"]
    config_keys, metric_keys = ic.get("config"), ic.get("metrics")
    if config_keys is None:
        config_keys = sorted({k for r in rows for k in r["config"]})
    if metric_keys is None:
        metric_keys = sorted({k for r in rows for k in r["metrics"]})
    return make_columns(config_keys, metric_keys)


def visible_columns(cols, guard, show_holdout=False):
    if show_holdout:
        return cols
    return [c for c in cols if not (c["kind"] == "metrics" and guard.hidden(c["key"]))]


def with_hidden_columns(cols, rows, guard):
    """--show-holdout: also show every hidden metric (`<holdout>.*` and unsplit keys) the runs have, even if
    index_columns does not list it."""
    for key in sorted({k for r in rows for k in r["metrics"] if guard.hidden(k)}):
        cols = ensure_metric_column(cols, key)
    return cols


def parse_columns(spec, rows, guard, show_holdout=False):
    """compare --columns: `x` (metric, config, or both when both exist), `config.x`, `metrics.x`; unknown -> exit 2."""
    config_all = sorted({k for r in rows for k in r["config"]})
    metric_all = sorted({k for r in rows for k in r["metrics"]})
    config_keys, metric_keys, unknown, explicit = [], [], [], set()
    for name in (k for k in spec.split(",") if k):
        found = False
        if name in metric_all:
            metric_keys.append(name)
            found = True
        if name in config_all:
            config_keys.append(name)
            found = True
        if found or name in CORE_NAMES:
            continue
        if name.startswith("config.") and name[7:] in config_all:
            config_keys.append(name[7:])
            explicit.add(("config", name[7:]))
        elif name.startswith("metrics.") and name[8:] in metric_all:
            metric_keys.append(name[8:])
            explicit.add(("metrics", name[8:]))
        else:
            unknown.append(name)
    if unknown:
        raise UsageError("unknown column(s) %s; available: config %s | metrics %s | always shown %s" % (
            ", ".join("%r%s" % (u, did_you_mean(u, config_all + metric_all)) for u in unknown),
            ", ".join(config_all) or "(none)", ", ".join(metric_all) or "(none)", ", ".join(sorted(CORE_NAMES))))
    hidden = [k for k in metric_keys if guard.hidden(k)]
    if hidden and not show_holdout:
        raise UsageError("column(s) %s are hidden by the holdout guard (holdout split %r, or unsplit metrics that may include "
                         "its items); pass --show-holdout (final check only) or list safe metrics in public_metrics" % (
                             ", ".join(hidden), guard.holdout))
    return make_columns(config_keys, metric_keys, explicit)


# ---------------------------------------------------------------- ranking

def ranking_spec(cfg, rows, sort=None, descending=None, show_holdout=False):
    """What the rows are ranked by -> ({"key", "descending", "name"} | None, [warning lines]).

    Default is the primary metric in its goal direction; `sort` names another metric (descending unless told
    otherwise). With a tuning split (holdout_split or rank_split, default "train"), the default ranking uses
    `<rank_split>.<primary>` when present."""
    pm, notes = cfg["primary_metric"], []
    name = sort or (pm["name"] if pm else None)
    if not name:
        return None, notes
    if descending is None:
        descending = (pm["goal"] == "max") if pm and name == pm["name"] else True
    key, guard = name, Guard(cfg, rows)
    if sort and guard.hidden(sort) and not show_holdout:
        raise UsageError("--sort %s is hidden by the holdout guard (holdout split %r, or an unsplit metric that may include its "
                         "items); pass --show-holdout (final check only) or list it in public_metrics" % (sort, guard.holdout))
    split = tuning_split(cfg)
    if split and not sort and not guard.in_split(name):
        tuned = "%s.%s" % (split, name)
        if any(is_num(r["metrics"].get(tuned)) for r in rows):
            key = tuned
        elif not guard.exempt(name):
            notes.append("WARNING: %s but no `%s` metric exists; ranking uses `%s`%s" % (
                "holdout_split is set" if cfg["holdout_split"] else "rank_split is `%s`" % split, tuned, name,
                ", which may include holdout items (emit split-prefixed metrics to keep tuning off the holdout)."
                if cfg["holdout_split"] else "."))
    return {"key": key, "descending": bool(descending), "name": name}, notes


def rank_rows(rows, key, descending):
    """Stable best-first order (rows without a numeric value last) -> (ordered, {run_id: rank}, {run_id: [tied ids]}).
    Ranks are competition ranks: 1, 1, 3."""
    have = [r for r in rows if is_num(r["metrics"].get(key))]
    have.sort(key=lambda r: r["metrics"][key], reverse=descending)
    ordered = have + [r for r in rows if not is_num(r["metrics"].get(key))]
    ranks, ties = {}, {}
    for pos, r in enumerate(have):
        v = r["metrics"][key]
        first = next(i for i, o in enumerate(have) if o["metrics"][key] == v)
        ranks[r["run_id"]] = first + 1
        ties[r["run_id"]] = [o["run_id"] for o in have if o["metrics"][key] == v and o is not r]
    return ordered, ranks, ties


# ---------------------------------------------------------------- comparison

def fixture_warnings(rows):
    groups = {}
    for r in rows:
        if r.get("fixtures"):
            groups.setdefault(r["fixtures"], []).append(r["label"])
    if len(groups) < 2:
        return []
    detail = "; ".join("%s: %s" % (fp, ", ".join(labels)) for fp, labels in sorted(groups.items()))
    return ["WARNING: compared runs used different fixtures (fingerprint: runs) -> %s. "
            "Differences may come from the data, not the configuration." % detail]


def code_warnings(rows):
    """One WARNING when the compared runs were produced / scored by different code (driver script, metrics script or
    `provenance` paths hashes in run.json); runs without recorded provenance are ignored."""
    parts = []
    for part, title in (("driver", "driver script"), ("metrics", "metrics script"), ("paths", "provenance paths")):
        groups = {}
        for r in rows:
            fp = (r.get("code") or {}).get(part)
            if fp:
                groups.setdefault(fp, []).append(r["label"])
        if len(groups) > 1:
            parts.append("%s (%s)" % (title, "; ".join("%s: %s" % (fp, ", ".join(labels)) for fp, labels in sorted(groups.items()))))
    if not parts:
        return []
    return ["WARNING: compared runs used different code (hash: runs) -> %s. Differences may come from code changes, not the "
            "configuration. A re-score with changed scorer code is recorded as rescored_provenance in run.json." % " | ".join(parts)]


def build_comparison(rows, cols, *, baseline_id=None, sort=None, top=None, notes=()):
    """Rows side by side; numeric metric deltas vs the baseline (default: first listed run); optional ranking."""
    base = next((r for r in rows if r["run_id"] == baseline_id), rows[0])
    delta_cols = [c for c in cols if c["kind"] == "metrics" and any(is_num(r["metrics"].get(c["key"])) for r in rows)]
    ordered, ranks, ties = (rank_rows(rows, sort["key"], sort["descending"]) if sort else (rows, {}, {}))
    by_id = {r["run_id"]: r for r in rows}
    out = []
    for r in ordered[:top] if top else ordered:
        deltas = {}
        for c in delta_cols:
            a, b = base["metrics"].get(c["key"]), r["metrics"].get(c["key"])
            deltas[c["header"]] = b - a if r is not base and is_num(a) and is_num(b) else None
        out.append({**{k: r[k] for k in TABLE_CORE}, "rank": ranks.get(r["run_id"]),
                    "tied_with": [by_id[i]["label"] for i in ties.get(r["run_id"], [])],
                    "values": {c["header"]: r[c["kind"]].get(c["key"]) for c in cols}, "deltas": deltas})
    keys = sorted({k for r in rows for k in r["config"]})
    diffs = [{"key": k, "values": {r["run_id"]: r["config"].get(k) for r in rows}} for k in keys
             if len({json.dumps(r["config"].get(k), sort_keys=True) for r in rows}) > 1]
    return {"baseline": base["run_id"], "columns": [c["header"] for c in cols], "delta_columns": [c["header"] for c in delta_cols],
            "sort": sort, "has_pass": any(r["has_criteria"] for r in rows), "shown": len(out), "total": len(rows),
            "rows": out, "config_differences": diffs, "warnings": fixture_warnings(rows) + code_warnings(rows) + list(notes)}


def common_label_prefix(labels):
    """`<x>__` when every label starts with the same `<x>__` (sweep runs), else None."""
    heads = {l.split("__", 1)[0] for l in labels if "__" in l}
    if len(heads) == 1 and all("__" in l for l in labels):
        return heads.pop() + "__"
    return None


def short_label(label, prefix):
    return label[len(prefix):] if prefix and label.startswith(prefix) and len(label) > len(prefix) else label


def pass_cell(v):
    return None if v is None else "pass" if v else "FAIL"


def rank_cell(row):
    if row["rank"] is None:
        return None
    tags = (["best"] if row["rank"] == 1 else []) + (["tied"] if row["tied_with"] else [])
    return "%d%s" % (row["rank"], " (%s)" % ", ".join(tags) if tags else "")


def tie_lines(comp, strip_prefix=None):
    seen, lines = set(), []
    for r in comp["rows"]:
        if r["tied_with"] and r["rank"] not in seen:
            seen.add(r["rank"])
            lines.append("Rank %d: %s tied with %s" % (r["rank"], short_label(r["label"], strip_prefix),
                                                       ", ".join(short_label(l, strip_prefix) for l in r["tied_with"])))
    return lines


def comparison_table(comp, *, wide=False, strip_prefix=None, attempt_header="attempt"):
    """Header + body of the comparison table; run_id only with wide (or when labels would be ambiguous)."""
    rows = comp["rows"]
    labels = [short_label(r["label"], strip_prefix) for r in rows]
    show_id = wide or len(set(labels)) != len(labels)
    headers = (["rank"] if comp["sort"] else []) + ["label"] + (["run_id"] if show_id else []) + [
        "status", attempt_header, "error_code", "fixtures"] + (["pass"] if comp["has_pass"] else []) + comp["columns"] + [
        "Δ" + h for h in comp["delta_columns"]]
    body = []
    for r, lab in zip(rows, labels):
        deltas = ["–" if r["deltas"][h] is None else "%+.6g" % r["deltas"][h] for h in comp["delta_columns"]]
        body.append(([rank_cell(r)] if comp["sort"] else []) + [lab] + ([r["run_id"]] if show_id else []) + [
            r["status"], r["attempt"], r["error_code"], r["fixtures"]] + ([pass_cell(r["pass"])] if comp["has_pass"] else []) + [
            r["values"][h] for h in comp["columns"]] + deltas)
    return headers, body


def config_diff_table(comp, strip_prefix=None):
    rows = comp["rows"]
    labels = {r["run_id"]: short_label(r["label"], strip_prefix) for r in rows}
    ids = [r["run_id"] for r in rows]
    heads = [labels[i] for i in ids] if len(set(labels.values())) == len(ids) else ids
    diffs = comp["config_differences"]
    if not diffs:
        return "None."
    return md_table(["key"] + heads, [[d["key"]] + [d["values"].get(i) for i in ids] for d in diffs])


def render_comparison_md(comp, title, *, wide=False, strip_prefix=None):
    headers, body = comparison_table(comp, wide=wide, strip_prefix=strip_prefix)
    lines = ["# %s" % title, "", "Deltas are relative to the baseline (the first listed run): `%s`" % comp["baseline"]]
    if comp["sort"]:
        lines.append("Ranked by `%s` (%s first)%s." % (comp["sort"]["key"], "highest" if comp["sort"]["descending"] else "lowest",
                                                         ", top %d of %d shown" % (comp["shown"], comp["total"]) if comp["shown"] < comp["total"] else ""))
    if strip_prefix:
        lines.append("Labels are shown without the `%s` prefix." % strip_prefix)
    lines += [""] + comp["warnings"] + ([""] if comp["warnings"] else [])
    ties = tie_lines(comp, strip_prefix)
    lines += [md_table(headers, body), ""] + ties + ([""] if ties else [])
    lines += ["## Config differences", "", config_diff_table(comp, strip_prefix)]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- index

def write_index(cfg):
    root = ensure_dir(cfg["history_root"])
    rows = load_runs(root)
    guard = Guard(cfg, rows)
    cols = resolve_columns(cfg, rows)
    shown = visible_columns(cols, guard)
    out = []
    for r in rows:
        row = {k: r[k] for k in TABLE_CORE}
        row.update(started_at=r["started_at"], duration_ms=r["duration_ms"], group=r["group"])
        for kind in ("config", "metrics"):
            row[kind] = {c["key"]: r[kind].get(c["key"]) for c in cols if c["kind"] == kind}
        if guard.active:  # index.json is raw data (not filtered); this marks which keys of the run's metrics the views hide
            row["hidden_metrics"] = sorted(k for k in r["metrics"] if guard.hidden(k))
        out.append(row)
    write_json(root / "index.json", out)
    has_pass = any(r["has_criteria"] for r in rows)
    headers = list(TABLE_CORE if has_pass else [c for c in TABLE_CORE if c != "pass"]) + [c["header"] for c in shown]
    body = []
    for r in rows:
        cells = [r[k] if k != "pass" else pass_cell(r["pass"]) for k in TABLE_CORE if k != "pass" or has_pass]
        body.append(cells + [r[c["kind"]].get(c["key"]) for c in shown])
    note = ""
    if len(shown) != len(cols):
        note = ("\nHoldout split `%s`: its metric columns are hidden here (index.json has them, unfiltered, and lists them per run "
                "in `hidden_metrics`; `compare --show-holdout` for the final check).\n" % cfg["holdout_split"])
        note += "".join("\n%s\n" % n for n in guard.hidden_note(rows))
    write_atomic(root / "index.md", "# Index: %s\n\n%s\n%s" % (cfg["suite"], md_table(headers, body), note))
    guard.announce(rows)
    return rows
