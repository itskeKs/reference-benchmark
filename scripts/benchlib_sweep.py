"""Sweep specs (parsing, schema help) and sweep summaries (ranking, repeats, holdout guard)."""
from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

from benchlib_config import LABEL_RE, SCHEMA_VERSION
from benchlib_runs import (Guard, build_comparison, comparison_table, config_diff_table, ensure_metric_column, load_run,
                           ranking_spec, resolve_columns, short_label, tie_lines, visible_columns)
from benchlib_stats import aggregate_repeats
from benchlib_util import (UsageError, allocate_dir, did_you_mean, ensure_dir, iso_now, is_num, md_table, say, stamp,
                           write_atomic, write_json)

SPEC_KEYS = ("name", "base", "configs", "matrix", "baseline", "repeats", "repeat_baseline", "cooldown_seconds", "allow_new_keys")

SPEC_HELP = """\
Sweep spec (JSON object; bench.py sweep --spec FILE). Every config is `base` + its own overrides, run
sequentially, one run per config (or `repeats` runs each). Labels become <name>__<config>.

  name              required; letters, digits, . _ - (must start alphanumeric); prefix of every run label
  base              object, optional: overrides applied to every config (keys must exist in bench.json
                    "defaults", unless allow_new_keys)
  configs           list of {"name": "<config name>", "<key>": <value>, ...}: one run each; the other
                    keys override `base`. Names follow the label rule and must be unique
  matrix            object {"<key>": [v1, v2, ...], ...}: cartesian product, one config per combination,
                    named like k1-1.2_b-0.75. Can be combined with `configs`
  baseline          config name used for deltas and for repeat_baseline (default: the first config)
  repeats           N >= 1 (default 1): run every config N times (labels ...-r1..-rN, one `group` per config);
                    the summary then ranks by mean and shows mean±sd for timing keys and flags
                    quality metrics that vary between repeats
  repeat_baseline   true | false (default false): run the baseline once more at the end (drift check)
  cooldown_seconds  seconds to sleep between runs (default 0)
  allow_new_keys    true | false (default false): allow keys that are not in bench.json "defaults"

At least one of `configs` / `matrix` is required. Unknown spec keys are an error. The whole spec is
validated before anything runs.

Example:
  {
    "name": "bm25",
    "base": {"top_k": 10},
    "baseline": "default",
    "configs": [{"name": "default"}, {"name": "wide", "top_k": 50}],
    "matrix": {"k1": [0.9, 1.2], "b": [0.5, 0.75]},
    "repeats": 3,
    "cooldown_seconds": 5
  }
  -> runs bm25__default, bm25__wide, bm25__k1-0.9_b-0.5, ... each three times (-r1..-r3)

Summary (history_root/sweeps/<name>_<stamp>/summary.md + summary.json): runs ranked by bench.json
"primary_metric" (rank column, best marked, ties noted), a `pass` column when "criteria" are set, labels
shown without the <name>__ prefix (--wide shows run ids too). With bench.json "holdout_split" (or "rank_split"),
ranking uses `<rank_split>.<primary metric>` (default `train.`) when present, holdout metric columns and unsplit
metrics that may include holdout items are hidden. The console line "best by <metric>: <config>" names ties
("tied with N others: ..."). Configs with identical results are flagged as "possibly inert parameter(s)" (the config
keys in which they differ) in summary.md / summary.json.
"""


def matrix_name(pairs):
    raw = "_".join("%s-%s" % (k, v if isinstance(v, str) else json.dumps(v)) for k, v in pairs)
    return re.sub(r"[^A-Za-z0-9._-]+", "-", raw).lstrip("._-")


def parse_sweep_spec(path):
    """Validate the whole spec up front -> {name, base, entries [(config_name, overrides)], ...}."""
    try:
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise UsageError("cannot read sweep spec: %s" % exc)
    except ValueError as exc:
        raise UsageError("invalid JSON in sweep spec: %s" % exc)
    if not isinstance(spec, dict):
        raise UsageError("sweep spec must be a JSON object (see: bench.py sweep --explain)")
    errors = ["unknown key %r%s" % (k, did_you_mean(k, SPEC_KEYS)) for k in sorted(set(spec) - set(SPEC_KEYS))]
    name, base = spec.get("name"), spec.get("base") or {}
    if not (isinstance(name, str) and LABEL_RE.match(name)):
        errors.append("'name' must match %s" % LABEL_RE.pattern)
    if not isinstance(base, dict):
        errors.append("'base' must be an object")
        base = {}
    entries, configs = [], spec.get("configs") or []
    if not isinstance(configs, list):
        errors.append("'configs' must be a list")
        configs = []
    for i, c in enumerate(configs):
        if isinstance(c, dict) and isinstance(c.get("name"), str):
            entries.append((c["name"], {k: v for k, v in c.items() if k != "name"}))
        else:
            errors.append("configs[%d] must be an object with a string 'name'" % i)
    matrix = spec.get("matrix")
    if matrix is not None:
        if not isinstance(matrix, dict) or not matrix or not all(isinstance(v, list) and v for v in matrix.values()):
            errors.append("'matrix' must map keys to non-empty lists")
        else:
            for combo in itertools.product(*matrix.values()):
                pairs = list(zip(matrix, combo))
                entries.append((matrix_name(pairs), dict(pairs)))
    names = [n for n, _ in entries]
    errors += ["config name %r must match %s" % (n, LABEL_RE.pattern) for n in names if not LABEL_RE.match(n)]
    errors += ["duplicate config name %r" % n for n in sorted({n for n in names if names.count(n) > 1})]
    if not entries:
        errors.append("no configs (add 'configs' or 'matrix')")
    cooldown = spec.get("cooldown_seconds", 0)
    if not is_num(cooldown) or cooldown < 0:
        errors.append("'cooldown_seconds' must be a number >= 0")
    for key in ("repeat_baseline", "allow_new_keys"):
        if not isinstance(spec.get(key, False), bool):
            errors.append("'%s' must be true or false" % key)
    repeats = spec.get("repeats", 1)
    if not (isinstance(repeats, int) and not isinstance(repeats, bool) and repeats >= 1):
        errors.append("'repeats' must be an integer >= 1")
        repeats = 1
    baseline = spec.get("baseline", names[0] if names else None)
    if names and baseline not in names:
        errors.append("'baseline' %r is not a config name%s" % (baseline, did_you_mean(baseline, names)))
    if errors:
        raise UsageError("invalid sweep spec: " + "; ".join(errors))
    return {"name": name, "base": base, "entries": entries, "cooldown": cooldown, "baseline": baseline, "repeats": repeats,
            "repeat_baseline": bool(spec.get("repeat_baseline", False)), "allow_new_keys": bool(spec.get("allow_new_keys", False))}


def sweep_plan(spec):
    """[(config, label, overrides, repeat index | None)] in execution order (repeat_baseline appended last)."""
    n, plan = spec["repeats"], []
    configs = [(c, "%s__%s" % (spec["name"], c), {**spec["base"], **ov}) for c, ov in spec["entries"]]
    if spec["repeat_baseline"]:
        base = next(c for c in configs if c[0] == spec["baseline"])
        configs.append((base[0] + "__repeat", base[1] + "__repeat", base[2]))
    for config, label, overrides in configs:
        for k in range(1, n + 1):
            plan.append((config, label if n == 1 else "%s-r%d" % (label, k), overrides, k if n > 1 else None))
    return plan


# ---------------------------------------------------------------- summary

def _config_of(label, prefix):
    return re.sub(r"-r\d+$", "", short_label(label, prefix))


def inert_candidates(entries, guard, timing_keys):
    """[(config name, run row | repeat aggregate)] -> [{"configs": [...], "keys": [...]}]: configs with identical
    results (every visible non-timing metric equal) that differ in these config keys, which therefore may not matter.
    Hidden (holdout) metrics are left out of the comparison so that this never reveals them."""
    sigs = {}
    for name, row in entries:
        if row["status"] != "success":
            continue
        sig = {k: v for k, v in row["metrics"].items() if k not in timing_keys and not guard.hidden(k)}
        if any(is_num(v) for v in sig.values()):
            sigs.setdefault(json.dumps(sig, sort_keys=True), []).append((name, row))
    out = []
    for members in sigs.values():
        keys = sorted(k for k in {k for _, r in members for k in r["config"]}
                      if len({json.dumps(r["config"].get(k), sort_keys=True) for _, r in members}) > 1)
        if len(members) > 1 and keys:
            out.append({"configs": [n for n, _ in members], "keys": keys})
    return out


def inert_lines(candidates):
    return ["NOTE: possibly inert parameter(s): %s (configs %s have identical results but differ in %s)" % (
        ", ".join("`%s`" % k for k in c["keys"]), ", ".join(c["configs"]), "it" if len(c["keys"]) == 1 else "them")
        for c in candidates]


def best_line(rows, ranking_key, prefix):
    """"best by <metric>: <config>[ (tied with N others: a, b)]" for the console, or None without a ranked row."""
    best = next((r for r in rows if r["rank"] == 1), None)
    if not best:
        return None
    tied = [short_label(l, prefix) for l in best["tied_with"]]
    shown = ", ".join(tied[:8]) + (", ..." if len(tied) > 8 else "")
    return "best by %s: %s%s" % (ranking_key, short_label(best["label"], prefix),
                                " (tied with %d other%s: %s)" % (len(tied), "" if len(tied) == 1 else "s", shown) if tied else "")


def build_sweep_summary(cfg, spec, include, results, started):
    root, prefix = cfg["history_root"], spec["name"] + "__"
    rows = [load_run(root / "runs" / rid) for rid in include + [r["run_id"] for r in results]]
    by_id = {r["run_id"]: r for r in rows}
    guard = Guard(cfg, rows)
    cols = resolve_columns(cfg, rows)
    sort, notes = ranking_spec(cfg, rows)
    notes = notes + guard.hidden_note(rows)
    guard.announce(rows)
    cols = visible_columns(ensure_metric_column(cols, sort["key"]) if sort else cols, guard)
    base_cfg_run = next((r["run_id"] for r in results if r["config"] == spec["baseline"]), None)
    baseline_id = include[0] if include else base_cfg_run
    # with repeats the per-run table stays in run order; the ranking happens per config (mean over the repeats)
    comp = build_comparison(rows, cols, baseline_id=baseline_id, sort=sort if spec["repeats"] == 1 else None, notes=notes)
    summary = {"schema_version": SCHEMA_VERSION, "sweep": spec["name"], "suite": cfg["suite"], "started_at": started,
               "finished_at": iso_now(), "include_runs": include, "baseline_config": spec["baseline"],
               "repeats": spec["repeats"], "ranking": sort, "results": results, "comparison": comp}
    inert_entries = [(r["config"], by_id[r["run_id"]]) for r in results]
    groups = None
    if spec["repeats"] > 1:
        pseudo, cells, varying = [], {}, {}
        for config in dict.fromkeys(r["config"] for r in results):
            members = [by_id[r["run_id"]] for r in results if r["config"] == config]
            row, cell_map, vary = aggregate_repeats(prefix + config, members, cfg["timing_keys"])
            pseudo.append(row)
            cells[row["run_id"]] = cell_map
            if vary:
                varying[config] = vary
        head = [by_id[i] for i in include]
        groups = build_comparison(head + pseudo, cols, baseline_id=include[0] if include else prefix + spec["baseline"],
                                  sort=sort, notes=notes)
        metric_cols = [c for c in cols if c["kind"] == "metrics"]
        for row in groups["rows"]:
            for c in metric_cols:
                if row["run_id"] in cells and c["key"] in cells[row["run_id"]]:
                    row["values"][c["header"]] = cells[row["run_id"]][c["key"]]
        flagged = ["WARNING: quality metrics vary between repeats of `%s`: %s. Single runs are not comparable; read mean±sd." % (
            c, ", ".join(k)) for c, k in varying.items()]
        summary["configs"], summary["non_deterministic"] = groups, varying
        comp["warnings"] = comp["warnings"] + flagged
        groups["warnings"] = groups["warnings"] + flagged
        inert_entries = [(row["label"][len(prefix):], row) for row in pseudo]
    summary["possibly_inert"] = inert_candidates(inert_entries, guard, cfg["timing_keys"])
    summary["hidden_metrics"] = guard.hidden_unsplit(k for r in rows for k in r["metrics"])
    return summary, comp, groups


def render_sweep_md(cfg, spec, summary, comp, groups, wide=False):
    prefix = spec["name"] + "__"
    results = summary["results"]
    intro = ["# Sweep %s" % spec["name"], ""]
    ranking = summary["ranking"]
    if ranking:
        intro.append("Ranked by `%s` (%s first%s); baseline for deltas: `%s`." % (
            ranking["key"], "highest" if ranking["descending"] else "lowest",
            ", mean over each config's repeats" if groups is not None else "", (groups or comp)["baseline"]))
    else:
        intro.append("No `primary_metric` in bench.json: rows are in run order. Baseline for deltas: `%s`." % comp["baseline"])
    intro.append("Labels are shown without the `%s` prefix%s." % (prefix, "" if wide else " and run ids are omitted (--wide shows them)"))
    if cfg["holdout_split"]:
        intro.append("Holdout split `%s`: its metric columns are hidden and not used for ranking. Look at it only for the "
                     "final chosen config (`bench.py compare <run> --show-holdout`), never to pick between configs." % cfg["holdout_split"])
    sections = ["\n".join(intro)]
    if comp["warnings"]:
        sections.append("\n".join(comp["warnings"]))
    if summary.get("possibly_inert"):
        sections.append("\n".join(inert_lines(summary["possibly_inert"])))

    def table_section(title, table, ties):
        return "\n\n".join(["## " + title, md_table(*table)] + (["\n".join(ties)] if ties else []))

    if groups is not None:
        title = "Configs (%d repeats each: mean, `mean±sd` where values vary, `(!)` = quality metric varies)" % spec["repeats"]
        sections.append(table_section(title, comparison_table(groups, wide=wide, strip_prefix=prefix, attempt_header="n"), tie_lines(groups, prefix)))
    sections.append(table_section("Runs", comparison_table(comp, wide=wide, strip_prefix=prefix), tie_lines(comp, prefix)))
    sections.append("## Config differences\n\n" + config_diff_table(groups or comp, prefix))
    rhead = ["config"] + (["repeat"] if spec["repeats"] > 1 else []) + ["status", "error_code"] + (["run_id"] if wide else [])
    rbody = [[r["config"]] + ([r.get("repeat")] if spec["repeats"] > 1 else []) + [r["status"], r["error_code"]] + ([r["run_id"]] if wide else [])
             for r in results]
    sections.append("## Results\n\n" + md_table(rhead, rbody))
    failed = list(dict.fromkeys(r["config"] for r in results if r["status"] != "success"))
    sections.append("Failed configs: %s" % (", ".join(failed) or "none"))
    return "\n\n".join(sections) + "\n"


def write_sweep_summary(cfg, spec, include, results, started, *, wide=False):
    root = cfg["history_root"]
    sweeps = ensure_dir(ensure_dir(root) / "sweeps")
    name = allocate_dir(sweeps, "%s_%s" % (spec["name"], stamp()))
    summary, comp, groups = build_sweep_summary(cfg, spec, include, results, started)
    write_json(sweeps / name / "summary.json", summary)
    write_atomic(sweeps / name / "summary.md", render_sweep_md(cfg, spec, summary, comp, groups, wide))
    for line in comp["warnings"] + inert_lines(summary["possibly_inert"]):
        say(line)
    best = best_line((groups or comp)["rows"], summary["ranking"]["key"], spec["name"] + "__") if summary["ranking"] else None
    if best:
        say(best)
    say("sweep summary: %s" % (sweeps / name))
    return sweeps / name
