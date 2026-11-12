#!/usr/bin/env python3
"""Generic reference-benchmark harness (stdlib only, Python >= 3.9).

Wraps any project driver + metrics command; keeps history under history_root/runs/<run_id>/.
Commands: init | run | sweep | metrics (alias: rescore) | index | compare | stats | determinism | items
(`bench.py <command> -h` shows its flags; `bench.py sweep --explain` prints the sweep spec format).
Helper modules (benchlib_*.py), textdiff.py and calibrate.py live next to this file; `init` copies all of them.

bench.json (relative paths resolve against the config file's directory, ~ is expanded; unknown keys are warned about)
  history_root          required: runs/, sweeps/, index.json, index.md live here (outside the repo for sensitive data)
  suite                 name shown in reports (default "bench")
  fixtures              {name: path}: inputs + references (file or directory). name, bytes and sha256 of each are
                        recorded in run.json; $BENCH_FIXTURE_<NAME> points the driver/metrics at them
  driver, metrics       argv (list, or a string that is shlex-split) of the commands that run the system under test
                        and score its result
  defaults              {key: value}: the complete config of a run; --set / sweep overrides must use these keys and
                        the TYPE of the default (bool / int / float / str / list / dict; an int is accepted where a
                        float is expected; a null default accepts anything; a str default takes --set text as is):
                        mismatches exit 2
  allow_new_keys        true | false (default false): allow overrides of keys that are not in defaults (untyped)
  timeout_seconds       per-command timeout (default 1800; 0 = none)
  retries               extra attempts after a killed/busy attempt (default 0)
  guard, guard_quiet_seconds, guard_poll_seconds, guard_max_wait_seconds
                        exclusive-resource guard: command that must exit 0 (continuously for guard_quiet_seconds,
                        polled every guard_poll_seconds, give up after guard_max_wait_seconds -> status busy)
  index_columns         {"config": [keys], "metrics": [keys]}: columns of index.md and compare (absent = all keys)
  provenance            [paths]: files/dirs of the system under test, hashed into run.json "provenance" so setups
                        without git still record code identity (driver and metrics scripts that appear in the
                        commands' argv are hashed automatically; __pycache__, .git, .DS_Store are ignored)
  primary_metric        {"name": "f1", "goal": "max" | "min"}: default ranking of compare and sweeps
  criteria              {"wer": "<=0.2", "invented": "==0", "recall": ">=0.8"}: pass/fail per run (operators
                        <= < >= > == !=); stored in run.json "criteria_result" and shown as a `pass` column (pass / FAIL /
                        - when a metric is missing). With holdout_split / rank_split an unprefixed name like "f1" is
                        evaluated against "<rank_split>.f1" when that metric exists (criteria_result details name the
                        key used); otherwise it falls back to the plain key and records a warning. Prefixed names
                        ("train.f1", "holdout.f1") are used as they are.
  timing_keys           [metric names] that legitimately vary between identical runs (latency_ms, ...): ignored
                        by `determinism`, shown as mean±sd in repeated sweeps, never hidden by the holdout guard
  holdout_split         split name kept out of the tuning loop (see "Holdout guard")
  rank_split            split that sweeps, compare and criteria rank on: "<rank_split>.<metric>" (default "train";
                        e.g. "dev" or "cv" for systems that fit on train). Any split names work (train / dev /
                        holdout ...); must differ from holdout_split. Used when holdout_split or rank_split is set
  public_metrics        [metric names or glob patterns, e.g. "dev.*"] that the holdout guard must not hide although
                        they are not split-prefixed (only for numbers that cannot contain holdout items)

Driver / metrics contract
  Both commands run with cwd = config dir and these extra environment variables:
    BENCH_RUN_DIR         the run directory (history_root/runs/<run_id>)
    BENCH_CONFIG          <run dir>/config.json = defaults merged with the run's overrides
    BENCH_FIXTURE_<NAME>  absolute path of each fixture (NAME upper-cased)
    BENCH_HOLDOUT_SPLIT   bench.json holdout_split (only when set)
    BENCH_RANK_SPLIT      the rank split in effect (only when holdout_split or rank_split is set)
    BENCH_SHOW_HOLDOUT    "1" only for the metrics command of `bench.py metrics RUN --show-holdout`, never inherited
  Driver  writes $BENCH_RUN_DIR/result.json (atomically: temp file + rename):
            {"schema_version": 1, "status": "success" | "error", "error": {"code", "message"}?, ...}
          Exit 0 normally. Exit 75 (EX_TEMPFAIL) = transient / killed by someone else: status
          "killed", retryable. Any other non-zero exit without a valid result.json = status "error",
          error_code "driver_exit_<n>". stdout+stderr are saved to driver.log (its tail is printed on failure).
  Metrics runs only after a successful result and writes metrics.json (flat object of numbers /
          strings / bools / null), optionally metrics.md (human report), diff.md / diff.json (per-segment diff) and
          items.json (per-item results, below). A failure = status "metrics_failed" (metrics.log tail is printed);
          result.json is kept and `bench.py metrics RUN` re-runs just this stage.
  Statuses: success | error | timeout | killed | busy | metrics_failed. Only killed and busy are
          retried (--retries); error, timeout and metrics_failed are results, not noise.
  Exit codes: run: 0 success, 1 error/timeout/metrics_failed, 3 killed, 4 busy, 2 usage/config error,
          130 interrupted; determinism: 0 deterministic, 1 not; every command: 2 = usage/config error.

items.json (optional, written by the metrics command; stored with the run)
  {"schema_version": 1, "items": {"<item_id>": {"split": "train" | "holdout" | ... | null,
                                                 "score": <number | null>,   (higher is better)
                                                 "pass": <true | false | null>, "detail": {...anything...}}}}
  `bench.py items A B` compares two runs per item. For every split name that occurs, bench adds
  <split>.score_mean, <split>.pass_rate and <split>.n to metrics.json (never overwriting keys the metrics
  script wrote), so train vs holdout shows up in index / compare / criteria.
  Systems that FIT on gold (classifiers, learned thresholds): the driver must read gold only for the training items
  (the split assignment is in the splits fixture: $BENCH_FIXTURE_SPLITS) and the metrics script should report
  scores on items the system did not train on (a dev split or cross-validation) as <rank_split>.<metric>.

Re-scoring: `bench.py metrics RUN... | --all | --label-glob P | --group G | --sweep NAME` re-runs only the metrics
  command (never the driver); old metrics.*, diff.*, items.json go to prev/<timestamp>/. Recorded fixture hashes are
  never overwritten: if a fixture changed since the run, it refuses (exit 2) unless --force, which records the new
  hashes as rescored_fixtures, sets fixture_drift: true and marks the index `fixtures` column as <old>><new>.
  --env KEY=VALUE (repeatable) is passed to the metrics command (A/B scorer settings), recorded as metrics_env.
  --show-holdout sets BENCH_SHOW_HOLDOUT=1 for the metrics command (see "Holdout guard"), recorded as show_holdout.

Holdout guard (bench.json "holdout_split": "holdout")
  An unprefixed aggregate (accuracy, f1) is computed over ALL items, holdout included, so with holdout_split set the
  metrics script must emit split-prefixed quality metrics only: <split>.<metric> (train.f1, dev.f1, holdout.f1, ...);
  `all.<metric>` is the reserved name for aggregates over every item. Hidden in index.md, compare, sweep summaries,
  stats and `items` (and masked in `determinism`): every "holdout.*" key, and every metric key that is not
  namespaced to a non-holdout split ("train", the rank_split, or a split found in items.json), not in timing_keys
  and not in public_metrics. A one-time warning names the hidden unsplit keys. index.json is raw data: it keeps every
  key, unfiltered, and lists the hidden ones per run as `hidden_metrics`. `--show-holdout` (compare, stats, items,
  determinism) shows everything; `items --split holdout` shows the holdout items. The default ranking (compare, sweeps)
  uses "<rank_split>.<primary metric>" when that metric exists, else warns that the primary metric may include
  holdout items. Look at holdout once, for the final chosen config only.
  Per-run files (metrics.md, diff.md, items.json) are written by YOUR metrics script, bench cannot filter them:
  omit holdout detail from metrics.md / diff.md unless env BENCH_SHOW_HOLDOUT=1 (BENCH_HOLDOUT_SPLIT names the split;
  the items template does this), then `bench.py metrics RUN --show-holdout` re-scores one run with the details (and
  the all.* aggregates) for the final check; a plain re-score removes them again. items.json stays complete.

Files in a run dir: config.json, driver.log, result.json, metrics.json, metrics.md, metrics.log, diff.md, diff.json,
  items.json, run.json (config, fixtures + fixture_fingerprint, provenance, repo commit/dirty, status, group,
  criteria_result, rescored_at / rescored_fixtures / fixture_drift / rescored_provenance / metrics_env /
  show_holdout after a re-score) and
  prev/<timestamp>/ (derived files + run.json as they were before each re-score). Run ids are
  <UTC timestamp with milliseconds>_<label>, e.g. 20261112T105549123Z_base, so they sort in execution order.
  Anywhere a run id is accepted you can also give a unique id prefix or a unique label (--latest: the newest).
  --group takes the same forms for group ids: exact id, unique prefix, or the base label (`det` for
  `run --label det --repeat 2`). compare and sweep summaries warn when the runs differ in fixtures OR in code
  (driver / metrics script / provenance hashes; a re-score with a changed metrics script keeps the original hash in
  provenance.metrics and records the new one as rescored_provenance).
Privacy: umask 077, dirs 0700, files 0600, every history write is atomic (temp file + rename); run.json, index and
  sweep summaries hold config, hashes and numbers only, never output content. --env values are recorded in
  run.json: do not pass secrets.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from benchlib_config import (LABEL_RE, SCHEMA_VERSION, as_argv, check_override_keys, check_override_types, fingerprint,
                             fixtures_now, load_config, override_type_errors, parse_env, parse_sets, provenance_info,
                             repo_info, script_files, tuning_split)
from benchlib_items import compare_items, load_items, render_items_md, split_aggregates
from benchlib_runs import (Guard, build_comparison, common_label_prefix, ensure_metric_column, evaluate_criteria, load_run,
                           parse_columns, ranking_spec, render_comparison_md, resolve_columns, resolve_ids, select_runs,
                           visible_columns, with_hidden_columns, write_index)
from benchlib_stats import check_determinism, compute_stats, render_determinism_md, render_stats_md
from benchlib_sweep import SPEC_HELP, parse_sweep_spec, sweep_plan, write_sweep_summary
from benchlib_templates import TEMPLATES, next_steps, template_files
from benchlib_util import (UsageError, allocate_dir, close_stdout, ensure_dir, flush_quietly, iso_now, md_table,  # noqa: F401
                           name_candidates, read_json, reset_warnings, say, stamp, tail_lines, warn_once, write_atomic,
                           write_json)  # md_table: re-exported for tests

EXIT_FOR_STATUS = {"success": 0, "error": 1, "timeout": 1, "metrics_failed": 1, "killed": 3, "busy": 4}
RETRYABLE = ("killed", "busy")
EX_TEMPFAIL = 75
DERIVED_PATTERNS = ("metrics.*", "diff.*", "items.json")
_sleep = time.sleep  # indirection so tests can stub guard polling and sweep cooldowns


class Interrupted(Exception):
    """Ctrl-C / SIGTERM handled after the run was recorded (exit 130); args[0] is its run record."""


# ---------------------------------------------------------------- running commands

def build_env(cfg, run_dir, show_holdout=False):
    env = dict(os.environ)
    for stale in ("BENCH_SHOW_HOLDOUT", "BENCH_HOLDOUT_SPLIT", "BENCH_RANK_SPLIT"):  # bench-controlled: never inherited
        env.pop(stale, None)
    env["BENCH_RUN_DIR"] = str(run_dir)
    env["BENCH_CONFIG"] = str(run_dir / "config.json")
    if cfg["holdout_split"]:
        env["BENCH_HOLDOUT_SPLIT"] = cfg["holdout_split"]
    if tuning_split(cfg):
        env["BENCH_RANK_SPLIT"] = tuning_split(cfg)
    if show_holdout:
        env["BENCH_SHOW_HOLDOUT"] = "1"
    for name, path in cfg["fixtures"].items():
        env["BENCH_FIXTURE_" + re.sub(r"[^A-Za-z0-9]", "_", name).upper()] = str(path)
    return env


def kill_group(proc, grace=5):
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, None)):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(wait)
        except subprocess.TimeoutExpired:
            pass


def run_command(cmd, env, cwd, log_path, timeout):
    """Run cmd in its own process group, stdout+stderr appended to log_path.

    Returns the exit code, or None after a timeout (process group killed)."""
    fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        try:
            proc = subprocess.Popen(cmd, env=env, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=fd,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            os.write(fd, ("bench: cannot start %s: %s\n" % (cmd[0], exc)).encode())
            return 127
        try:
            return proc.wait(timeout=timeout or None)
        except subprocess.TimeoutExpired:
            kill_group(proc)
            return None
        except BaseException:  # Ctrl-C / SIGTERM: never leave the driver behind
            kill_group(proc)
            raise
    finally:
        os.close(fd)


def wait_for_guard(cfg):
    """True once the guard command returned 0 continuously for guard_quiet_seconds."""
    if not cfg["guard"]:
        return True
    quiet, poll, limit = cfg["guard_quiet_seconds"], cfg["guard_poll_seconds"], cfg["guard_max_wait_seconds"]
    start, quiet_since = time.monotonic(), None
    while True:
        now = time.monotonic()
        try:
            ok = subprocess.run(cfg["guard"], cwd=str(cfg["dir"]), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=60).returncode == 0
        except (OSError, subprocess.SubprocessError):
            ok = False
        if ok:
            quiet_since = now if quiet_since is None else quiet_since
            if now - quiet_since >= quiet:
                return True
        else:
            quiet_since = None
        remaining = limit - (time.monotonic() - start)
        if remaining <= 0:
            return False
        _sleep(min(poll, remaining))


def read_result(path):
    """-> (verdict, error_code): verdict is "success", "error" or None (missing / invalid)."""
    if not path.exists():
        return None, "no_result"
    res = read_json(path)
    if not isinstance(res, dict) or res.get("schema_version") != SCHEMA_VERSION or res.get("status") not in ("success", "error"):
        return None, "invalid_result"
    if res["status"] == "error":
        err = res.get("error")
        code = err.get("code") if isinstance(err, dict) else None
        return "error", code if isinstance(code, str) and code else "driver_error"
    return "success", None


def archive_derived(run_dir, snapshot=False):
    """Move derived files (metrics.*, diff.*, items.json) to prev/<timestamp>/ instead of deleting them; with
    snapshot also copy run.json there. Returns the directory or None when there was nothing to keep."""
    files = sorted({p for pat in DERIVED_PATTERNS for p in run_dir.glob(pat) if p.is_file()})
    if not files and not (snapshot and (run_dir / "run.json").exists()):
        return None
    prev = ensure_dir(run_dir / "prev")
    dest = prev / allocate_dir(prev, stamp())
    for f in files:
        os.replace(str(f), str(dest / f.name))
    if snapshot and (run_dir / "run.json").exists():
        write_atomic(dest / "run.json", (run_dir / "run.json").read_text(encoding="utf-8"))
    return dest


def run_metrics(cfg, run_dir, env, timeout):
    """Run the metrics command -> (status, error_code); previous derived files are archived first."""
    if not cfg["metrics"]:
        return "success", None
    archive_derived(run_dir)
    rc = run_command(cfg["metrics"], env, cfg["dir"], run_dir / "metrics.log", timeout)
    if rc is None:
        return "metrics_failed", "metrics_timeout"
    if rc != 0:
        return "metrics_failed", "metrics_exit_%d" % rc
    if not (run_dir / "metrics.json").exists():
        return "metrics_failed", "metrics_missing"
    m = read_json(run_dir / "metrics.json")
    if not isinstance(m, dict) or any(isinstance(v, (dict, list)) for v in m.values()):
        return "metrics_failed", "metrics_invalid"
    try:
        items = load_items(run_dir)
    except UsageError:
        return "metrics_failed", "items_invalid"
    if items is not None:  # per-split aggregates, never clobbering what the metrics script wrote
        extra = {k: v for k, v in split_aggregates(items).items() if k not in m}
        if extra:
            write_json(run_dir / "metrics.json", {**m, **extra})
    return "success", None


def drive(cfg, run_dir, driver, timeout):
    """One attempt after the run dir exists -> (status, error_code)."""
    env = build_env(cfg, run_dir)
    if not wait_for_guard(cfg):
        return "busy", "guard_timeout"
    rc = run_command(driver, env, cfg["dir"], run_dir / "driver.log", timeout)
    if rc is None:
        return "timeout", "driver_timeout"
    if rc == EX_TEMPFAIL:
        return "killed", "driver_exit_75"
    verdict, code = read_result(run_dir / "result.json")
    if verdict is None:
        return "error", code if rc == 0 else "driver_exit_%d" % rc
    if verdict == "error":
        return "error", code
    if rc != 0:
        return "error", "driver_exit_%d" % rc
    return run_metrics(cfg, run_dir, env, timeout)


def report_failure(run_dir, status, code):
    """Print where the evidence is (log path + last 15 lines) when the driver or the metrics command failed."""
    if status == "metrics_failed":
        name = "metrics.log"
    elif status in ("error", "timeout", "killed") and code != "interrupted":
        name = "driver.log"
    else:
        return
    path = run_dir / name
    if not path.exists():
        return
    print("bench: %s failed (%s); %s (last 15 lines):" % ("metrics" if name == "metrics.log" else "driver", code, path), file=sys.stderr)
    lines = tail_lines(path, 15)
    for line in lines or ["(log is empty)"]:
        print("  | " + line, file=sys.stderr)


def scoring_record(cfg, run_dir, status):
    """criteria_result for a run: evaluated on metrics.json after success, all-missing (pass null) otherwise."""
    metrics = read_json(run_dir / "metrics.json") if status == "success" else None
    result = evaluate_criteria(cfg["criteria"], metrics if isinstance(metrics, dict) else {}, cfg)
    for note in (result or {}).get("warnings", []):
        warn_once(note)
    return result


def run_attempt(cfg, label, merged, driver, timeout, fixtures, provenance, attempt, allowed, group=None):
    runs_dir = ensure_dir(ensure_dir(cfg["history_root"]) / "runs")
    run_id = allocate_dir(runs_dir, "%s_%s" % (stamp(), label))
    run_dir = runs_dir / run_id
    write_json(run_dir / "config.json", merged)
    started, t0, interrupted = iso_now(), time.monotonic(), False
    try:
        status, code = drive(cfg, run_dir, driver, timeout)
    except KeyboardInterrupt:
        status, code, interrupted = "killed", "interrupted", True
    record = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "label": label, "suite": cfg["suite"],
        "started_at": started, "finished_at": iso_now(), "status": status, "error_code": code,
        "attempt": attempt, "attempts_allowed": allowed, "config": merged, "fixtures": fixtures,
        "fixture_fingerprint": fingerprint(fixtures), "provenance": provenance,
        "repo": repo_info(cfg["dir"]), "host": {"platform": platform.platform(), "python": platform.python_version()},
        "duration_ms": int((time.monotonic() - t0) * 1000),
    }
    if group:
        record["group"] = group
    criteria = scoring_record(cfg, run_dir, status)
    if criteria:
        record["criteria_result"] = criteria
    write_json(run_dir / "run.json", record)
    write_index(cfg)
    if not interrupted:
        report_failure(run_dir, status, code)
    if interrupted:
        raise Interrupted(record)
    return record


def print_plan(cfg, label, merged, driver, retries, timeout):
    runs_dir = cfg["history_root"] / "runs"
    run_id = next(n for n in name_candidates("%s_%s" % (stamp(), label)) if not (runs_dir / n).exists())
    run_dir = runs_dir / run_id
    say("DRY RUN: nothing is created or executed")
    say("run_id:  %s" % run_id)
    say("config:  %s" % json.dumps(merged, sort_keys=True))
    say("env:")
    for k, v in sorted(build_env(cfg, run_dir).items()):
        if k.startswith("BENCH_"):
            say("  %s=%s" % (k, v))
    say("driver:  %s" % " ".join(shlex.quote(a) for a in driver))
    say("metrics: %s" % (" ".join(shlex.quote(a) for a in cfg["metrics"]) if cfg["metrics"] else "(none)"))
    say("guard:   %s" % (" ".join(shlex.quote(a) for a in cfg["guard"]) if cfg["guard"] else "(none)"))
    say("timeout_seconds: %s  retries: %d  cwd: %s" % (timeout, retries, cfg["dir"]))
    return {"run_id": run_id, "status": "dry-run", "error_code": None}


def execute_run(cfg, label, overrides, *, retries=None, timeout=None, driver=None, dry_run=False, group=None):
    """Run one config (with retries). Returns the final attempt's run record."""
    if not LABEL_RE.match(label):
        raise UsageError("label %r must match %s" % (label, LABEL_RE.pattern))
    driver = as_argv(driver, "--driver") or cfg["driver"]
    if not driver:
        raise UsageError("no driver: set 'driver' in bench.json or pass --driver")
    retries = cfg["retries"] if retries is None else retries
    timeout = cfg["timeout_seconds"] if timeout is None else timeout
    if retries < 0 or timeout < 0:
        raise UsageError("--retries and --timeout-seconds must be >= 0")
    merged = {**cfg["defaults"], **overrides}
    if dry_run:
        return print_plan(cfg, label, merged, driver, retries, timeout)
    missing = [n for n, p in cfg["fixtures"].items() if not p.exists()]
    if missing:
        raise UsageError("fixture(s) not found: %s" % ", ".join(missing))
    fixtures, provenance = fixtures_now(cfg), provenance_info(cfg, driver)
    for attempt in range(1, retries + 2):
        record = run_attempt(cfg, label, merged, driver, timeout, fixtures, provenance, attempt, retries + 1, group)
        say("%s  %s%s" % (record["run_id"], record["status"], "  (%s)" % record["error_code"] if record["error_code"] else ""))
        if record["status"] not in RETRYABLE or attempt > retries:
            break
        say("attempt %d/%d %s, retrying" % (attempt, retries + 1, record["status"]))
    return record


# ---------------------------------------------------------------- re-scoring

def unscorable_reason(run_dir):
    result = read_json(run_dir / "result.json")
    if not (isinstance(result, dict) and result.get("status") == "success") or not (run_dir / "config.json").exists():
        return "no successful result.json / config.json to score"
    return None


def drifted_fixtures(run, current):
    """Fixture names whose sha256 differs from what run.json recorded when the run was made."""
    recorded = run.get("fixtures") if isinstance(run.get("fixtures"), dict) else {}
    return [n for n, f in current.items() if n in recorded and isinstance(recorded[n], dict) and recorded[n].get("sha256") != f["sha256"]]


def rescore_run(cfg, run_id, current, drift, timeout, env_extra, show_holdout=False):
    """Re-run only the metrics command on one run dir -> (status, error_code). Recorded fixture hashes and the recorded
    metrics-script hash stay as they were; drift (allowed by --force) and a changed metrics script are recorded
    separately (rescored_fixtures, rescored_provenance) next to rescored_at."""
    run_dir = cfg["history_root"] / "runs" / run_id
    run = read_json(run_dir / "run.json")
    run = run if isinstance(run, dict) else {"run_id": run_id}
    archive_derived(run_dir, snapshot=True)
    env = build_env(cfg, run_dir, show_holdout)
    env.update(env_extra)
    status, code = run_metrics(cfg, run_dir, env, timeout)
    run.update(status=status, error_code=code, rescored_at=iso_now())
    for key, value in (("metrics_env", env_extra or None), ("fixture_drift", True if drift else None),
                       ("show_holdout", True if show_holdout else None),
                       ("rescored_fixtures", current if drift or "fixtures" not in run else None),
                       ("criteria_result", scoring_record(cfg, run_dir, status))):
        if value is None:
            run.pop(key, None)
        else:
            run[key] = value
    prov = run.get("provenance") if isinstance(run.get("provenance"), dict) else {}
    metrics_now = script_files(cfg["metrics"], cfg["dir"])
    if prov.get("metrics") != metrics_now:
        run["rescored_provenance"] = {"metrics": metrics_now}
    else:
        run.pop("rescored_provenance", None)
    write_json(run_dir / "run.json", run)
    return status, code


def rescore(cfg, run_ids, *, explicit=None, force=False, env_extra=None, timeout=None, show_holdout=False):
    """Re-score runs (metrics stage only) -> list of (run_id, status, error_code | skip reason).

    `explicit` = ids the user named (unscorable ones are a usage error; others are skipped). Fixture drift on any
    selected run refuses everything unless force."""
    if not cfg["metrics"]:
        raise UsageError("no metrics command configured")
    explicit = set(run_ids if explicit is None else explicit)
    runs = cfg["history_root"] / "runs"
    todo, skipped = [], []
    for rid in run_ids:
        why = unscorable_reason(runs / rid)
        if why and rid in explicit:
            raise UsageError("run %s has %s" % (rid, why))
        (skipped if why else todo).append(rid)
    if not todo:
        raise UsageError("none of the selected runs has a successful result to score")
    current = fixtures_now(cfg)
    drift = {rid: drifted_fixtures(read_json(runs / rid / "run.json") or {}, current) for rid in todo}
    drift = {rid: names for rid, names in drift.items() if names}
    if drift and not force:
        shown = "; ".join("%s: %s" % (rid, ", ".join(names)) for rid, names in list(drift.items())[:5])
        raise UsageError("refusing to re-score: fixture(s) changed since the run was recorded (%s%s). Scores would mix "
                         "different data. Pass --force to re-score anyway (recorded hashes stay untouched; the run is marked "
                         "fixture_drift: true with the new hashes in rescored_fixtures)."
                         % (shown, " ..." if len(drift) > 5 else ""))
    timeout = cfg["timeout_seconds"] if timeout is None else timeout
    out = [(rid, "skipped", "no successful result") for rid in skipped]
    try:
        for rid in todo:
            status, code = rescore_run(cfg, rid, current, drift.get(rid), timeout, env_extra or {}, show_holdout)
            say("%s  %s%s%s" % (rid, status, "  (%s)" % code if code else "", "  [fixture drift, --force]" if rid in drift else ""))
            out.append((rid, status, code))
    finally:
        write_index(cfg)
    for rid in skipped:
        say("%s  skipped (no successful result.json to score)" % rid)
    return out


# ---------------------------------------------------------------- commands

GITIGNORE_LINES = ("history/", "__pycache__/")


def update_gitignore(target):
    """Create DIR/.gitignore with the history and bytecode directories, or append the missing lines to an existing one.
    -> (path, lines added)."""
    path = target / ".gitignore"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    have = {line.strip().rstrip("/") for line in text.splitlines()}
    missing = [line for line in GITIGNORE_LINES if line.rstrip("/") not in have]
    if missing:
        write_atomic(path, text + ("\n" if text and not text.endswith("\n") else "") + "\n".join(missing) + "\n")
    return path, missing


def cmd_init(args):
    target = Path(args.dir or ".")
    here = Path(__file__).resolve().parent
    files = template_files(args.template)
    toolkit = ["bench.py"] + sorted(p.name for p in here.glob("benchlib_*.py")) + [n for n in ("textdiff.py", "calibrate.py") if (here / n).exists()]
    for name in toolkit:
        files[name] = (here / name).read_text(encoding="utf-8")
    existing = [n for n in files if (target / n).exists()]
    if existing:
        raise UsageError("refusing to overwrite: %s" % ", ".join(existing))
    for name, text in files.items():
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        write_atomic(target / name, text)
    os.chmod(target / "bench.py", 0o700)
    ignore, added = update_gitignore(target)
    try:
        rel = os.path.relpath(str(target), os.getcwd())
    except ValueError:
        rel = str(target)
    bench, config = (shlex.quote(os.path.join(rel, n)) for n in ("bench.py", "bench.json"))
    say("wrote %d files into %s (template: %s; bench.py and its helpers are copied, nothing points back to the skill)" % (
        len(files), target, args.template))
    if added:
        say("%s: added %s (run history can hold sensitive data: keep it out of git)" % (ignore.name, ", ".join(added)))
    say("Next, from this directory:\n  " + "\n  ".join(next_steps(args.template, bench, config)))
    say("history_root is ./history next to bench.json; move it outside the repo when fixtures or outputs are sensitive.\n"
        "bench.json keys: python3 %s -h" % bench)
    return 0


def cmd_run(args):
    if not LABEL_RE.match(args.label):
        raise UsageError("--label must match %s" % LABEL_RE.pattern)
    if args.repeat < 1:
        raise UsageError("--repeat must be >= 1")
    cfg = load_config(args.config)
    overrides = parse_sets(args.set, cfg["defaults"])
    check_override_keys(cfg, overrides, args.allow_new_keys)
    check_override_types(cfg, overrides)
    group = "%s_%s" % (stamp(), args.label) if args.repeat > 1 else None
    codes = []
    for k in range(1, args.repeat + 1):
        label = args.label if args.repeat == 1 else "%s-r%d" % (args.label, k)
        record = execute_run(cfg, label, overrides, retries=args.retries, timeout=args.timeout_seconds, driver=args.driver,
                             dry_run=args.dry_run, group=group)
        codes.append(0 if args.dry_run else EXIT_FOR_STATUS[record["status"]])
    if group and not args.dry_run:
        say("group: %s  (bench.py stats --group %s)" % (group, group))
    return max(codes)


def cmd_metrics(args):
    cfg = load_config(args.config)
    explicit = resolve_ids(cfg, list(args.run_ids) + list(args.run or []), args.latest)
    selected = select_runs(cfg, (), group=args.group, label_glob=args.label_glob, all_runs=args.all, sweep=args.sweep,
                           latest=args.latest)
    ids = sorted(set(explicit) | set(selected))
    if not ids:
        raise UsageError("no runs selected: give RUN_ID(s) or use --all, --label-glob, --group, --sweep")
    results = rescore(cfg, ids, explicit=explicit, force=args.force, env_extra=parse_env(args.env), timeout=args.timeout_seconds,
                      show_holdout=args.show_holdout)
    failed = [r for r in results if r[1] == "metrics_failed"]
    ok = [r for r in results if r[1] == "success"]
    skipped = [r for r in results if r[1] == "skipped"]
    if len(results) > 1:
        say("re-scored %d run(s): %d success, %d metrics_failed, %d skipped" % (len(ok) + len(failed), len(ok), len(failed), len(skipped)))
    return 1 if failed else 0


def cmd_index(args):
    cfg = load_config(args.config)
    rows = write_index(cfg)
    say("indexed %d run(s): %s" % (len(rows), cfg["history_root"] / "index.md"))
    return 0


def cmd_compare(args):
    cfg = load_config(args.config)
    if args.top is not None and args.top < 1:
        raise UsageError("--top must be >= 1")
    runs = cfg["history_root"] / "runs"
    rows = [load_run(runs / rid) for rid in resolve_ids(cfg, args.run_ids, args.latest)]
    guard = Guard(cfg, rows)
    cols = parse_columns(args.columns, rows, guard, args.show_holdout) if args.columns else visible_columns(
        resolve_columns(cfg, rows), guard, args.show_holdout)
    if args.show_holdout and not args.columns:
        cols = with_hidden_columns(cols, rows, guard)
    if args.sort and not any(args.sort in r["metrics"] for r in rows):
        raise UsageError("--sort %r is not a metric of these runs; metrics: %s" % (args.sort, ", ".join(sorted({k for r in rows for k in r["metrics"]})) or "(none)"))
    sort, notes = (None, []) if args.no_sort else ranking_spec(cfg, rows, args.sort, True if args.desc else False if args.asc else None, args.show_holdout)
    if not args.show_holdout:
        notes = notes + guard.hidden_note(rows)
        guard.announce(rows)
    if sort:
        cols = ensure_metric_column(cols, sort["key"])
    comp = build_comparison(rows, cols, baseline_id=rows[0]["run_id"], sort=sort, top=args.top, notes=notes)
    prefix = None if args.wide else common_label_prefix([r["label"] for r in rows])
    md = render_comparison_md(comp, "Comparison", wide=args.wide, strip_prefix=prefix)
    say(md)
    if args.markdown_out:
        write_atomic(Path(args.markdown_out), md)
    if args.json_out:
        write_json(Path(args.json_out), comp)
    return 0


def selected_rows(cfg, args):
    ids = select_runs(cfg, args.run_ids, group=args.group, label_glob=args.label_glob, latest=args.latest)
    if not ids:
        raise UsageError("no runs selected: give RUN_ID(s), --group or --label-glob")
    return [load_run(cfg["history_root"] / "runs" / rid) for rid in ids]


def cmd_stats(args):
    cfg = load_config(args.config)
    rows = selected_rows(cfg, args)
    stats = compute_stats(rows, cfg["timing_keys"])
    guard, note = Guard(cfg, rows), []
    if not args.show_holdout:
        stats["metrics"] = {k: v for k, v in stats["metrics"].items() if not guard.hidden(k)}
        note = guard.hidden_note(rows)
        guard.announce(rows)
    if not stats["metrics"]:
        raise UsageError("the selected runs have no numeric metrics")
    md = render_stats_md(stats) + "".join("\n%s\n" % n for n in note)
    say(md)
    if args.markdown_out:
        write_atomic(Path(args.markdown_out), md)
    if args.json_out:
        write_json(Path(args.json_out), stats)
    return 0


def cmd_determinism(args):
    cfg = load_config(args.config)
    rows = selected_rows(cfg, args)
    res = check_determinism(rows, cfg["timing_keys"])
    guard = Guard(cfg, rows)
    if not args.show_holdout:  # the verdict counts every key; hidden keys only show that they differ, not their values
        for key in [k for k in res["differing"] if guard.hidden(k)]:
            res["differing"][key] = {rid: "(hidden)" for rid in res["differing"][key]}
        guard.announce(rows)
    md = render_determinism_md(res)
    say(md)
    if args.markdown_out:
        write_atomic(Path(args.markdown_out), md)
    if args.json_out:
        write_json(Path(args.json_out), res)
    return 0 if res["identical"] else 1


def cmd_items(args):
    cfg = load_config(args.config)
    if args.threshold < 0:
        raise UsageError("--threshold must be >= 0")
    a_id, b_id = resolve_ids(cfg, [args.run_a, args.run_b], args.latest)
    runs = cfg["history_root"] / "runs"
    loaded = []
    for rid in (a_id, b_id):
        items = load_items(runs / rid)
        if items is None:
            raise UsageError("run %s has no items.json (the metrics command must write it; see bench.py -h)" % rid)
        loaded.append(items)
    hide = None if args.show_holdout else cfg["holdout_split"]
    res = compare_items(loaded[0], loaded[1], split=args.split, threshold=args.threshold, hide_split=hide)
    md = render_items_md(res, a_id, b_id, hide)
    say(md)
    if args.markdown_out:
        write_atomic(Path(args.markdown_out), md)
    if args.json_out:
        write_json(Path(args.json_out), {"a": a_id, "b": b_id, **res})
    return 0


def sweep_result(config, label, rec, repeat, group):
    return {"config": config, "label": label, "run_id": rec["run_id"], "status": rec["status"],
            "error_code": rec.get("error_code"), "repeat": repeat, "group": group}


def cmd_sweep(args):
    if args.explain:
        say(SPEC_HELP)
        return 0
    if not args.spec:
        raise UsageError("--spec is required (bench.py sweep --explain prints the spec format)")
    cfg = load_config(args.config)
    spec = parse_sweep_spec(args.spec)
    keys = set(spec["base"]).union(*(set(ov) for _, ov in spec["entries"]))
    check_override_keys(cfg, keys, args.allow_new_keys or spec["allow_new_keys"], "sweep spec")
    type_errors = [("base", override_type_errors(cfg, spec["base"]))] + [("config %r" % n, override_type_errors(cfg, ov)) for n, ov in spec["entries"]]
    type_errors = ["%s: %s" % (where, "; ".join(errs)) for where, errs in type_errors if errs]
    if type_errors:
        raise UsageError("sweep spec: wrong type for %s. Keys not in bench.json defaults (allow_new_keys) are untyped; a null "
                         "default accepts any type" % " | ".join(type_errors))
    include = resolve_ids(cfg, args.include_run or [], args.latest, what="run id (--include-run)")
    groups, started, results = {}, iso_now(), []
    try:
        for idx, (config, label, overrides, repeat) in enumerate(sweep_plan(spec)):
            if idx and spec["cooldown"] and not args.dry_run:
                _sleep(spec["cooldown"])  # between runs only
            group = None
            if repeat:
                group = groups.setdefault(config, "%s_%s__%s" % (stamp(), spec["name"], config))
            try:
                rec = execute_run(cfg, label, overrides, retries=args.retries, timeout=args.timeout_seconds,
                                  driver=args.driver, dry_run=args.dry_run, group=group)
            except Interrupted as exc:  # the interrupted run is part of the sweep's record
                results.append(sweep_result(config, label, exc.args[0], repeat, group))
                raise
            results.append(sweep_result(config, label, rec, repeat, group))
    finally:  # also after Ctrl-C: keep what was measured so far
        if results and not args.dry_run:
            write_sweep_summary(cfg, spec, include, results, started, wide=args.wide)
    return 0 if args.dry_run or all(r["status"] == "success" for r in results) else 1


# ---------------------------------------------------------------- parser

RUN_HELP = """\
Runs the driver once (or --repeat N times) with defaults + --set overrides; records run.json, result.json,
metrics, then rebuilds the index. --set KEY=VALUE takes a JSON value or a plain string and KEY must exist in
bench.json "defaults" (typos exit 2 with the valid keys; --allow-new-keys or "allow_new_keys": true bypasses) and
must have the type of its default (int / float / bool / str / list / dict; an int is fine for a float; a null default
accepts anything; for a str default the text is taken as is, so `--set model=3` is "3"): a mismatch exits 2.
--repeat N makes N runs labeled <label>-r1..-rN that share one `group` id; `bench.py stats --group G` summarizes.
Exit: 0 success, 1 error/timeout/metrics_failed, 3 killed, 4 busy (the worst over repeats), 2 usage, 130 interrupted.
"""
METRICS_HELP = """\
Re-runs only the metrics command on finished runs (the driver never runs again); the previous metrics.*, diff.*,
items.json move to prev/<timestamp>/ in the run dir. Select runs by RUN... (id / unique prefix / unique label),
--run ID, --all, --label-glob PATTERN, --group G, --sweep NAME (union). Each run prints a status line; a failing
run does not stop the others (exit 1 if any failed). Recorded fixture hashes are never overwritten: if a fixture
differs from the recorded hash the command refuses (exit 2) unless --force, which records the new hashes in
rescored_fixtures and marks the run fixture_drift: true. A changed metrics script is recorded as rescored_provenance
(provenance.metrics keeps the original hash). --env KEY=VALUE (repeatable) is passed to the metrics command to A/B
scorer settings; it is recorded in run.json as metrics_env (so never pass secrets). --show-holdout sets
BENCH_SHOW_HOLDOUT=1 for the metrics command, so a metrics script that honors it writes holdout details too (final
check only; a plain re-score removes them). --group takes a group id, a unique prefix or the base label.
"""
COMPARE_HELP = """\
Side-by-side table of runs (id / unique prefix / unique label), deltas vs the first listed run (the baseline).
With a primary_metric in bench.json (or --sort METRIC) rows are ranked best first with a `rank` column; --top N
keeps the first N rows; --asc/--desc override the direction (default: the primary metric's goal, else descending).
--columns takes config / metric names (x, config.x, metrics.x); unknown names exit 2. Labels are shown without a
common <sweep>__ prefix and without run ids unless --wide. Warns when the runs used different fixtures or different
code (driver / metrics script / provenance hashes). With bench.json holdout_split, holdout.* and unsplit metrics (which
may include holdout items) are hidden: --show-holdout shows them (final check only), public_metrics allowlists keys.
"""
ITEMS_HELP = """\
Per-item comparison of two runs from their items.json (B relative to A): regressed / improved / unchanged /
only in A / only in B, sorted by score delta. A pass flip decides; otherwise the score delta must exceed
--threshold (default 0). Item scores are higher-is-better. Item ids, splits, scores and pass flags only,
never `detail` content.
"""
INIT_HELP = """\
Writes bench.json, a stub driver, a stub metrics script and sample fixtures into DIR and COPIES bench.py,
benchlib_*.py, textdiff.py and calibrate.py there (no absolute paths); refuses to overwrite existing files.
  --template text   one text fixture scored with WER (default)
  --template items  several JSON items + gold + splits.json scored per item (precision/recall/F1/exact; plain-string
                    gold = single-label classification: accuracy, macro-F1, per-class P/R/F1, confusion matrix via
                    benchlib_classify.py), writes items.json, split-prefixed metrics only (train.f1, holdout.f1; all.*
                    only with BENCH_SHOW_HOLDOUT=1), reports that leave the holdout out, a per-item diff.md;
                    bench.json shows primary_metric, criteria, timing_keys, holdout_split, rank_split, public_metrics
                    and provenance
Also writes DIR/.gitignore with history/ and __pycache__/ (appends missing lines to an existing one).
"""


def build_parser():
    fmt = argparse.RawDescriptionHelpFormatter
    ap = argparse.ArgumentParser(prog="bench.py", description=__doc__.splitlines()[0], epilog=__doc__, formatter_class=fmt)
    sub = ap.add_subparsers(dest="command", required=True, metavar="{init,run,sweep,metrics,index,compare,stats,determinism,items}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="bench.json", help="path to bench.json (default ./bench.json)")
    latest = argparse.ArgumentParser(add_help=False)
    latest.add_argument("--latest", action="store_true", help="a label that matches several runs means the newest one")
    runopts = argparse.ArgumentParser(add_help=False)
    runopts.add_argument("--retries", type=int, help="extra attempts after a killed/busy attempt")
    runopts.add_argument("--timeout-seconds", type=float, help="per-command timeout (0 = none)")
    runopts.add_argument("--driver", help="override the driver command (shlex-split), e.g. a stub")
    runopts.add_argument("--dry-run", action="store_true", help="print what would run; create nothing")
    runopts.add_argument("--allow-new-keys", action="store_true", help="allow overrides of keys not in bench.json defaults")
    outs = argparse.ArgumentParser(add_help=False)
    outs.add_argument("--markdown-out", metavar="FILE")
    outs.add_argument("--json-out", metavar="FILE")
    pick = argparse.ArgumentParser(add_help=False)
    pick.add_argument("run_ids", nargs="*", metavar="RUN", help="run id, unique id prefix or unique label")
    pick.add_argument("--group", help="all runs of a --repeat / sweep `repeats` group: group id, unique id prefix or base label "
                                      "(e.g. `det` for `run --label det --repeat 2`; --latest: the newest group of that label)")
    pick.add_argument("--label-glob", metavar="PATTERN", help="runs whose label matches, e.g. 'sw__base-r*'")

    p = sub.add_parser("init", help="write a starter bench.json + stub driver/metrics + fixtures, copy the toolkit",
                       description=INIT_HELP, epilog=INIT_HELP, formatter_class=fmt)
    p.add_argument("dir", nargs="?", default=".")
    p.add_argument("--template", choices=TEMPLATES, default="text")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("run", parents=[common, runopts], help="run one config end to end", epilog=RUN_HELP, formatter_class=fmt)
    p.add_argument("--label", required=True, help="letters, digits, . _ - (must start alphanumeric)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config key (JSON value or string)")
    p.add_argument("--repeat", type=int, default=1, metavar="N", help="N runs labeled <label>-r1..-rN in one group")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("sweep", parents=[common, runopts, latest], help="run a sweep spec sequentially (--explain: spec format)",
                       description="Run a sweep spec sequentially.\n\n" + SPEC_HELP, formatter_class=fmt)
    p.add_argument("--spec", help="sweep spec JSON")
    p.add_argument("--explain", action="store_true", help="print the sweep spec format with an example and exit")
    p.add_argument("--include-run", action="append", metavar="RUN", help="older run as a leading baseline row")
    p.add_argument("--wide", action="store_true", help="summary tables with run ids and full labels")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("metrics", aliases=["rescore"], parents=[common, pick, latest], help="re-run only the metrics command on runs",
                       description=METRICS_HELP, formatter_class=fmt)
    p.add_argument("--run", action="append", metavar="RUN", help="a run (same as a positional RUN; repeatable)")
    p.add_argument("--all", action="store_true", help="every run in history")
    p.add_argument("--sweep", metavar="NAME", help="the runs of the newest sweep summary with this name")
    p.add_argument("--force", action="store_true", help="re-score although fixtures differ from the recorded hashes")
    p.add_argument("--env", action="append", metavar="KEY=VALUE", help="environment for the metrics command (recorded in run.json)")
    p.add_argument("--show-holdout", action="store_true",
                   help="set BENCH_SHOW_HOLDOUT=1 for the metrics command so it writes holdout details too (final check only)")
    p.add_argument("--timeout-seconds", type=float)
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser("index", parents=[common], help="rebuild index.json + index.md")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("compare", parents=[common, latest, outs], help="side-by-side runs with deltas vs the first",
                       description=COMPARE_HELP, formatter_class=fmt)
    p.add_argument("run_ids", nargs="+", metavar="RUN", help="run id, unique id prefix or unique label")
    p.add_argument("--columns", help="comma-separated config/metric keys (default: index_columns)")
    p.add_argument("--sort", metavar="METRIC", help="rank by this metric (default: primary_metric)")
    direction = p.add_mutually_exclusive_group()
    direction.add_argument("--asc", action="store_true", help="lowest first")
    direction.add_argument("--desc", action="store_true", help="highest first")
    p.add_argument("--no-sort", action="store_true", help="keep the listed order, no ranking")
    p.add_argument("--top", type=int, metavar="N", help="show only the first N rows")
    p.add_argument("--show-holdout", action="store_true", help="include holdout_split metric columns and the unsplit metrics the guard hides")
    p.add_argument("--wide", action="store_true", help="run ids and full labels")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("stats", parents=[common, pick, latest, outs], help="mean, sd, min, max, n, cv%% per metric over runs")
    p.add_argument("--show-holdout", action="store_true", help="include holdout_split metrics and the unsplit metrics the guard hides")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("determinism", parents=[common, pick, latest, outs],
                       help="exit 0 when all non-timing metrics are identical across runs, else 1")
    p.add_argument("--show-holdout", action="store_true", help="show the values of hidden (holdout / unsplit) metrics that differ")
    p.set_defaults(func=cmd_determinism)

    p = sub.add_parser("items", parents=[common, latest, outs], help="per-item comparison of two runs (items.json)",
                       description=ITEMS_HELP, formatter_class=fmt)
    p.add_argument("run_a", metavar="A")
    p.add_argument("run_b", metavar="B")
    p.add_argument("--split", help="only items of this split")
    p.add_argument("--threshold", type=float, default=0.0, help="score change that counts as regressed/improved (default 0)")
    p.add_argument("--show-holdout", action="store_true", help="include the holdout_split items")
    p.set_defaults(func=cmd_items)
    return ap


def _on_sigterm(signum, frame):
    raise KeyboardInterrupt()


def main(argv=None):
    os.umask(0o077)
    signal.signal(signal.SIGINT, signal.default_int_handler)  # even when started with SIGINT ignored
    signal.signal(signal.SIGTERM, _on_sigterm)  # same cleanup path as Ctrl-C
    reset_warnings()
    try:
        args = build_parser().parse_args(argv)
        return args.func(args)
    except UsageError as exc:
        print("bench: error: %s" % exc, file=sys.stderr)
        return 2
    except BrokenPipeError:  # stdout closed (`| head`)
        close_stdout()
        return 0
    except (Interrupted, KeyboardInterrupt):
        print("bench: interrupted", file=sys.stderr)
        return 130
    finally:  # also for argparse's -h / SystemExit: buffered output must not die at interpreter exit
        flush_quietly()


if __name__ == "__main__":
    sys.exit(main())
