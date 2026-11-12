#!/usr/bin/env python3
"""Generic reference-benchmark harness (stdlib only, Python >= 3.9).

Commands: run | sweep | metrics (alias: rescore) | index | compare | init.
Wraps any project driver + metrics command; keeps history under history_root/runs/<run_id>/.

bench.json (relative paths resolve against the config file's directory, ~ is expanded):
  suite, history_root, fixtures {name: path}, driver [argv], metrics [argv], defaults {},
  timeout_seconds (1800; 0 = none), retries (0), guard [argv], guard_quiet_seconds (0),
  guard_poll_seconds (5), guard_max_wait_seconds (600),
  index_columns {"config": [keys], "metrics": [keys]} (absent = union of all keys)

Driver / metrics contract
  Both commands run with cwd = config dir and these extra environment variables:
    BENCH_RUN_DIR         the run directory (history_root/runs/<run_id>)
    BENCH_CONFIG          <run dir>/config.json = defaults merged with the run's overrides
    BENCH_FIXTURE_<NAME>  absolute path of each fixture (NAME upper-cased)
  Driver  writes $BENCH_RUN_DIR/result.json:
            {"schema_version": 1, "status": "success" | "error", "error": {"code", "message"}?, ...}
          Exit 0 normally. Exit 75 (EX_TEMPFAIL) = transient / killed by someone else: status
          "killed", retryable. Any other non-zero exit without a valid result.json = status "error",
          error_code "driver_exit_<n>". stdout+stderr are saved to driver.log.
  Metrics runs only after a successful result and writes metrics.json (flat object of numbers /
          strings / null), optionally metrics.md, diff.md, diff.json. A failure = status
          "metrics_failed"; result.json is kept and `bench.py metrics --run ID` re-runs just this
          stage and refreshes run.json + index.
  Statuses: success | error | timeout | killed | busy | metrics_failed. Only killed and busy are
          retried (--retries); error, timeout and metrics_failed are results, not noise.
  Exit codes (run): 0 success, 1 error/timeout/metrics_failed, 3 killed, 4 busy, 2 usage/config
          error, 130 interrupted.
Privacy: umask 077, dirs 0700, files 0600; run.json, index and sweep summaries hold config,
hashes and numbers only, never output content.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
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

SCHEMA_VERSION = 1
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
EXIT_FOR_STATUS = {"success": 0, "error": 1, "timeout": 1, "metrics_failed": 1, "killed": 3, "busy": 4}
RETRYABLE = ("killed", "busy")
EX_TEMPFAIL = 75
CORE = ("run_id", "label", "status", "attempt")
DERIVED_FILES = ("metrics.json", "metrics.md", "diff.md", "diff.json")
_sleep = time.sleep  # indirection so tests can stub guard polling and sweep cooldowns


class UsageError(Exception):
    """Bad arguments or configuration (exit 2)."""


class Interrupted(Exception):
    """Ctrl-C / SIGTERM handled after the run was recorded (exit 130); args[0] is its run record."""


# ---------------------------------------------------------------- files and small helpers

def say(msg):
    print(msg, flush=True)


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def write_atomic(path, text):
    tmp = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(str(tmp), str(path))


def write_json(path, obj):
    write_atomic(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def iso_now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def name_candidates(base):
    yield base
    for n in itertools.count(2):
        yield "%s-%d" % (base, n)


def allocate_dir(parent, base):
    """Atomically create parent/base (or base-2, base-3, ...); return the name used."""
    for name in name_candidates(base):
        try:
            (parent / name).mkdir(mode=0o700)
        except FileExistsError:
            continue
        os.chmod(parent / name, 0o700)
        return name


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def fixture_info(path):
    """{path_name (basename only), bytes, sha256}; a directory hashes names + contents."""
    info = {"path_name": path.name, "bytes": None, "sha256": None}
    if not path.exists():
        return info
    digest, total = hashlib.sha256(), 0
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for f in files:
        if path.is_dir():
            digest.update(str(f.relative_to(path)).encode() + b"\0")
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
                total += len(chunk)
    info.update(bytes=total, sha256=digest.hexdigest())
    return info


def repo_info(directory):
    def git(*args):
        try:
            r = subprocess.run(["git", "-C", str(directory)] + list(args), capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() if r.returncode == 0 else None
    commit = git("rev-parse", "HEAD")
    if commit is None:
        return None
    status = git("status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


# ---------------------------------------------------------------- config

def as_argv(value, what):
    if value is None:
        return None
    if isinstance(value, str):
        value = shlex.split(value)
    if not (isinstance(value, list) and value and all(isinstance(x, str) for x in value)):
        raise UsageError("%s must be a non-empty command (list of strings)" % what)
    return value


def load_config(path):
    path = Path(path).expanduser()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise UsageError("cannot read config %s: %s" % (path, exc))
    except ValueError as exc:
        raise UsageError("invalid JSON in %s: %s" % (path, exc))
    if not isinstance(raw, dict) or not raw.get("history_root"):
        raise UsageError("%s: 'history_root' is required" % path)
    base = Path(os.path.abspath(str(path))).parent

    def resolve(p):
        p = Path(str(p)).expanduser()
        return p if p.is_absolute() else Path(os.path.normpath(str(base / p)))

    def number(key, default):
        v = raw.get(key, default)
        if not is_num(v) or v < 0:
            raise UsageError("%s: '%s' must be a number >= 0" % (path, key))
        return v

    fixtures, defaults = raw.get("fixtures") or {}, raw.get("defaults") or {}
    index_columns = raw.get("index_columns") or {}
    if not all(isinstance(x, dict) for x in (fixtures, defaults, index_columns)):
        raise UsageError("%s: 'fixtures', 'defaults' and 'index_columns' must be objects" % path)
    return {
        "dir": base, "suite": str(raw.get("suite") or "bench"), "history_root": resolve(raw["history_root"]),
        "fixtures": {str(k): resolve(v) for k, v in fixtures.items()}, "defaults": defaults,
        "driver": as_argv(raw.get("driver"), "driver"), "metrics": as_argv(raw.get("metrics"), "metrics"),
        "guard": as_argv(raw.get("guard"), "guard"), "index_columns": index_columns,
        "timeout_seconds": number("timeout_seconds", 1800), "retries": int(number("retries", 0)),
        "guard_quiet_seconds": number("guard_quiet_seconds", 0), "guard_poll_seconds": number("guard_poll_seconds", 5),
        "guard_max_wait_seconds": number("guard_max_wait_seconds", 600),
    }


def parse_sets(items):
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise UsageError("--set expects key=value, got %r" % item)
        try:
            out[key] = json.loads(value)
        except ValueError:
            out[key] = value
    return out


def build_env(cfg, run_dir):
    env = dict(os.environ)
    env["BENCH_RUN_DIR"] = str(run_dir)
    env["BENCH_CONFIG"] = str(run_dir / "config.json")
    for name, path in cfg["fixtures"].items():
        env["BENCH_FIXTURE_" + re.sub(r"[^A-Za-z0-9]", "_", name).upper()] = str(path)
    return env


# ---------------------------------------------------------------- running commands

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


def run_metrics(cfg, run_dir, env, timeout):
    """Run the metrics command -> (status, error_code)."""
    if not cfg["metrics"]:
        return "success", None
    for name in DERIVED_FILES:
        (run_dir / name).unlink(missing_ok=True)
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


def run_attempt(cfg, label, merged, driver, timeout, fixtures, attempt, allowed):
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
        "repo": repo_info(cfg["dir"]), "host": {"platform": platform.platform(), "python": platform.python_version()},
        "duration_ms": int((time.monotonic() - t0) * 1000),
    }
    write_json(run_dir / "run.json", record)
    write_index(cfg)
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


def execute_run(cfg, label, overrides, *, retries=None, timeout=None, driver=None, dry_run=False):
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
    fixtures = {n: fixture_info(p) for n, p in cfg["fixtures"].items()}
    for attempt in range(1, retries + 2):
        record = run_attempt(cfg, label, merged, driver, timeout, fixtures, attempt, retries + 1)
        say("%s  %s%s" % (record["run_id"], record["status"], "  (%s)" % record["error_code"] if record["error_code"] else ""))
        if record["status"] not in RETRYABLE or attempt > retries:
            break
        say("attempt %d/%d %s, retrying" % (attempt, retries + 1, record["status"]))
    return record


def rescore(cfg, run_id, timeout=None):
    """Re-run only the metrics command on an existing run dir; refresh run.json + index."""
    run_dir = cfg["history_root"] / "runs" / run_id
    if not run_dir.is_dir():
        raise UsageError("unknown run id %r" % run_id)
    if not cfg["metrics"]:
        raise UsageError("no metrics command configured")
    result = read_json(run_dir / "result.json")
    if not (isinstance(result, dict) and result.get("status") == "success") or not (run_dir / "config.json").exists():
        raise UsageError("run %s has no successful result.json / config.json to score" % run_id)
    run = read_json(run_dir / "run.json")
    run = run if isinstance(run, dict) else {"run_id": run_id}
    fixtures = {n: fixture_info(p) for n, p in cfg["fixtures"].items()}
    changed = [n for n, f in fixtures.items() if n in (run.get("fixtures") or {}) and run["fixtures"][n].get("sha256") != f["sha256"]]
    if changed:
        print("bench: warning: fixture(s) changed since the run: %s" % ", ".join(changed), file=sys.stderr)
    timeout = cfg["timeout_seconds"] if timeout is None else timeout
    status, code = run_metrics(cfg, run_dir, build_env(cfg, run_dir), timeout)
    run.update(status=status, error_code=code, fixtures=fixtures, rescored_at=iso_now())
    write_json(run_dir / "run.json", run)
    write_index(cfg)
    say("%s  %s%s" % (run_id, status, "  (%s)" % code if code else ""))
    return status


# ---------------------------------------------------------------- index and comparison

def load_run(run_dir):
    """Tolerant loader: a dir may lack run.json and/or metrics.json."""
    run, metrics = read_json(run_dir / "run.json"), read_json(run_dir / "metrics.json")
    run = run if isinstance(run, dict) else {}
    config = run["config"] if "config" in run else read_json(run_dir / "config.json")
    return {
        "run_id": run_dir.name, "label": run.get("label") or run_dir.name.split("_", 1)[-1],
        "status": run.get("status") or "unknown", "attempt": run.get("attempt"), "error_code": run.get("error_code"),
        "started_at": run.get("started_at"), "duration_ms": run.get("duration_ms"),
        "config": config if isinstance(config, dict) else {}, "metrics": metrics if isinstance(metrics, dict) else {},
    }


def load_runs(root):
    runs = root / "runs"
    dirs = sorted(p for p in runs.iterdir() if p.is_dir() and not p.name.startswith(".")) if runs.is_dir() else []
    return [load_run(d) for d in dirs]


def make_columns(config_keys, metric_keys):
    cols = []
    for kind, keys, other in (("config", config_keys, metric_keys), ("metrics", metric_keys, config_keys)):
        for k in map(str, keys):
            clash = k in CORE or k in map(str, other)
            cols.append({"key": k, "kind": kind, "header": "%s.%s" % (kind, k) if clash else k})
    return cols


def resolve_columns(cfg, rows):
    ic = cfg["index_columns"]
    config_keys = ic.get("config")
    metric_keys = ic.get("metrics")
    if config_keys is None:
        config_keys = sorted({k for r in rows for k in r["config"]})
    if metric_keys is None:
        metric_keys = sorted({k for r in rows for k in r["metrics"]})
    return make_columns(config_keys, metric_keys)


def cell(v):
    if v is None:
        return "–"
    if isinstance(v, float):
        return "%.6g" % v
    if isinstance(v, (bool, dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def md_table(headers, body):
    def esc(v):
        return cell(v).replace("|", "\\|").replace("\r", "").replace("\n", "<br>")
    lines = ["| " + " | ".join(esc(h) for h in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(esc(v) for v in row) + " |" for row in body]
    return "\n".join(lines)


def write_index(cfg):
    root = ensure_dir(cfg["history_root"])
    rows = load_runs(root)
    cols = resolve_columns(cfg, rows)
    out = []
    for r in rows:
        row = {k: r[k] for k in CORE + ("error_code", "started_at", "duration_ms")}
        for kind in ("config", "metrics"):
            row[kind] = {c["key"]: r[kind].get(c["key"]) for c in cols if c["kind"] == kind}
        out.append(row)
    write_json(root / "index.json", out)
    headers = list(CORE) + [c["header"] for c in cols]
    body = [[r[k] for k in CORE] + [r[c["kind"]].get(c["key"]) for c in cols] for r in rows]
    write_atomic(root / "index.md", "# Index: %s\n\n%s\n" % (cfg["suite"], md_table(headers, body)))
    return rows


def build_comparison(rows, cols):
    """Rows side by side; numeric metric deltas vs the first row; config keys that differ."""
    base = rows[0]
    delta_cols = [c for c in cols if c["kind"] == "metrics" and any(is_num(r["metrics"].get(c["key"])) for r in rows)]
    out = []
    for r in rows:
        deltas = {}
        for c in delta_cols:
            a, b = base["metrics"].get(c["key"]), r["metrics"].get(c["key"])
            deltas[c["header"]] = b - a if r is not base and is_num(a) and is_num(b) else None
        out.append({**{k: r[k] for k in CORE}, "values": {c["header"]: r[c["kind"]].get(c["key"]) for c in cols}, "deltas": deltas})
    keys = sorted({k for r in rows for k in r["config"]})
    diffs = [{"key": k, "values": {r["run_id"]: r["config"].get(k) for r in rows}} for k in keys
             if len({json.dumps(r["config"].get(k), sort_keys=True) for r in rows}) > 1]
    return {"baseline": base["run_id"], "columns": [c["header"] for c in cols],
            "delta_columns": [c["header"] for c in delta_cols], "rows": out, "config_differences": diffs}


def render_comparison_md(comp, title):
    headers = list(CORE) + comp["columns"] + ["Δ" + h for h in comp["delta_columns"]]
    body = []
    for r in comp["rows"]:
        deltas = ["–" if r["deltas"][h] is None else "%+.6g" % r["deltas"][h] for h in comp["delta_columns"]]
        body.append([r[k] for k in CORE] + [r["values"][h] for h in comp["columns"]] + deltas)
    text = "# %s\n\nDeltas are relative to the first run: `%s`\n\n%s\n\n## Config differences\n\n" % (
        title, comp["baseline"], md_table(headers, body))
    diffs = comp["config_differences"]
    if not diffs:
        return text + "None.\n"
    ids = [r["run_id"] for r in comp["rows"]]
    return text + md_table(["key"] + ids, [[d["key"]] + [d["values"][i] for i in ids] for d in diffs]) + "\n"


# ---------------------------------------------------------------- sweep

SPEC_KEYS = {"name", "base", "configs", "matrix", "repeat_baseline", "cooldown_seconds"}


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
        raise UsageError("sweep spec must be a JSON object")
    errors = ["unknown key %r" % k for k in sorted(set(spec) - SPEC_KEYS)]
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
    if not isinstance(spec.get("repeat_baseline", False), bool):
        errors.append("'repeat_baseline' must be true or false")
    if errors:
        raise UsageError("invalid sweep spec: " + "; ".join(errors))
    return {"name": name, "base": base, "entries": entries, "cooldown": cooldown,
            "repeat_baseline": bool(spec.get("repeat_baseline", False))}


def write_sweep_summary(cfg, spec, include, results, started):
    root = cfg["history_root"]
    sweeps = ensure_dir(ensure_dir(root) / "sweeps")
    name = allocate_dir(sweeps, "%s_%s" % (spec["name"], stamp()))
    rows = [load_run(root / "runs" / rid) for rid in include + [r["run_id"] for r in results]]
    comp = build_comparison(rows, resolve_columns(cfg, rows))
    summary = {"schema_version": SCHEMA_VERSION, "sweep": spec["name"], "suite": cfg["suite"], "started_at": started,
               "finished_at": iso_now(), "include_runs": include, "results": results, "comparison": comp}
    write_json(sweeps / name / "summary.json", summary)
    failed = [r["config"] for r in results if r["status"] != "success"]
    md = render_comparison_md(comp, "Sweep %s" % spec["name"])
    md += "\n## Results\n\n" + md_table(["config", "label", "run_id", "status", "error_code"],
                                       [[r[k] for k in ("config", "label", "run_id", "status", "error_code")] for r in results])
    write_atomic(sweeps / name / "summary.md", md + "\n\nFailed configs: %s\n" % (", ".join(failed) or "none"))
    say("sweep summary: %s" % (sweeps / name))


# ---------------------------------------------------------------- init templates

STUB_DRIVER = '''#!/usr/bin/env python3
"""Stub driver for bench.py: copies fixture 'input' to result.json["output"].

Config keys (all optional): drop_every=N drops every Nth word; fail="error" (result with
status error) | "crash" (exit 3, no result) | "killed" (exit 75) | "sleep:<seconds>";
fail_attempts=N makes only the first N invocations fail (counter file in the history root)."""
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

STUB_METRICS = '''#!/usr/bin/env python3
"""Stub metrics for bench.py: WER of result.json["output"] vs fixture 'reference' (one segment per line)."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(__TEXTDIFF_DIR__)
import textdiff

run_dir = Path(os.environ["BENCH_RUN_DIR"])
output = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))["output"]
reference = Path(os.environ["BENCH_FIXTURE_REFERENCE"]).read_text(encoding="utf-8")
segments = [{"id": str(n), "text": t} for n, t in enumerate(reference.splitlines(), 1) if t.strip()]
report = textdiff.build_report(segments, output)
s = report["summary"]
(run_dir / "metrics.json").write_text(json.dumps({k: s[k] for k in ("wer", "S", "D", "I", "N_ref", "N_hyp")}))
(run_dir / "metrics.md").write_text("WER %.2f %% (S %d, D %d, I %d)\\n" % (100 * s["wer"], s["S"], s["D"], s["I"]))
(run_dir / "diff.md").write_text(textdiff.render_markdown(report, "Diff"))
'''

STARTER_CONFIG = {
    "suite": "starter", "history_root": "history",
    "fixtures": {"input": "fixtures/input.txt", "reference": "fixtures/reference.txt"},
    "driver": ["python3", "bench_stub_driver.py"], "metrics": ["python3", "bench_stub_metrics.py"],
    "defaults": {"drop_every": 0}, "timeout_seconds": 600, "retries": 0,
    "index_columns": {"config": ["drop_every"], "metrics": ["wer", "S", "D", "I"]},
}


def cmd_init(args):
    target = Path(args.dir or ".")
    files = {
        "bench.json": json.dumps(STARTER_CONFIG, indent=2) + "\n",
        "bench_stub_driver.py": STUB_DRIVER,
        "bench_stub_metrics.py": STUB_METRICS.replace("__TEXTDIFF_DIR__", repr(str(Path(__file__).resolve().parent))),
        "fixtures/input.txt": "the quick brown fox jumps over the lazy dog\n",
        "fixtures/reference.txt": "the quick brown fox jumped over the lazy dog today\n",
    }
    existing = [n for n in files if (target / n).exists()]
    if existing:
        raise UsageError("refusing to overwrite: %s" % ", ".join(existing))
    (target / "fixtures").mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        write_atomic(target / name, text)
    say("wrote %s into %s\nTry: python3 %s run --label smoke --config %s\n"
        "history_root is ./history; move it outside the repo when fixtures or outputs are sensitive."
        % (", ".join(files), target, Path(__file__).name, target / "bench.json"))
    return 0


# ---------------------------------------------------------------- commands

def cmd_run(args):
    if not LABEL_RE.match(args.label):
        raise UsageError("--label must match %s" % LABEL_RE.pattern)
    cfg = load_config(args.config)
    record = execute_run(cfg, args.label, parse_sets(args.set), retries=args.retries, timeout=args.timeout_seconds,
                         driver=args.driver, dry_run=args.dry_run)
    return 0 if args.dry_run else EXIT_FOR_STATUS[record["status"]]


def cmd_metrics(args):
    return 0 if rescore(load_config(args.config), args.run, args.timeout_seconds) == "success" else 1


def cmd_index(args):
    cfg = load_config(args.config)
    rows = write_index(cfg)
    say("indexed %d run(s): %s" % (len(rows), cfg["history_root"] / "index.md"))
    return 0


def cmd_compare(args):
    cfg = load_config(args.config)
    rows = []
    for rid in args.run_ids:
        if not (cfg["history_root"] / "runs" / rid).is_dir():
            raise UsageError("unknown run id %r" % rid)
        rows.append(load_run(cfg["history_root"] / "runs" / rid))
    if args.columns:
        keys = [k for k in args.columns.split(",") if k]
        metric_keys = [k for k in keys if any(k in r["metrics"] for r in rows)]
        cols = make_columns([k for k in keys if k not in metric_keys], metric_keys)
    else:
        cols = resolve_columns(cfg, rows)
    comp = build_comparison(rows, cols)
    md = render_comparison_md(comp, "Comparison")
    say(md)
    if args.markdown_out:
        write_atomic(Path(args.markdown_out), md)
    if args.json_out:
        write_json(Path(args.json_out), comp)
    return 0


def sweep_result(name, label, rec):
    return {"config": name, "label": label, "run_id": rec["run_id"], "status": rec["status"],
            "error_code": rec.get("error_code")}


def cmd_sweep(args):
    cfg = load_config(args.config)
    spec = parse_sweep_spec(args.spec)
    include = args.include_run or []
    for rid in include:
        if not (cfg["history_root"] / "runs" / rid).is_dir():
            raise UsageError("unknown run id %r (--include-run)" % rid)
    plan = [(n, "%s__%s" % (spec["name"], n), {**spec["base"], **ov}) for n, ov in spec["entries"]]
    if spec["repeat_baseline"]:
        plan.append((plan[0][0] + "__repeat", plan[0][1] + "__repeat", plan[0][2]))
    started, results = iso_now(), []
    try:
        for idx, (name, label, overrides) in enumerate(plan):
            if idx and spec["cooldown"] and not args.dry_run:
                _sleep(spec["cooldown"])  # between runs only
            try:
                rec = execute_run(cfg, label, overrides, retries=args.retries, timeout=args.timeout_seconds,
                                  driver=args.driver, dry_run=args.dry_run)
            except Interrupted as exc:  # the interrupted run is part of the sweep's record
                results.append(sweep_result(name, label, exc.args[0]))
                raise
            results.append(sweep_result(name, label, rec))
    finally:  # also after Ctrl-C: keep what was measured so far
        if results and not args.dry_run:
            write_sweep_summary(cfg, spec, include, results, started)
    return 0 if args.dry_run or all(r["status"] == "success" for r in results) else 1


def build_parser():
    ap = argparse.ArgumentParser(prog="bench.py", description=__doc__.splitlines()[0], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True, metavar="{run,sweep,metrics,index,compare,init}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="bench.json", help="path to bench.json (default ./bench.json)")
    runopts = argparse.ArgumentParser(add_help=False)
    runopts.add_argument("--retries", type=int, help="extra attempts after a killed/busy attempt")
    runopts.add_argument("--timeout-seconds", type=float, help="per-command timeout (0 = none)")
    runopts.add_argument("--driver", help="override the driver command (shlex-split), e.g. a stub")
    runopts.add_argument("--dry-run", action="store_true", help="print what would run; create nothing")

    p = sub.add_parser("run", parents=[common, runopts], help="run one config end to end")
    p.add_argument("--label", required=True, help="letters, digits, . _ - (must start alphanumeric)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config key (JSON value or string)")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("sweep", parents=[common, runopts], help="run a sweep spec sequentially")
    p.add_argument("--spec", required=True, help="sweep spec JSON")
    p.add_argument("--include-run", action="append", metavar="RUN_ID", help="older run as a leading baseline row")
    p.set_defaults(func=cmd_sweep)
    p = sub.add_parser("metrics", aliases=["rescore"], parents=[common], help="re-run only the metrics command on a run")
    p.add_argument("--run", required=True, metavar="RUN_ID")
    p.add_argument("--timeout-seconds", type=float)
    p.set_defaults(func=cmd_metrics)
    p = sub.add_parser("index", parents=[common], help="rebuild index.json + index.md")
    p.set_defaults(func=cmd_index)
    p = sub.add_parser("compare", parents=[common], help="side-by-side runs with deltas vs the first")
    p.add_argument("run_ids", nargs="+", metavar="RUN_ID")
    p.add_argument("--columns", help="comma-separated config/metric keys (default: index_columns)")
    p.add_argument("--markdown-out")
    p.add_argument("--json-out")
    p.set_defaults(func=cmd_compare)
    p = sub.add_parser("init", help="write a starter bench.json + stub driver + stub metrics")
    p.add_argument("dir", nargs="?", default=".")
    p.set_defaults(func=cmd_init)
    return ap


def _on_sigterm(signum, frame):
    raise KeyboardInterrupt()


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.umask(0o077)
    signal.signal(signal.SIGINT, signal.default_int_handler)  # even when started with SIGINT ignored
    signal.signal(signal.SIGTERM, _on_sigterm)  # same cleanup path as Ctrl-C
    try:
        return args.func(args)
    except UsageError as exc:
        print("bench: error: %s" % exc, file=sys.stderr)
        return 2
    except (Interrupted, KeyboardInterrupt):
        print("bench: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
