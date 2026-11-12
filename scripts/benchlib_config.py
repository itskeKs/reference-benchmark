"""bench.json loading and validation, override keys and their types, fixture / provenance hashing (stdlib only)."""
from __future__ import annotations

import hashlib
import json
import operator
import os
import re
import shlex
import subprocess
from pathlib import Path

from benchlib_util import UsageError, did_you_mean, is_num, warn

SCHEMA_VERSION = 1
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
CONFIG_KEYS = (
    "suite", "history_root", "fixtures", "driver", "metrics", "defaults", "timeout_seconds", "retries", "guard",
    "guard_quiet_seconds", "guard_poll_seconds", "guard_max_wait_seconds", "index_columns", "provenance",
    "primary_metric", "criteria", "timing_keys", "holdout_split", "rank_split", "public_metrics", "allow_new_keys",
)
TUNING_SPLIT = "train"  # default rank_split: sweeps / criteria use `<rank_split>.<metric>` when it exists
IGNORED_NAMES = {"__pycache__", ".git", ".DS_Store"}  # never hashed (spurious churn)
MAX_SCRIPT_BYTES = 256 << 20
CRITERION_OPS = {"<=": operator.le, ">=": operator.ge, "==": operator.eq, "!=": operator.ne, "<": operator.lt, ">": operator.gt}
_CRITERION_RE = re.compile(r"^\s*(<=|>=|==|!=|<|>)\s*(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$")


# ---------------------------------------------------------------- hashing

def tree_files(path):
    if path.is_file():
        return [path]
    out = []
    for f in sorted(path.rglob("*")):
        rel = f.relative_to(path).parts
        if f.is_file() and not any(part in IGNORED_NAMES for part in rel) and f.suffix != ".pyc":
            out.append(f)
    return out


def path_info(path):
    """{path_name (basename only), bytes, sha256}; a directory hashes relative names + contents
    (ignoring __pycache__, .git, .DS_Store, *.pyc). Missing path -> bytes/sha256 None."""
    info = {"path_name": path.name, "bytes": None, "sha256": None}
    if not path.exists():
        return info
    digest, total = hashlib.sha256(), 0
    for f in tree_files(path):
        if path.is_dir():
            digest.update(str(f.relative_to(path)).encode() + b"\0")
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
                total += len(chunk)
    info.update(bytes=total, sha256=digest.hexdigest())
    return info


fixture_info = path_info


def fingerprint(fixtures):
    """Short combined sha (8 hex) over all fixture names + hashes; None without recorded fixtures."""
    if not isinstance(fixtures, dict) or not fixtures:
        return None
    digest = hashlib.sha256()
    for name in sorted(fixtures):
        info = fixtures[name] if isinstance(fixtures[name], dict) else {}
        digest.update(("%s\0%s\0" % (name, info.get("sha256"))).encode())
    return digest.hexdigest()[:8]


def _entries_fp(entries):
    if not isinstance(entries, list):
        return None
    digest = hashlib.sha256()
    names = [(str(e.get("path") or e.get("path_name")), e.get("sha256")) for e in entries if isinstance(e, dict)]
    for name, sha in sorted(names, key=str):
        digest.update(("%s\0%s\0" % (name, sha)).encode())
    return digest.hexdigest()[:8]


def code_identity(run):
    """{driver, metrics, paths}: short fingerprints (8 hex) of the code a run was produced and scored with, from
    run.json "provenance"; None for runs without provenance. `metrics` is the scorer of the CURRENT metrics, i.e.
    "rescored_provenance" wins over "provenance" after a re-score with a changed metrics script."""
    prov = run.get("provenance")
    if not isinstance(prov, dict):
        return None
    resc = run.get("rescored_provenance")
    metrics = resc["metrics"] if isinstance(resc, dict) and "metrics" in resc else prov.get("metrics")
    return {"driver": _entries_fp(prov.get("driver")), "metrics": _entries_fp(metrics), "paths": _entries_fp(prov.get("paths"))}


def fixtures_now(cfg):
    return {n: path_info(p) for n, p in cfg["fixtures"].items()}


def script_files(argv, base):
    """Hashes of argv elements that are existing files (driver / metrics scripts, binaries)."""
    out, seen = [], set()
    for arg in argv or []:
        try:
            p = Path(arg).expanduser()
            p = p if p.is_absolute() else Path(os.path.normpath(str(base / p)))
            if not p.is_file() or str(p) in seen:
                continue
        except (OSError, ValueError):
            continue
        seen.add(str(p))
        info = path_info(p) if p.stat().st_size <= MAX_SCRIPT_BYTES else {"path_name": p.name, "bytes": None, "sha256": None}
        out.append(info)
    return out


def provenance_info(cfg, driver):
    """Code identity for non-git setups: bench.json `provenance` paths + driver/metrics script hashes."""
    paths = []
    for shown, p in cfg["provenance"]:
        info = path_info(p)
        if info["sha256"] is None:
            raise UsageError("provenance path not found: %s" % shown)
        paths.append({"path": shown if not Path(shown).expanduser().is_absolute() else p.name, **info})
    return {"paths": paths, "driver": script_files(driver, cfg["dir"]), "metrics": script_files(cfg["metrics"], cfg["dir"])}


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


def parse_criteria(raw, where="criteria"):
    """{"wer": "<=0.2", "invented": "==0"} -> [{name, op, threshold, text}]"""
    if not isinstance(raw, dict):
        raise UsageError("%s must be an object like {\"wer\": \"<=0.2\"}" % where)
    out = []
    for name, text in raw.items():
        m = _CRITERION_RE.match(text) if isinstance(text, str) else None
        if not m:
            raise UsageError("%s[%r]: expected an operator (<=, <, >=, >, ==, !=) followed by a number, got %r" % (where, name, text))
        out.append({"name": str(name), "op": m.group(1), "threshold": float(m.group(2)), "text": "%s%s" % (m.group(1), m.group(2))})
    return out


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
    for key in sorted(set(raw) - set(CONFIG_KEYS)):
        warn("%s: unknown key %r ignored%s" % (path.name, key, did_you_mean(key, CONFIG_KEYS)))
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
    prov = raw.get("provenance") or []
    if not (isinstance(prov, list) and all(isinstance(x, str) and x for x in prov)):
        raise UsageError("%s: 'provenance' must be a list of paths" % path)
    pm = raw.get("primary_metric")
    if pm is not None:
        if not (isinstance(pm, dict) and isinstance(pm.get("name"), str) and pm.get("name") and pm.get("goal") in ("max", "min")
                and set(pm) <= {"name", "goal"}):
            raise UsageError('%s: \'primary_metric\' must be {"name": "<metric>", "goal": "max" | "min"}' % path)
        pm = {"name": pm["name"], "goal": pm["goal"]}
    timing = raw.get("timing_keys") or []
    if not (isinstance(timing, list) and all(isinstance(x, str) for x in timing)):
        raise UsageError("%s: 'timing_keys' must be a list of metric names" % path)
    holdout = raw.get("holdout_split")
    if holdout is not None and not (isinstance(holdout, str) and holdout):
        raise UsageError("%s: 'holdout_split' must be a split name" % path)
    rank = raw.get("rank_split")
    if rank is not None and not (isinstance(rank, str) and rank):
        raise UsageError("%s: 'rank_split' must be a split name" % path)
    if rank is not None and rank == holdout:
        raise UsageError("%s: 'rank_split' must differ from 'holdout_split' (tuning on the holdout defeats it)" % path)
    if pm and holdout and pm["name"].startswith(holdout + "."):
        raise UsageError("%s: 'primary_metric' %r belongs to the holdout split %r" % (path, pm["name"], holdout))
    public = raw.get("public_metrics") or []
    if not (isinstance(public, list) and all(isinstance(x, str) and x for x in public)):
        raise UsageError("%s: 'public_metrics' must be a list of metric names (glob patterns like \"dev.*\" allowed)" % path)
    if not isinstance(raw.get("allow_new_keys", False), bool):
        raise UsageError("%s: 'allow_new_keys' must be true or false" % path)
    return {
        "dir": base, "suite": str(raw.get("suite") or "bench"), "history_root": resolve(raw["history_root"]),
        "fixtures": {str(k): resolve(v) for k, v in fixtures.items()}, "defaults": defaults,
        "driver": as_argv(raw.get("driver"), "driver"), "metrics": as_argv(raw.get("metrics"), "metrics"),
        "guard": as_argv(raw.get("guard"), "guard"), "index_columns": index_columns,
        "timeout_seconds": number("timeout_seconds", 1800), "retries": int(number("retries", 0)),
        "guard_quiet_seconds": number("guard_quiet_seconds", 0), "guard_poll_seconds": number("guard_poll_seconds", 5),
        "guard_max_wait_seconds": number("guard_max_wait_seconds", 600),
        "provenance": [(p, resolve(p)) for p in prov], "primary_metric": pm,
        "criteria": parse_criteria(raw["criteria"], "criteria") if raw.get("criteria") else [],
        "timing_keys": timing, "holdout_split": holdout, "allow_new_keys": bool(raw.get("allow_new_keys", False)),
        "rank_split": rank or TUNING_SPLIT, "rank_split_set": rank is not None, "public_metrics": public,
    }


def tuning_split(cfg):
    """The split rankings and criteria are computed on (`<split>.<metric>`), or None when neither holdout_split nor
    rank_split is configured (then the plain metric names are used)."""
    return cfg["rank_split"] if cfg["holdout_split"] or cfg["rank_split_set"] else None


# ---------------------------------------------------------------- overrides

def parse_sets(items, defaults=None):
    """--set KEY=VALUE -> {KEY: JSON value, or the plain string when it is not JSON}. When the key's default in
    bench.json is a string, the raw text is always taken as the string (`--set model=3` stays "3")."""
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise UsageError("--set expects key=value, got %r" % item)
        try:
            out[key] = json.loads(value)
        except ValueError:
            out[key] = value
        if isinstance((defaults or {}).get(key), str) and not isinstance(out[key], str):
            out[key] = value
    return out


def parse_env(items):
    """--env KEY=VALUE (repeatable) -> {KEY: VALUE}; values are plain strings, never JSON-decoded."""
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not ENV_KEY_RE.match(key):
            raise UsageError("--env expects KEY=VALUE with KEY like [A-Za-z_][A-Za-z0-9_]*, got %r" % item)
        if key.startswith("BENCH_"):
            raise UsageError("--env: %s is reserved (BENCH_* variables are set by bench.py)" % key)
        out[key] = value
    return out


def type_name(v):
    return "null" if v is None else "bool" if isinstance(v, bool) else "int" if isinstance(v, int) else (
        "float" if isinstance(v, float) else "str" if isinstance(v, str) else "list" if isinstance(v, list) else "dict")


def type_matches(default, value):
    """Overrides must have the type of the value in bench.json `defaults` (bool / int / float / str / list / dict;
    an int is accepted where a float is expected; a null default accepts anything)."""
    if default is None:
        return True
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, int):
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(default, float):
        return is_num(value)
    return isinstance(value, type(default))


def override_type_errors(cfg, overrides):
    """['top_k: expected int (the type of its default 10), got str "ten"', ...] for keys that exist in defaults."""
    out = []
    for key, value in overrides.items():
        if key in cfg["defaults"] and not type_matches(cfg["defaults"][key], value):
            default = cfg["defaults"][key]
            hint = " (write true / false)" if isinstance(default, bool) else ""
            out.append("%s: expected %s (the type of its default %s), got %s %s%s" % (
                key, type_name(default), json.dumps(default), type_name(value), json.dumps(value), hint))
    return out


def check_override_types(cfg, overrides, where="--set"):
    errors = override_type_errors(cfg, overrides)
    if errors:
        raise UsageError("%s: wrong type for %s. Keys not in bench.json defaults (allow_new_keys) are untyped; a null "
                         "default accepts any type" % (where, "; ".join(errors)))


def check_override_keys(cfg, keys, allow=False, where="--set"):
    """Overrides must name keys that exist in bench.json `defaults` (catches typos that would silently
    become a new, ignored config key)."""
    if allow or cfg["allow_new_keys"]:
        return
    valid = sorted(cfg["defaults"])
    unknown = sorted(set(keys) - set(valid))
    if not unknown:
        return
    parts = []
    for k in unknown:
        parts.append("%r%s" % (k, did_you_mean(k, valid)))
    raise UsageError("%s: unknown config key(s) %s; valid keys (bench.json defaults): %s. "
                     "Add the key to defaults or pass --allow-new-keys" % (where, ", ".join(parts), ", ".join(valid) or "(none)"))
