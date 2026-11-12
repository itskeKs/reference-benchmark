"""Small bench-agnostic helpers shared by bench.py, its benchlib_* modules and calibrate.py.

Files, atomic writes, markdown tables, run-id stamps, BrokenPipe-safe output (stdlib only).
"""
from __future__ import annotations

import datetime as dt
import difflib
import itertools
import json
import os
import sys
from pathlib import Path


class UsageError(Exception):
    """Bad arguments or configuration (exit 2)."""


# ---------------------------------------------------------------- output

_stdout_closed = False


def close_stdout():
    """stdout went away (`| head`): silence it for good so the interpreter's exit flush stays quiet."""
    global _stdout_closed
    _stdout_closed = True
    try:
        fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(fd, sys.stdout.fileno())
        os.close(fd)
    except (OSError, ValueError):
        pass


def say(msg):
    """print() that survives a closed stdout: the work (runs, history) must finish regardless."""
    if _stdout_closed:
        return
    try:
        print(msg, flush=True)
    except BrokenPipeError:
        close_stdout()


def flush_quietly():
    """Flush stdout; a reader that went away (`| head`) is not an error."""
    try:
        sys.stdout.flush()
    except BrokenPipeError:
        close_stdout()


_warned = set()


def warn(msg):
    print("bench: warning: %s" % msg, file=sys.stderr)


def warn_once(msg):
    """warn(), but each distinct message only once per bench.py invocation (index rebuilds run after every run)."""
    if msg not in _warned:
        _warned.add(msg)
        warn(msg)


def reset_warnings():
    _warned.clear()


# ---------------------------------------------------------------- files

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def write_atomic(path, text):
    """Write text via tmp file + rename (mode 0600): readers see the old or the new file, never a torn one."""
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


def tail_lines(path, count):
    """Last `count` lines of a text file ([] when unreadable or empty); reads at most the last 64 KiB."""
    try:
        with open(str(path), "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 65536))
            data = fh.read()
    except OSError:
        return []
    return data.decode("utf-8", "replace").splitlines()[-count:]


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


# ---------------------------------------------------------------- time and numbers

def iso_now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp():
    """UTC timestamp with millisecond resolution, no dots: 20261112T105549123Z (ids sort in execution order)."""
    now = dt.datetime.now(dt.timezone.utc)
    return "%s%03dZ" % (now.strftime("%Y%m%dT%H%M%S"), now.microsecond // 1000)


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# ---------------------------------------------------------------- text and tables

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


def close_matches(word, choices):
    return difflib.get_close_matches(str(word), [str(c) for c in choices], n=3, cutoff=0.6)


def did_you_mean(word, choices):
    near = close_matches(word, choices)
    return " (did you mean %s?)" % " or ".join(repr(n) for n in near) if near else ""
