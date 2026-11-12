#!/usr/bin/env python3
"""Calibrate an automatic scorer against human scores (stdlib only, Python >= 3.9).

Reads the per-item results (items.json, see `bench.py -h`) of finished runs.

  calibrate.py sheet --runs RUN_DIR... --out sheet.csv [--key-out key.json] [--seed N] [--content-key KEY ...]
      Blinded hand-scoring sheet: one row per (run, item), shuffled, with an opaque row_id, item_id, a blinded
      run code (R1, R2, ... in random order) and empty human_score / human_pass columns to fill in. The automatic
      score and pass flag are NOT in the sheet; they go to a private key file (mode 0600) that maps every row_id
      back to its run and item. --content-key KEY adds detail[KEY] of items.json as a `content:KEY` column so the
      rater can see the output (use keys that hold raw output, never the automatic judgement).
  calibrate.py score --sheet filled.csv --key key.json [--unit item|run] [--tolerance 0.1]
                     [--folds K] [--seed N] [--markdown-out F] [--json-out F]
      Compares auto vs human: n, MAE, bias (auto - human; positive = the scorer is too generous), max |error|,
      share within ±tolerance, Pearson r, and agreement of the boolean pass (human_pass vs auto pass: agreement,
      confusion counts, Cohen's kappa; --unit item only). --unit item (default) treats every (run, item) row as a
      unit; --unit run first averages human and auto scores per run. Rows without a human_score are skipped.
      --folds K additionally reports every statistic per fold. Folds are assigned by ITEM from the sorted item ids
      in the key and --seed (default 0) only, never from any score: fix K and the seed before looking at the
      results, so thresholds tuned on K-1 folds can be checked on the held-out one without peeking.

Exit codes: 0 ok, 2 usage / data error. Needs benchlib_util.py and benchlib_items.py next to it (`bench.py init`
copies the whole toolkit).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import secrets
import sys
from pathlib import Path

from benchlib_items import load_items
from benchlib_util import UsageError, close_stdout, flush_quietly, iso_now, is_num, md_table, read_json, say, write_atomic, write_json

SHEET_FIXED = ("row_id", "item_id", "blinded_run")
SHEET_HUMAN = ("human_score", "human_pass")
_TRUE, _FALSE = {"1", "true", "yes", "y", "pass", "p", "ok"}, {"0", "false", "no", "n", "fail", "f"}


# ---------------------------------------------------------------- sheet

def build_sheet(run_dirs, seed=None, content_keys=()):
    """-> (csv header, csv rows, key dict). Deterministic for a given seed and run dir order."""
    seed = secrets.randbits(32) if seed is None else int(seed)
    rng = random.Random(seed)
    names = [Path(d).name for d in run_dirs]
    if len(set(names)) != len(names):
        raise UsageError("run directory names must be unique: %s" % ", ".join(names))
    entries = []
    for d, name in zip(run_dirs, names):
        items = load_items(Path(d))
        if items is None:
            raise UsageError("%s has no items.json" % d)
        entries += [(name, iid, item) for iid, item in items.items()]
    codes = ["R%d" % (n + 1) for n in range(len(names))]
    rng.shuffle(codes)
    blind = dict(zip(names, codes))
    rng.shuffle(entries)
    ids, rows, key_rows = set(), [], {}
    for name, iid, item in entries:
        row_id = "r%06x" % rng.getrandbits(24)
        while row_id in ids:
            row_id = "r%06x" % rng.getrandbits(24)
        ids.add(row_id)
        detail = item.get("detail") if isinstance(item.get("detail"), dict) else {}
        row = {"row_id": row_id, "item_id": iid, "blinded_run": blind[name], "human_score": "", "human_pass": ""}
        for k in content_keys:
            v = detail.get(k)
            row["content:%s" % k] = v if isinstance(v, str) else "" if v is None else json.dumps(v, ensure_ascii=False)
        rows.append(row)
        key_rows[row_id] = {"run": name, "blinded_run": blind[name], "item_id": iid, "split": item.get("split"),
                            "auto_score": item.get("score"), "auto_pass": item.get("pass")}
    header = list(SHEET_FIXED) + ["content:%s" % k for k in content_keys] + list(SHEET_HUMAN)
    key = {"schema_version": 1, "created_at": iso_now(), "seed": seed, "runs": {c: n for n, c in blind.items()}, "rows": key_rows}
    return header, rows, key


def sheet_csv(header, rows):
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def cmd_sheet(args):
    out = Path(args.out)
    key_out = Path(args.key_out) if args.key_out else out.with_name(out.stem + ".key.json")
    for p in (out, key_out):
        if p.exists():
            raise UsageError("refusing to overwrite %s" % p)
    header, rows, key = build_sheet(args.runs, args.seed, args.content_key or ())
    write_json(key_out, key)  # 0600: it unblinds the sheet
    write_atomic(out, sheet_csv(header, rows))
    say("sheet: %s (%d rows, runs blinded as %s)\nkey:   %s  (private, mode 0600: do not give it to the rater)"
        % (out, len(rows), ", ".join(sorted(key["runs"])), key_out))
    return 0


# ---------------------------------------------------------------- score

def parse_score(text, row_id):
    text = (text or "").strip().replace(",", ".")
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        raise UsageError("row %s: human_score %r is not a number" % (row_id, text))
    if not math.isfinite(value):
        raise UsageError("row %s: human_score %r is not finite" % (row_id, text))
    return value


def parse_bool(text, row_id):
    t = (text or "").strip().lower()
    if not t:
        return None
    if t in _TRUE:
        return True
    if t in _FALSE:
        return False
    raise UsageError("row %s: human_pass %r must be one of pass/fail, yes/no, true/false, 1/0" % (row_id, text))


def join_sheet(sheet_text, key):
    """Sheet rows + key -> list of units {row_id, run, item, split, auto, human, auto_pass, human_pass}."""
    reader = csv.DictReader(io.StringIO(sheet_text, newline=""))
    if not reader.fieldnames or "row_id" not in reader.fieldnames or "human_score" not in reader.fieldnames:
        raise UsageError("sheet needs the columns row_id and human_score")
    keyed, seen, out = key.get("rows") or {}, set(), []
    for row in reader:
        rid = (row.get("row_id") or "").strip()
        if rid not in keyed:
            raise UsageError("row_id %r is not in the key file (wrong key for this sheet?)" % rid)
        if rid in seen:
            raise UsageError("row_id %s appears twice in the sheet" % rid)
        seen.add(rid)
        k = keyed[rid]
        out.append({"row_id": rid, "run": k["run"], "item": k["item_id"], "split": k.get("split"),
                    "auto": k.get("auto_score") if is_num(k.get("auto_score")) else None,
                    "human": parse_score(row.get("human_score"), rid), "auto_pass": k.get("auto_pass"),
                    "human_pass": parse_bool(row.get("human_pass"), rid)})
    return out


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = math.fsum(xs) / n, math.fsum(ys) / n
    sxx, syy = math.fsum((x - mx) ** 2 for x in xs), math.fsum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def to_units(rows, unit):
    """Scored rows -> comparison units: every row (item), or one mean per run (run)."""
    scored = [r for r in rows if r["human"] is not None and r["auto"] is not None]
    if unit == "item":
        return [{"id": "%s/%s" % (r["run"], r["item"]), "human": r["human"], "auto": r["auto"]} for r in scored]
    runs = {}
    for r in scored:
        runs.setdefault(r["run"], []).append(r)
    return [{"id": run, "n_items": len(rs), "human": math.fsum(x["human"] for x in rs) / len(rs),
             "auto": math.fsum(x["auto"] for x in rs) / len(rs)} for run, rs in sorted(runs.items())]


def agreement(units, tolerance):
    n = len(units)
    if not n:
        return {"n": 0, "mae": None, "bias": None, "max_abs_error": None, "max_abs_error_unit": None,
                f"within_{tolerance:g}": None, "pearson_r": None, "human_mean": None, "auto_mean": None}
    errs = [u["auto"] - u["human"] for u in units]
    worst = max(range(n), key=lambda i: abs(errs[i]))
    return {"n": n, "mae": math.fsum(abs(e) for e in errs) / n, "bias": math.fsum(errs) / n,
            "max_abs_error": abs(errs[worst]), "max_abs_error_unit": units[worst]["id"],
            "within_%g" % tolerance: sum(1 for e in errs if abs(e) <= tolerance + 1e-12) / n,
            "pearson_r": pearson([u["human"] for u in units], [u["auto"] for u in units]),
            "human_mean": math.fsum(u["human"] for u in units) / n, "auto_mean": math.fsum(u["auto"] for u in units) / n}


def pass_agreement(rows):
    pairs = [(r["auto_pass"], r["human_pass"]) for r in rows if isinstance(r["auto_pass"], bool) and isinstance(r["human_pass"], bool)]
    n = len(pairs)
    if not n:
        return {"n": 0, "agreement": None, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "kappa": None}
    tp = sum(1 for a, h in pairs if a and h)
    fp = sum(1 for a, h in pairs if a and not h)
    fn = sum(1 for a, h in pairs if not a and h)
    tn = n - tp - fp - fn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    return {"n": n, "agreement": po, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "kappa": (po - pe) / (1 - pe) if pe != 1 else None}


def assign_folds(items, k, seed):
    """item id -> fold 0..k-1: sorted ids, shuffled with Random(seed), dealt round-robin. Depends on nothing else."""
    ids = sorted(set(items))
    if not 2 <= k <= len(ids):
        raise UsageError("--folds must be between 2 and the number of items (%d)" % len(ids))
    order = ids[:]
    random.Random("folds:%s" % seed).shuffle(order)
    return {iid: pos % k for pos, iid in enumerate(order)}


def score_sheet(sheet_text, key, *, unit="item", tolerance=0.1, folds=None, seed=0):
    rows = join_sheet(sheet_text, key)
    scored = [r for r in rows if r["human"] is not None]
    result = {"schema_version": 1, "unit": unit, "tolerance": tolerance, "rows_in_sheet": len(rows), "rows_scored": len(scored),
              "rows_skipped_no_human_score": len(rows) - len(scored),
              "rows_without_auto_score": sum(1 for r in scored if r["auto"] is None),
              "overall": agreement(to_units(rows, unit), tolerance),
              "pass": pass_agreement(rows) if unit == "item" else None}
    if folds:
        fold_of = assign_folds({k["item_id"] for k in key["rows"].values()}, folds, seed)
        result["folds"] = {"k": folds, "seed": seed, "assignment": dict(sorted(fold_of.items())), "results": []}
        for f in range(folds):
            part = [r for r in rows if fold_of[r["item"]] == f]
            result["folds"]["results"].append({"fold": f, "items": sum(1 for v in fold_of.values() if v == f),
                                               "stats": agreement(to_units(part, unit), tolerance),
                                               "pass": pass_agreement(part) if unit == "item" else None})
    return result


def _stat_row(label, st, tol):
    return [label, st["n"], st["mae"], st["bias"], st["max_abs_error"], st.get("within_%g" % tol), st["pearson_r"]]


def render_markdown(res):
    tol, hdr = res["tolerance"], ["scope", "n", "MAE", "bias", "max |err|", "within ±%g" % res["tolerance"], "Pearson r"]
    lines = ["# Calibration: auto vs human", "",
             "unit: %s | tolerance ±%g | rows in sheet %d, scored %d, skipped (no human_score) %d, scored without auto score %d" % (
                 res["unit"], tol, res["rows_in_sheet"], res["rows_scored"], res["rows_skipped_no_human_score"],
                 res["rows_without_auto_score"]),
             "bias = mean(auto - human): positive means the scorer is more generous than the human.", "",
             md_table(hdr, [_stat_row("overall", res["overall"], tol)] + (
                 [_stat_row("fold %d (%d items)" % (f["fold"], f["items"]), f["stats"], tol) for f in res["folds"]["results"]] if "folds" in res else []))]
    worst = res["overall"].get("max_abs_error_unit")
    if worst:
        lines += ["", "Largest overall error: `%s` (%.4g)." % (worst, res["overall"]["max_abs_error"])]
    if res["pass"] is not None:
        ph = ["scope", "n", "agreement", "kappa", "tp", "fp", "fn", "tn"]
        prow = lambda name, p: [name, p["n"], p["agreement"], p["kappa"], p["tp"], p["fp"], p["fn"], p["tn"]]
        lines += ["", "## Pass agreement (auto pass vs human_pass; tp = both pass)", "",
                  md_table(ph, [prow("overall", res["pass"])] + ([prow("fold %d" % f["fold"], f["pass"]) for f in res["folds"]["results"]] if "folds" in res else []))]
    if "folds" in res:
        lines += ["", "Folds: %d, assigned by item from the sorted item ids and seed %s only (never from scores)." % (res["folds"]["k"], res["folds"]["seed"])]
    return "\n".join(lines) + "\n"


def cmd_score(args):
    if args.tolerance < 0:
        raise UsageError("--tolerance must be >= 0")
    key = read_json(args.key)
    if not isinstance(key, dict) or not isinstance(key.get("rows"), dict):
        raise UsageError("cannot read key file %s" % args.key)
    try:
        sheet_text = Path(args.sheet).read_text(encoding="utf-8")
    except OSError as exc:
        raise UsageError("cannot read sheet: %s" % exc)
    res = score_sheet(sheet_text, key, unit=args.unit, tolerance=args.tolerance, folds=args.folds, seed=args.seed)
    md = render_markdown(res)
    say(md)
    if args.markdown_out:
        write_atomic(Path(args.markdown_out), md)
    if args.json_out:
        write_json(Path(args.json_out), res)
    return 0


def build_parser():
    ap = argparse.ArgumentParser(prog="calibrate.py", description=__doc__.splitlines()[0], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sheet", help="blinded hand-scoring sheet + private key from items.json of several runs")
    p.add_argument("--runs", nargs="+", required=True, metavar="RUN_DIR")
    p.add_argument("--out", required=True, metavar="SHEET.csv")
    p.add_argument("--key-out", metavar="KEY.json", help="default: <out>.key.json")
    p.add_argument("--seed", type=int, help="shuffle / blinding seed (default: random, recorded in the key)")
    p.add_argument("--content-key", action="append", metavar="KEY", help="add detail[KEY] as a content column (repeatable)")
    p.set_defaults(func=cmd_sheet)
    p = sub.add_parser("score", help="compare auto vs human scores")
    p.add_argument("--sheet", required=True, metavar="FILLED.csv")
    p.add_argument("--key", required=True, metavar="KEY.json")
    p.add_argument("--unit", choices=("item", "run"), default="item")
    p.add_argument("--tolerance", type=float, default=0.1, help="|auto - human| counted as agreement (default 0.1)")
    p.add_argument("--folds", type=int, metavar="K", help="also report per fold (folds by item, fixed by --seed)")
    p.add_argument("--seed", type=int, default=0, help="fold assignment seed (default 0)")
    p.add_argument("--markdown-out", metavar="FILE")
    p.add_argument("--json-out", metavar="FILE")
    p.set_defaults(func=cmd_score)
    return ap


def main(argv=None):
    os.umask(0o077)
    try:
        args = build_parser().parse_args(argv)
        return args.func(args)
    except UsageError as exc:
        print("calibrate: error: %s" % exc, file=sys.stderr)
        return 2
    except BrokenPipeError:
        close_stdout()
        return 0
    finally:
        flush_quietly()


if __name__ == "__main__":
    sys.exit(main())
