"""Tests for calibrate.py with synthetic data (plus one end-to-end pass through an init'ed items project)."""
import collections
import contextlib
import csv
import io
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
import calibrate  # noqa: E402
from benchlib_util import UsageError  # noqa: E402


def write_items(run_dir, scores, passes=None, detail=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    items = {}
    for iid, score in scores.items():
        items[iid] = {"split": "train", "score": score, "pass": (passes or {}).get(iid), "detail": (detail or {}).get(iid, {})}
    (run_dir / "items.json").write_text(json.dumps({"schema_version": 1, "items": items}))
    return run_dir


def cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = calibrate.main(list(argv))
    return rc, out.getvalue(), err.getvalue()


class CalibrateCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="calibrate-test-")
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def read_sheet(self, path):
        with open(str(path), newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))


class SheetTests(CalibrateCase):
    def setUp(self):
        super().setUp()
        self.a = write_items(self.tmp / "run_a", {"i1": 0.91, "i2": 0.52, "i3": 0.23, "i4": 0.64},
                             {"i1": True, "i2": True, "i3": False, "i4": True},
                             {"i1": {"actual": "ANSWER-1", "verdict": "good"}, "i2": {"actual": {"k": 2}}})
        self.b = write_items(self.tmp / "run_b", {"i1": 0.81, "i2": 0.47, "i3": 0.76, "i4": 0.17})
        self.sheet, self.key = self.tmp / "sheet.csv", self.tmp / "sheet.key.json"

    def make(self, *extra, seed="7"):
        rc, out, err = cli("sheet", "--runs", str(self.a), str(self.b), "--out", str(self.sheet), "--seed", seed, *extra)
        self.assertEqual(rc, 0, err)
        return out

    def test_sheet_is_blinded_shuffled_and_has_a_private_key(self):
        out = self.make()
        self.assertIn("private", out)
        rows = self.read_sheet(self.sheet)
        self.assertEqual(list(rows[0]), ["row_id", "item_id", "blinded_run", "human_score", "human_pass"])
        self.assertEqual(len(rows), 8)
        self.assertEqual({r["human_score"] for r in rows} | {r["human_pass"] for r in rows}, {""})
        text = self.sheet.read_text()
        for secret in ("run_a", "run_b", "0.91", "0.52", "0.17", "auto"):  # neither run names nor auto scores in the sheet
            self.assertNotIn(secret, text)
        self.assertEqual({r["blinded_run"] for r in rows}, {"R1", "R2"})
        self.assertEqual(len({r["row_id"] for r in rows}), 8)
        key = json.loads(self.key.read_text())
        self.assertEqual(sorted(key["runs"].values()), ["run_a", "run_b"])
        self.assertEqual(key["seed"], 7)
        for r in rows:  # every row maps back to its run, item and auto score
            k = key["rows"][r["row_id"]]
            self.assertEqual((k["item_id"], k["blinded_run"]), (r["item_id"], r["blinded_run"]))
            self.assertEqual(key["runs"][r["blinded_run"]], k["run"])
        auto = {(k["run"], k["item_id"]): k["auto_score"] for k in key["rows"].values()}
        self.assertEqual(auto[("run_a", "i1")], 0.91)
        self.assertEqual(auto[("run_b", "i4")], 0.17)
        self.assertIs([k for k in key["rows"].values() if k["run"] == "run_a" and k["item_id"] == "i3"][0]["auto_pass"], False)
        natural = sorted(((key["rows"][r["row_id"]]["run"], r["item_id"]) for r in rows))
        self.assertNotEqual([(key["rows"][r["row_id"]]["run"], r["item_id"]) for r in rows], natural)  # shuffled
        for p in (self.sheet, self.key):
            self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o600)
        self.assertEqual([p.name for p in self.tmp.glob(".*")], [])  # atomic writes leave no temp files

    def test_same_seed_reproduces_and_another_seed_differs(self):
        self.make()
        first = (self.sheet.read_text(), json.loads(self.key.read_text())["rows"])
        for p in (self.sheet, self.key):
            p.unlink()
        self.make()
        self.assertEqual((self.sheet.read_text(), json.loads(self.key.read_text())["rows"]), first)
        for p in (self.sheet, self.key):
            p.unlink()
        self.make(seed="8")
        self.assertNotEqual(self.sheet.read_text(), first[0])
        for p in (self.sheet, self.key):
            p.unlink()
        rc, _, _ = cli("sheet", "--runs", str(self.a), str(self.b), "--out", str(self.sheet))  # no seed: random, recorded
        self.assertEqual(rc, 0)
        self.assertIsInstance(json.loads(self.key.read_text())["seed"], int)

    def test_content_columns_and_refusals(self):
        self.make("--content-key", "actual", "--content-key", "missing")
        rows = self.read_sheet(self.sheet)
        self.assertEqual(list(rows[0])[3:5], ["content:actual", "content:missing"])
        key = json.loads(self.key.read_text())
        by = {(key["rows"][r["row_id"]]["run"], r["item_id"]): r for r in rows}
        self.assertEqual(by[("run_a", "i1")]["content:actual"], "ANSWER-1")
        self.assertEqual(json.loads(by[("run_a", "i2")]["content:actual"]), {"k": 2})
        self.assertEqual(by[("run_b", "i1")]["content:actual"], "")
        self.assertNotIn("good", self.sheet.read_text())  # only the requested detail keys are exported
        rc, _, err = cli("sheet", "--runs", str(self.a), "--out", str(self.sheet))  # refuses to overwrite
        self.assertEqual(rc, 2)
        self.assertIn("refusing to overwrite", err)
        empty = self.tmp / "no_items"
        empty.mkdir()
        rc, _, err = cli("sheet", "--runs", str(empty), "--out", str(self.tmp / "x.csv"))
        self.assertEqual(rc, 2)
        self.assertIn("no items.json", err)
        other = write_items(self.tmp / "other" / "run_a", {"i1": 1})
        rc, _, err = cli("sheet", "--runs", str(self.a), str(other), "--out", str(self.tmp / "y.csv"))
        self.assertEqual(rc, 2)
        self.assertIn("unique", err)
        self.assertFalse((self.tmp / "y.csv").exists())


def synthetic():
    """(key, sheet text, rows) with hand-computable statistics. rows: (run, item, auto, human, auto_pass, human_pass)"""
    rows = [("A", "i1", 0.9, 0.8, True, True), ("A", "i2", 0.5, 0.7, True, False), ("A", "i3", 0.2, 0.2, False, False),
            ("A", "i4", 0.6, 0.3, False, True), ("B", "i1", 0.8, 0.8, True, True), ("B", "i2", 0.4, 0.15, False, False),
            ("B", "i3", 0.7, 0.9, False, False), ("B", "i4", 0.1, 0.1, False, False)]
    key = {"rows": {}}
    lines = ["row_id,item_id,blinded_run,human_score,human_pass"]
    for n, (run, item, auto, human, ap, hp) in enumerate(rows):
        rid = "r%d" % n
        key["rows"][rid] = {"run": run, "item_id": item, "split": None, "auto_score": auto, "auto_pass": ap}
        lines.append("%s,%s,X,%s,%s" % (rid, item, human, "pass" if hp else "fail"))
    return key, "\n".join(lines) + "\n", rows


class ScoreTests(CalibrateCase):
    def test_item_level_statistics_match_a_hand_calculation(self):
        key, sheet, rows = synthetic()
        res = calibrate.score_sheet(sheet, key, tolerance=0.1)
        o = res["overall"]
        # errors (auto - human): .1 -.2 0 .3 0 .25 -.2 0
        self.assertEqual(o["n"], 8)
        self.assertAlmostEqual(o["mae"], 1.05 / 8)
        self.assertAlmostEqual(o["bias"], 0.25 / 8)
        self.assertAlmostEqual(o["max_abs_error"], 0.3)
        self.assertEqual(o["max_abs_error_unit"], "A/i4")
        self.assertAlmostEqual(o["within_0.1"], 0.5)  # .1, 0, 0, 0 are inside; the boundary counts
        xs, ys = [r[3] for r in rows], [r[2] for r in rows]  # human, auto
        mx, my = sum(xs) / 8, sum(ys) / 8
        expected_r = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(
            sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
        self.assertAlmostEqual(o["pearson_r"], expected_r)
        self.assertGreater(o["pearson_r"], 0.7)
        p = res["pass"]
        self.assertEqual((p["n"], p["tp"], p["fp"], p["fn"], p["tn"]), (8, 2, 1, 1, 4))
        self.assertAlmostEqual(p["agreement"], 0.75)
        self.assertAlmostEqual(p["kappa"], (0.75 - 34 / 64) / (1 - 34 / 64))
        self.assertEqual((res["rows_in_sheet"], res["rows_scored"], res["rows_skipped_no_human_score"]), (8, 8, 0))
        self.assertAlmostEqual(calibrate.score_sheet(sheet, key, tolerance=0.3)["overall"]["within_0.3"], 1.0)

    def test_run_level_unit_averages_per_run(self):
        key, sheet, _ = synthetic()
        res = calibrate.score_sheet(sheet, key, unit="run")
        o = res["overall"]
        self.assertEqual(o["n"], 2)
        # run A: auto mean .55 vs human .5 (+.05); run B: auto .5 vs human .4875 (+.0125)
        self.assertAlmostEqual(o["mae"], (0.05 + 0.0125) / 2)
        self.assertAlmostEqual(o["bias"], (0.05 + 0.0125) / 2)
        self.assertAlmostEqual(o["max_abs_error"], 0.05)
        self.assertEqual(o["max_abs_error_unit"], "A")
        self.assertIsNone(res["pass"])  # pass agreement is item-level only

    def test_missing_scores_and_degenerate_inputs(self):
        key, sheet, _ = synthetic()
        lines = sheet.splitlines()
        lines[1] = "r0,i1,X,,"  # rater skipped a row
        lines[2] = "r1,i2,X,\"0,7\",fail"  # decimal comma
        key["rows"]["r2"]["auto_score"] = None  # scorer gave no score
        res = calibrate.score_sheet("\n".join(lines) + "\n", key)
        self.assertEqual((res["rows_scored"], res["rows_skipped_no_human_score"], res["rows_without_auto_score"]), (7, 1, 1))
        self.assertEqual(res["overall"]["n"], 6)
        one = calibrate.score_sheet("row_id,human_score\nr0,0.5\n", key)
        self.assertEqual(one["overall"]["n"], 1)
        self.assertIsNone(one["overall"]["pearson_r"])  # undefined for a single point
        self.assertIsNone(calibrate.pearson([1, 1, 1], [1, 2, 3]))  # zero variance
        none = calibrate.score_sheet("row_id,human_score\nr0,\n", key)
        self.assertEqual(none["overall"]["n"], 0)
        self.assertIsNone(none["overall"]["mae"])
        self.assertIsNone(calibrate.pass_agreement([])["kappa"])

    def test_data_errors_are_usage_errors(self):
        key, sheet, _ = synthetic()
        for bad in ("row_id,human_score\nnope,0.5\n", "row_id,human_score\nr0,abc\n", "row_id,human_score\nr0,0.5\nr0,0.6\n",
                    "row_id,human_score,human_pass\nr0,0.5,maybe\n", "item_id,human_score\ni1,0.5\n", "row_id,human_score\nr0,nan\n"):
            with self.subTest(bad):
                with self.assertRaises(UsageError):
                    calibrate.score_sheet(bad, key)

    def test_folds_are_assigned_by_item_before_looking_at_scores(self):
        key, sheet, rows = synthetic()
        res = calibrate.score_sheet(sheet, key, folds=2, seed=3)
        folds = res["folds"]
        assignment = folds["assignment"]
        self.assertEqual(sorted(assignment), ["i1", "i2", "i3", "i4"])
        self.assertEqual(sorted(collections.Counter(assignment.values()).values()), [2, 2])  # balanced
        self.assertEqual(calibrate.score_sheet(sheet, key, folds=2, seed=3)["folds"]["assignment"], assignment)  # deterministic
        other_scores = sheet.replace(",0.8,", ",0.1,").replace(",0.9,", ",0.2,")
        self.assertEqual(calibrate.score_sheet(other_scores, key, folds=2, seed=3)["folds"]["assignment"], assignment)  # not score-driven
        self.assertEqual(sorted(calibrate.assign_folds({"i1", "i2", "i3", "i4"}, 2, 3).items()), sorted(assignment.items()))
        seeds = {tuple(sorted(calibrate.assign_folds({"i%d" % n for n in range(12)}, 3, s).items())) for s in range(6)}
        self.assertGreater(len(seeds), 1)  # the seed matters
        self.assertEqual(sum(f["stats"]["n"] for f in folds["results"]), res["overall"]["n"])
        for f in folds["results"]:  # every fold's MAE equals the MAE of exactly its items' rows
            errs = [abs(a - h) for run, item, a, h, _, _ in rows if assignment[item] == f["fold"]]
            self.assertEqual(f["stats"]["n"], len(errs))
            self.assertAlmostEqual(f["stats"]["mae"], sum(errs) / len(errs))
            self.assertEqual(f["items"], 2)
            self.assertIsNotNone(f["pass"])
        self.assertEqual(folds["k"], 2)
        for bad in (1, 5):
            with self.assertRaises(UsageError):
                calibrate.score_sheet(sheet, key, folds=bad)
        run_level = calibrate.score_sheet(sheet, key, unit="run", folds=2)
        self.assertEqual([f["stats"]["n"] for f in run_level["folds"]["results"]], [2, 2])  # per-run means inside each fold

    def test_cli_writes_markdown_and_json(self):
        key, sheet, _ = synthetic()
        (self.tmp / "k.json").write_text(json.dumps(key))
        (self.tmp / "s.csv").write_text(sheet)
        rc, out, err = cli("score", "--sheet", str(self.tmp / "s.csv"), "--key", str(self.tmp / "k.json"), "--folds", "2",
                           "--seed", "1", "--tolerance", "0.2", "--markdown-out", str(self.tmp / "r.md"), "--json-out", str(self.tmp / "r.json"))
        self.assertEqual(rc, 0, err)
        for text in ("MAE", "bias", "Pearson", "within ±0.2", "fold 0", "fold 1", "Pass agreement", "never from scores"):
            self.assertIn(text, out)
        self.assertEqual((self.tmp / "r.md").read_text().strip(), out.strip())
        res = json.loads((self.tmp / "r.json").read_text())
        self.assertEqual(res["tolerance"], 0.2)
        self.assertIn("within_0.2", res["overall"])
        self.assertEqual(stat.S_IMODE((self.tmp / "r.json").stat().st_mode), 0o600)
        self.assertEqual(cli("score", "--sheet", str(self.tmp / "nope.csv"), "--key", str(self.tmp / "k.json"))[0], 2)
        self.assertEqual(cli("score", "--sheet", str(self.tmp / "s.csv"), "--key", str(self.tmp / "nokey.json"))[0], 2)
        self.assertEqual(cli("score", "--sheet", str(self.tmp / "s.csv"), "--key", str(self.tmp / "k.json"), "--tolerance", "-1")[0], 2)


class EndToEndTests(CalibrateCase):
    def test_init_project_copy_of_calibrate_runs_sheet_and_score(self):
        r = subprocess.run([sys.executable, str(SCRIPTS / "bench.py"), "init", "proj", "--template", "items"], cwd=str(self.tmp),
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        proj = self.tmp / "proj"
        for label, extra in (("base", []), ("nodate", ["--set", "extract_date=false"])):
            done = subprocess.run([sys.executable, str(proj / "bench.py"), "run", "--label", label, *extra], cwd=str(proj),
                                  capture_output=True, text=True, timeout=60)
            self.assertEqual(done.returncode, 0, done.stderr)
        runs = sorted(p for p in (proj / "history" / "runs").iterdir())
        cal = lambda *a: subprocess.run([sys.executable, str(proj / "calibrate.py"), *a], cwd=str(proj), capture_output=True, text=True, timeout=60)
        done = cal("sheet", "--runs", *map(str, runs), "--out", str(self.tmp / "sheet.csv"), "--seed", "1", "--content-key", "actual")
        self.assertEqual(done.returncode, 0, done.stderr)
        rows = self.read_sheet(self.tmp / "sheet.csv")
        self.assertEqual(len(rows), 16)
        key = json.loads((self.tmp / "sheet.key.json").read_text())
        for row in rows:  # a perfect rater: human = auto, so the scorer agrees on everything
            row["human_score"] = key["rows"][row["row_id"]]["auto_score"]
            row["human_pass"] = "pass" if key["rows"][row["row_id"]]["auto_pass"] else "fail"
        with open(str(self.tmp / "filled.csv"), "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        done = cal("score", "--sheet", str(self.tmp / "filled.csv"), "--key", str(self.tmp / "sheet.key.json"), "--folds", "2")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("| overall | 16 | 0 | 0 | 0 |", done.stdout)
        self.assertEqual(cal("--help").returncode, 0)


if __name__ == "__main__":
    unittest.main()
