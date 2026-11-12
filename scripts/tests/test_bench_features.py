"""Tests for the provenance, usability, repeats, items/holdout and calibration-adjacent features of bench.py.

Most tests call bench.main() in-process (fast); signals, pipes and the init'ed project's own copy of bench.py
run as real subprocesses. Everything lives in temp dirs.
"""
import contextlib
import hashlib
import io
import json
import os
import shlex
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
import benchlib_config as cfglib  # noqa: E402
import benchlib_items as itemslib  # noqa: E402
import benchlib_runs as runslib  # noqa: E402
import benchlib_stats as statslib  # noqa: E402
import benchlib_util as util  # noqa: E402
from test_bench import BenchCase  # noqa: E402

# Metrics script whose output is dictated by the run's config: m (metrics), items (items.json), fail_metrics,
# rand (q becomes random), noise (t = wall clock); env SCORER is echoed as metrics["scorer"].
PARAM_SCORER = '''import json, os, random, time
d = os.environ["BENCH_RUN_DIR"]
cfg = json.load(open(os.environ["BENCH_CONFIG"]))
if cfg.get("fail_metrics"):
    raise SystemExit(7)
m = dict(cfg.get("m", {}))
if cfg.get("rand"):
    m["q"] = random.random()
if cfg.get("noise"):
    m["t"] = time.time()
if "SCORER" in os.environ:
    m["scorer"] = os.environ["SCORER"]
json.dump(m, open(d + "/metrics.json", "w"))
if "items" in cfg:
    json.dump(cfg["items"], open(d + "/items.json", "w"))
'''


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def items_doc(scores, splits):
    return {"schema_version": 1, "items": {
        k: {"split": splits.get(k), "score": v, "pass": None if v is None else v >= 0.5, "detail": {"secret": "CONTENT-" + k}}
        for k, v in scores.items()}}


class FeatureCase(BenchCase):
    def cli(self, command, *args, config=True):
        """In-process bench.main(); returns an object with returncode / stdout / stderr."""
        argv = [command] + (["--config", str(self.conf)] if config else []) + list(args)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rc = bench.main(argv)
            except SystemExit as exc:
                rc = exc.code
        return types.SimpleNamespace(returncode=rc, stdout=out.getvalue(), stderr=err.getvalue())

    def use_param_scorer(self, **config_changes):
        # the metrics come from run config key `m`, so show every metric (and no config column) in the tables
        changes = {"index_columns": {"config": []}, **config_changes}
        self.update_config(metrics=[sys.executable, self.script("param_scorer.py", PARAM_SCORER)], **changes)

    def set_args(self, **sets):
        out = []
        for k, v in sets.items():
            out += ["--set", "%s=%s" % (k, json.dumps(v))]
        return out

    def run_ok(self, label, **sets):
        r = self.cli("run", "--label", label, *self.set_args(**sets))
        self.assertEqual(r.returncode, 0, r.stderr)
        return self.run_for(label)

    def metrics_json(self, rid):
        return json.loads((self.hist / "runs" / rid / "metrics.json").read_text())

    def index_rows(self):
        return {r["label"]: r for r in json.loads((self.hist / "index.json").read_text())}

    def no_tmp_files(self):
        self.assertEqual([str(p) for p in self.hist.rglob("*") if p.name.endswith(".tmp")], [])


# ---------------------------------------------------------------- A. provenance

class ProvenanceTests(FeatureCase):
    REF_TEXT = "the quick brown fox jumped over the lazy dog today\n"

    def test_rescore_refuses_fixture_drift_and_never_overwrites_recorded_hashes(self):
        rid = self.run_ok("a")
        run_dir = self.hist / "runs" / rid
        original = self.run_json(rid)["fixtures"]
        old_metrics = self.metrics_json(rid)
        ref = self.proj / "fixtures" / "reference.txt"
        ref.write_text("a different reference text\n")

        r = self.cli("metrics", "--run", rid)
        self.assertEqual(r.returncode, 2)
        self.assertIn("--force", r.stderr)
        self.assertIn(rid, r.stderr)
        run = self.run_json(rid)
        self.assertEqual(run["fixtures"], original)
        self.assertNotIn("rescored_at", run)
        self.assertFalse((run_dir / "prev").exists())  # refused before anything was touched
        self.assertEqual(self.metrics_json(rid), old_metrics)

        r = self.cli("metrics", "--run", rid, "--force")
        self.assertEqual(r.returncode, 0, r.stderr)
        run = self.run_json(rid)
        self.assertEqual(run["fixtures"], original)  # still the hashes from the original run
        self.assertTrue(run["fixture_drift"])
        self.assertEqual(run["rescored_fixtures"]["reference"]["sha256"], sha256_of(ref))
        self.assertIn("rescored_at", run)
        self.assertNotEqual(self.metrics_json(rid), old_metrics)
        (prev,) = list((run_dir / "prev").iterdir())  # the old derived files + run.json were kept, not deleted
        self.assertEqual(json.loads((prev / "metrics.json").read_text()), old_metrics)
        self.assertTrue((prev / "metrics.log").exists() and (prev / "diff.md").exists())
        self.assertNotIn("rescored_at", json.loads((prev / "run.json").read_text()))
        self.assertRegex(self.index_rows()["a"]["fixtures"], r"^[0-9a-f]{8}>[0-9a-f]{8}$")

        ref.write_text(self.REF_TEXT)  # fixture restored: a plain re-score works and clears the drift mark
        self.assertEqual(self.cli("metrics", "--run", rid).returncode, 0)
        run = self.run_json(rid)
        self.assertNotIn("fixture_drift", run)
        self.assertNotIn("rescored_fixtures", run)
        self.assertEqual(len(list((run_dir / "prev").iterdir())), 2)
        self.assertRegex(self.index_rows()["a"]["fixtures"], r"^[0-9a-f]{8}$")
        self.assert_private(self.hist)
        self.no_tmp_files()

    def test_multi_run_drift_check_is_all_or_nothing(self):
        self.run_ok("a")
        self.run_ok("b")
        (self.proj / "fixtures" / "input.txt").write_text("changed input\n")
        r = self.cli("metrics", "--all")
        self.assertEqual(r.returncode, 2)
        self.assertEqual([self.run_json(i).get("rescored_at") for i in self.run_ids()], [None, None])

    def test_fixture_fingerprint_column_and_warnings(self):
        a = self.run_ok("a")
        (self.proj / "fixtures" / "input.txt").write_text("something else\n")
        b = self.run_ok("b")
        rows = self.index_rows()
        fa, fb = rows["a"]["fixtures"], rows["b"]["fixtures"]
        self.assertRegex(fa, r"^[0-9a-f]{8}$")
        self.assertNotEqual(fa, fb)
        self.assertEqual(self.run_json(a)["fixture_fingerprint"], fa)
        self.assertIn("| fixtures |", (self.hist / "index.md").read_text())
        out = self.tmp / "cmp.json"
        r = self.cli("compare", a, b, "--json-out", str(out))
        self.assertIn("WARNING: compared runs used different fixtures", r.stdout)
        self.assertIn(fa, r.stdout)
        self.assertEqual(len(json.loads(out.read_text())["warnings"]), 1)
        self.assertNotIn("WARNING", self.cli("compare", a, a).stdout)
        r = self.cli("sweep", "--spec", self.write_spec({"name": "sw", "configs": [{"name": "x"}]}), "--include-run", a)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("WARNING: compared runs used different fixtures", r.stdout)
        (summary_md,) = list((self.hist / "sweeps").glob("*/summary.md"))
        self.assertIn("WARNING", summary_md.read_text())

    def test_provenance_paths_and_command_scripts_are_hashed(self):
        sut = self.proj / "sut"
        (sut / "__pycache__").mkdir(parents=True)
        (sut / "model.py").write_text("print('v1')\n")
        (sut / "__pycache__" / "model.pyc").write_bytes(b"cache")
        self.update_config(provenance=["sut"])
        a = self.run_ok("a")
        prov = self.run_json(a)["provenance"]
        (entry,) = prov["paths"]
        self.assertEqual((entry["path"], entry["path_name"]), ("sut", "sut"))
        self.assertEqual(entry["sha256"], cfglib.path_info(sut)["sha256"])
        driver = [d for d in prov["driver"] if d["path_name"] == "bench_stub_driver.py"]
        metrics = [d for d in prov["metrics"] if d["path_name"] == "bench_stub_metrics.py"]
        self.assertEqual(driver[0]["sha256"], sha256_of(self.proj / "bench_stub_driver.py"))
        self.assertEqual(metrics[0]["sha256"], sha256_of(self.proj / "bench_stub_metrics.py"))
        (sut / "__pycache__" / "model.pyc").write_bytes(b"other cache")  # ignored: no spurious churn
        self.assertEqual(cfglib.path_info(sut)["sha256"], entry["sha256"])
        (sut / "model.py").write_text("print('v2')\n")
        b = self.run_ok("b")
        self.assertNotEqual(self.run_json(b)["provenance"]["paths"][0]["sha256"], entry["sha256"])
        self.assertNotIn(str(self.proj), (self.hist / "runs" / a / "run.json").read_text())
        before = self.run_ids()
        self.update_config(provenance=["missing-dir"])
        r = self.cli("run", "--label", "c")
        self.assertEqual(r.returncode, 2)
        self.assertIn("provenance path not found", r.stderr)
        self.assertEqual(self.run_ids(), before)

    def test_unknown_config_keys_are_warned_about(self):
        self.update_config(primary_metrc={"name": "wer", "goal": "min"})
        r = self.cli("index")
        self.assertEqual(r.returncode, 0)
        self.assertIn("unknown key 'primary_metrc'", r.stderr)
        self.assertIn("primary_metric", r.stderr)


# ---------------------------------------------------------------- B. usability

class OverrideKeyTests(FeatureCase):
    def setUp(self):
        super().setUp()
        self.update_config(allow_new_keys=False)

    def test_set_with_unknown_key_lists_valid_keys_and_close_matches(self):
        r = self.cli("run", "--label", "typo", "--set", "drop_evry=2")
        self.assertEqual(r.returncode, 2)
        self.assertIn("drop_evry", r.stderr)
        self.assertIn("drop_every", r.stderr)
        self.assertIn("did you mean", r.stderr)
        self.assertEqual(self.run_ids(), [])
        r = self.cli("run", "--label", "ok", "--set", "drop_every=2")
        self.assertEqual(r.returncode, 0)
        r = self.cli("run", "--label", "new", "--set", "brand_new=1", "--allow-new-keys")
        self.assertEqual(r.returncode, 0)
        self.assertEqual(self.run_json(self.run_for("new"))["config"]["brand_new"], 1)

    def test_sweep_keys_are_checked_up_front(self):
        for where, spec in {
            "base": {"name": "s", "base": {"nope": 1}, "configs": [{"name": "a"}]},
            "config": {"name": "s", "configs": [{"name": "a"}, {"name": "b", "drop_evry": 2}]},
            "matrix": {"name": "s", "matrix": {"drop_evry": [1, 2]}},
        }.items():
            with self.subTest(where):
                r = self.cli("sweep", "--spec", self.write_spec(spec))
                self.assertEqual(r.returncode, 2)
                self.assertIn("valid keys", r.stderr)
                self.assertFalse(self.hist.exists())
        ok = {"name": "s", "base": {"nope": 1}, "configs": [{"name": "a"}], "allow_new_keys": True}
        self.assertEqual(self.cli("sweep", "--spec", self.write_spec(ok)).returncode, 0)
        flag = {"name": "t", "configs": [{"name": "a", "other": 1}]}
        self.assertEqual(self.cli("sweep", "--spec", self.write_spec(flag, "t.json"), "--allow-new-keys").returncode, 0)

    def test_bench_json_allow_new_keys(self):
        self.update_config(allow_new_keys=True)
        self.assertEqual(self.cli("run", "--label", "x", "--set", "anything=1").returncode, 0)

    def test_close_matches_helper(self):
        self.assertIn("did you mean 'alpha'", util.did_you_mean("alpah", ["alpha", "beta"]))
        self.assertEqual(util.did_you_mean("zzz", ["alpha"]), "")


class FailureOutputTests(FeatureCase):
    def test_driver_failure_prints_log_path_and_last_15_lines(self):
        drv = self.script("noisy.py", "import sys\nfor i in range(20):\n    print('line%02d' % i, file=sys.stderr)\nsys.exit(3)\n")
        r = self.cli("run", "--label", "noisy", "--driver", "%s %s" % (sys.executable, drv))
        self.assertEqual(r.returncode, 1)
        log = self.hist / "runs" / self.only_run() / "driver.log"
        self.assertIn(str(log), r.stderr)
        self.assertIn("line19", r.stderr)
        self.assertIn("line05", r.stderr)
        self.assertNotIn("line04", r.stderr)  # only the last 15

    def test_metrics_failure_prints_metrics_log(self):
        self.update_config(metrics=[sys.executable, "-c", "import sys; print('boom from scorer'); sys.exit(7)"])
        r = self.cli("run", "--label", "m")
        self.assertEqual(r.returncode, 1)
        self.assertIn("metrics.log", r.stderr)
        self.assertIn("boom from scorer", r.stderr)
        r = self.cli("run", "--label", "ok", "--driver", "%s -c pass" % sys.executable)  # no result.json: driver.log is empty
        self.assertIn("(log is empty)", r.stderr)


class RunIdTests(FeatureCase):
    def test_millisecond_stamp_without_dots_sorts_in_execution_order(self):
        self.assertRegex(util.stamp(), r"^\d{8}T\d{9}Z$")
        self.run_ok("zzz")
        self.run_ok("aaa")
        ids = self.run_ids()
        self.assertEqual([self.run_json(i)["label"] for i in ids], ["zzz", "aaa"])  # not alphabetical by label
        self.assertTrue(all("." not in i.split("_")[0] for i in ids))

    def test_references_resolve_by_id_prefix_and_label(self):
        self.cli("run", "--label", "dup")
        self.cli("run", "--label", "dup")
        self.cli("run", "--label", "uniq")
        uniq = self.run_for("uniq")
        d1, d2 = [i for i in self.run_ids() if i != uniq]
        for token in (uniq, uniq.split("_")[0], "uniq"):  # exact id, unique prefix, unique label
            out = self.tmp / "c.json"
            self.assertEqual(self.cli("compare", token, "--json-out", str(out)).returncode, 0, token)
            self.assertEqual(json.loads(out.read_text())["baseline"], uniq)
        r = self.cli("compare", "dup")
        self.assertEqual(r.returncode, 2)
        self.assertIn("ambiguous", r.stderr)
        self.assertIn(d1, r.stderr)
        self.assertIn(d2, r.stderr)
        self.assertIn("--latest", r.stderr)
        out = self.tmp / "l.json"
        self.assertEqual(self.cli("compare", "dup", "--latest", "--json-out", str(out)).returncode, 0)
        self.assertEqual(json.loads(out.read_text())["baseline"], d2)
        r = self.cli("compare", "20")  # a prefix shared by every run
        self.assertEqual(r.returncode, 2)
        self.assertIn("ambiguous", r.stderr)
        r = self.cli("compare", "uniqq")
        self.assertEqual(r.returncode, 2)
        self.assertIn("unknown run id", r.stderr)
        self.assertIn("uniq", r.stderr)  # similar label suggested
        # metrics and --include-run accept the same references
        self.assertEqual(self.cli("metrics", "uniq").returncode, 0)
        self.assertEqual(self.cli("metrics", "dup").returncode, 2)
        self.assertEqual(self.cli("metrics", "dup", "--latest").returncode, 0)
        r = self.cli("sweep", "--spec", self.write_spec({"name": "s", "configs": [{"name": "a"}]}), "--include-run", "uniq")
        self.assertEqual(r.returncode, 0, r.stderr)
        summary = json.loads(next((self.hist / "sweeps").glob("*/summary.json")).read_text())
        self.assertEqual(summary["include_runs"], [uniq])


class MetricsSelectionTests(FeatureCase):
    def test_all_glob_and_explicit_ids_with_status_lines_and_summary(self):
        self.use_param_scorer()
        for label in ("x-1", "x-2", "y"):
            self.run_ok(label, m={"v": 1})
        r = self.cli("metrics", "--all")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(sum(1 for l in r.stdout.splitlines() if l.endswith("  success")), 3)
        self.assertIn("re-scored 3 run(s): 3 success, 0 metrics_failed, 0 skipped", r.stdout)
        for rid in self.run_ids():
            self.assertEqual(len(list((self.hist / "runs" / rid / "prev").iterdir())), 1)
        r = self.cli("metrics", "--label-glob", "x-*")
        self.assertIn("re-scored 2 run(s)", r.stdout)
        r = self.cli("metrics", "x-1", "y", "--run", "x-2")
        self.assertIn("re-scored 3 run(s)", r.stdout)
        r = self.cli("metrics", "--label-glob", "none-*")
        self.assertEqual(r.returncode, 2)
        self.assertEqual(self.cli("metrics").returncode, 2)  # nothing selected
        self.assert_private(self.hist)

    def test_failures_do_not_stop_the_others_and_unscorable_runs_are_skipped(self):
        self.use_param_scorer()
        self.run_ok("ok1", m={"v": 1})
        self.cli("run", "--label", "bad", *self.set_args(fail_metrics=True))
        self.run_ok("ok2", m={"v": 2})
        self.cli("run", "--label", "sysfail", "--set", "fail=error")
        r = self.cli("metrics", "--all")
        self.assertEqual(r.returncode, 1)
        self.assertIn("re-scored 3 run(s): 2 success, 1 metrics_failed, 1 skipped", r.stdout)
        self.assertIn("metrics_exit_7", r.stdout)
        self.assertEqual(self.run_json(self.run_for("ok2"))["status"], "success")
        self.assertEqual(self.run_json(self.run_for("bad"))["status"], "metrics_failed")
        self.assertEqual(self.cli("metrics", "sysfail").returncode, 2)  # explicitly named but unscorable

    def test_sweep_selector_uses_the_sweep_summary(self):
        self.run_ok("other")
        r = self.cli("sweep", "--spec", self.write_spec({"name": "ms", "configs": [{"name": "a"}, {"name": "b", "drop_every": 2}]}))
        self.assertEqual(r.returncode, 0, r.stderr)
        r = self.cli("metrics", "--sweep", "ms")
        self.assertIn("re-scored 2 run(s)", r.stdout)
        rescored = {self.run_json(i)["label"] for i in self.run_ids() if "rescored_at" in self.run_json(i)}
        self.assertEqual(rescored, {"ms__a", "ms__b"})
        self.assertEqual(self.cli("metrics", "--sweep", "nosuch").returncode, 2)

    def test_rescore_with_env_for_scorer_ab_tests(self):
        self.use_param_scorer()
        rid = self.run_ok("a", m={"v": 1})
        r = self.cli("metrics", "--run", rid, "--env", "SCORER=v2", "--env", "OTHER=x")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.metrics_json(rid)["scorer"], "v2")
        self.assertEqual(self.run_json(rid)["metrics_env"], {"SCORER": "v2", "OTHER": "x"})
        (prev,) = list((self.hist / "runs" / rid / "prev").iterdir())
        self.assertNotIn("scorer", json.loads((prev / "metrics.json").read_text()))
        self.assertEqual(self.cli("metrics", "--run", rid).returncode, 0)  # a plain re-score forgets it
        self.assertNotIn("scorer", self.metrics_json(rid))
        self.assertNotIn("metrics_env", self.run_json(rid))
        for bad in ("NOEQUALS", "1BAD=x", "BENCH_RUN_DIR=/x"):
            self.assertEqual(self.cli("metrics", "--run", rid, "--env", bad).returncode, 2, bad)


class CriteriaTests(FeatureCase):
    def test_pass_column_and_recompute_on_rescore(self):
        self.use_param_scorer(criteria={"f1": ">=0.8", "invented": "==0", "ratio": "<1.5"})
        good = self.run_ok("good", m={"f1": 0.9, "invented": 0, "ratio": 1.0})
        bad = self.run_ok("bad", m={"f1": 0.9, "invented": 2, "ratio": 1.0})
        unknown = self.run_ok("unknown", m={"f1": 0.9, "invented": 0})
        self.cli("run", "--label", "err", "--set", "fail=error")
        cr = self.run_json(good)["criteria_result"]
        self.assertTrue(cr["pass"])
        self.assertEqual(cr["details"]["f1"], {"criterion": ">=0.8", "value": 0.9, "pass": True})
        self.assertFalse(self.run_json(bad)["criteria_result"]["pass"])
        self.assertEqual(self.run_json(bad)["criteria_result"]["details"]["invented"]["pass"], False)
        self.assertIsNone(self.run_json(unknown)["criteria_result"]["pass"])  # metric missing -> null, not false
        self.assertIsNone(self.run_json(unknown)["criteria_result"]["details"]["ratio"]["pass"])
        self.assertIsNone(self.run_json(self.run_for("err"))["criteria_result"]["pass"])
        rows = self.index_rows()
        self.assertEqual([rows[l]["pass"] for l in ("good", "bad", "unknown", "err")], [True, False, None, None])
        header = next(l for l in (self.hist / "index.md").read_text().splitlines() if l.startswith("| run_id"))
        self.assertIn("| pass |", header)
        cmp_out = self.cli("compare", good, bad, unknown).stdout
        self.assertIn("| pass |", cmp_out)
        self.assertIn("FAIL", cmp_out)
        self.update_config(criteria={"f1": ">=0.95"})  # stricter criteria, recomputed by a re-score
        self.assertEqual(self.cli("metrics", "--run", good).returncode, 0)
        self.assertFalse(self.run_json(good)["criteria_result"]["pass"])
        self.update_config(criteria=None)
        self.cli("metrics", "--all")
        self.assertNotIn("criteria_result", self.run_json(good))

    def test_operators_and_validation(self):
        crit = cfglib.parse_criteria({"a": "<=1", "b": "<1", "c": ">= 2", "d": ">2", "e": "==0", "f": "!=3", "g.h": "<=-0.5e1"})
        self.assertEqual([c["op"] for c in crit], ["<=", "<", ">=", ">", "==", "!=", "<="])
        ev = lambda m: runslib.evaluate_criteria(crit, m)["details"]
        d = ev({"a": 1, "b": 1, "c": 2, "d": 2, "e": 0, "f": 3, "g.h": -5})
        self.assertEqual([v["pass"] for v in d.values()], [True, False, True, False, True, False, True])
        self.assertIsNone(runslib.evaluate_criteria([], {"a": 1}))
        self.assertIsNone(ev({"a": "text", "b": True})["a"]["pass"])  # non-numeric and bools do not count as values
        self.assertIsNone(ev({"b": True})["b"]["pass"])
        for bad in ({"a": "<>1"}, {"a": "<=x"}, {"a": 1}, ["a"]):
            with self.assertRaises(util.UsageError):
                cfglib.parse_criteria(bad)
        self.update_config(criteria={"wer": "<>0.2"})
        self.assertEqual(self.cli("index").returncode, 2)

    def test_primary_metric_validation(self):
        for bad in ({"name": "wer"}, {"name": "wer", "goal": "up"}, "wer", {"name": "wer", "goal": "min", "x": 1}):
            self.update_config(primary_metric=bad)
            self.assertEqual(self.cli("index").returncode, 2, bad)


class CompareTests2(FeatureCase):
    def setUp(self):
        super().setUp()
        self.a = self.run_ok("a")
        self.b = self.run_ok("b", drop_every=3)
        self.c = self.run_ok("c", drop_every=2)

    def compare_json(self, *args):
        out = self.tmp / ("cmp-%d.json" % len(list(self.tmp.glob("cmp-*"))))
        r = self.cli("compare", *args, "--json-out", str(out))
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(out.read_text()), r

    def test_primary_metric_ranks_rows_and_deltas_stay_vs_the_first_listed(self):
        comp, r = self.compare_json(self.c, self.a, self.b)  # wer: a 0.2 < b 0.5 < c 0.6; goal min
        self.assertEqual([row["run_id"] for row in comp["rows"]], [self.a, self.b, self.c])
        self.assertEqual([row["rank"] for row in comp["rows"]], [1, 2, 3])
        self.assertEqual(comp["baseline"], self.c)
        self.assertAlmostEqual(comp["rows"][0]["deltas"]["wer"], 0.2 - 0.6)
        self.assertIsNone(comp["rows"][2]["deltas"]["wer"])
        self.assertIn("1 (best)", r.stdout)
        self.assertIn("Ranked by `wer` (lowest first)", r.stdout)

    def test_sort_direction_top_and_no_sort(self):
        comp, _ = self.compare_json(self.a, self.b, self.c, "--sort", "wer", "--desc")
        self.assertEqual([r["run_id"] for r in comp["rows"]], [self.c, self.b, self.a])
        comp, _ = self.compare_json(self.c, self.b, self.a, "--sort", "wer", "--top", "2")  # primary metric: goal direction (min)
        self.assertEqual([r["run_id"] for r in comp["rows"]], [self.a, self.b])
        self.assertEqual((comp["shown"], comp["total"]), (2, 3))
        comp, r = self.compare_json(self.c, self.a, "--sort", "wer", "--asc", "--top", "1")
        self.assertEqual([r["run_id"] for r in comp["rows"]], [self.a])
        self.assertIn("top 1 of 2", r.stdout)
        comp, _ = self.compare_json(self.c, self.a, self.b, "--no-sort")
        self.assertEqual([r["run_id"] for r in comp["rows"]], [self.c, self.a, self.b])
        self.assertTrue(all(r["rank"] is None for r in comp["rows"]))
        comp, r = self.compare_json(self.a, self.b, "--sort", "S", "--asc")  # S is equal: tied
        self.assertEqual([r["rank"] for r in comp["rows"]], [1, 1])
        self.assertEqual(comp["rows"][0]["tied_with"], ["b"])
        self.assertIn("a tied with b", r.stdout)
        self.assertEqual(self.cli("compare", self.a, self.b, "--sort", "nosuch").returncode, 2)
        self.assertEqual(self.cli("compare", self.a, self.b, "--asc", "--desc").returncode, 2)

    def test_error_code_and_columns_selection(self):
        self.cli("run", "--label", "err", "--set", "fail=error")
        err = self.run_for("err")
        r = self.cli("compare", self.a, err)
        self.assertIn("error_code", r.stdout)
        self.assertIn("stub_error", r.stdout)
        index_header = next(l for l in (self.hist / "index.md").read_text().splitlines() if l.startswith("| run_id"))
        self.assertIn("error_code", index_header)
        r = self.cli("compare", self.a, self.b, "--columns", "wer,bogus,nope")
        self.assertEqual(r.returncode, 2)
        self.assertIn("'bogus'", r.stderr)
        self.assertIn("'nope'", r.stderr)
        self.assertIn("available", r.stderr)
        self.assertEqual(self.cli("compare", self.a, self.b, "--columns", "wer,drop_every,status,pass").returncode, 0)

    def test_key_in_config_and_metrics_gives_two_selectable_columns(self):
        d = self.run_ok("d", wer=9)
        for spec, expected in (("wer", ["config.wer", "metrics.wer"]), ("config.wer", ["config.wer"]),
                               ("metrics.wer", ["metrics.wer"])):
            comp, _ = self.compare_json(self.a, d, "--columns", spec, "--no-sort")
            self.assertEqual(comp["columns"], expected, spec)
        row = comp["rows"][0] if comp["rows"][0]["run_id"] == d else comp["rows"][1]
        self.assertEqual(row["values"]["metrics.wer"], self.metrics_json(d)["wer"])

    def test_compact_labels_and_wide(self):
        spec = {"name": "sw", "configs": [{"name": "one"}, {"name": "two", "drop_every": 2}]}
        self.assertEqual(self.cli("sweep", "--spec", self.write_spec(spec)).returncode, 0)
        (summary_json,) = list((self.hist / "sweeps").glob("*/summary.json"))
        md = summary_json.with_name("summary.md").read_text()
        ids = [r["run_id"] for r in json.loads(summary_json.read_text())["results"]]
        tables = "\n".join(l for l in md.splitlines() if l.startswith("|"))
        self.assertNotIn("sw__", tables)  # no sweep prefix in any table cell
        self.assertTrue(all(i not in tables for i in ids))  # compact: run ids omitted from the tables
        self.assertIn("| one |", md)
        self.assertIn(ids[0], summary_json.read_text())  # the json keeps the full ids
        self.assertEqual(self.cli("sweep", "--spec", self.write_spec(spec, "w.json"), "--wide").returncode, 0)
        wide_md = sorted((self.hist / "sweeps").glob("*/summary.md"))[-1].read_text()
        self.assertIn("run_id", wide_md)
        self.assertIn("sw__", wide_md)  # --wide: ids (which contain the label) are shown
        sw_ids = [i for i in self.run_ids() if "__" in i][:2]
        r = self.cli("compare", *sw_ids)
        self.assertNotIn("sw__", "\n".join(l for l in r.stdout.splitlines() if l.startswith("|")))
        self.assertIn("without the `sw__` prefix", r.stdout)
        r = self.cli("compare", *sw_ids, "--wide")
        self.assertIn("| run_id |", r.stdout)
        self.assertIn(sw_ids[0], r.stdout)


# ---------------------------------------------------------------- sweeps

class SweepFeatureTests(FeatureCase):
    def sweep(self, spec, *extra):
        r = self.cli("sweep", "--spec", self.write_spec(spec), *extra)
        return r

    def summary(self, index=-1):
        d = sorted((self.hist / "sweeps").iterdir())[index]
        return json.loads((d / "summary.json").read_text()), (d / "summary.md").read_text()

    def test_ranking_best_marker_and_ties(self):
        spec = {"name": "rk", "configs": [{"name": "base"}, {"name": "twin", "marker": 1}, {"name": "worse", "drop_every": 3}]}
        r = self.sweep(spec)
        self.assertEqual(r.returncode, 0, r.stderr)
        summary, md = self.summary()
        rows = summary["comparison"]["rows"]
        self.assertEqual([row["rank"] for row in rows], [1, 1, 3])
        self.assertEqual(summary["ranking"]["key"], "wer")
        self.assertIn("1 (best, tied)", md)
        self.assertIn("base tied with twin", md)
        self.assertIn("best by wer: base", r.stdout)
        self.assertEqual([row["label"] for row in rows][2], "rk__worse")

    def test_baseline_key_drives_deltas_and_repeat_baseline(self):
        spec = {"name": "bl", "baseline": "b", "repeat_baseline": True,
                "configs": [{"name": "a"}, {"name": "b", "drop_every": 3}, {"name": "c", "drop_every": 2}]}
        self.assertEqual(self.sweep(spec).returncode, 0)
        summary, md = self.summary()
        b_id = next(x["run_id"] for x in summary["results"] if x["config"] == "b")
        self.assertEqual(summary["comparison"]["baseline"], b_id)
        self.assertEqual(summary["baseline_config"], "b")
        by = {r["label"]: r for r in summary["comparison"]["rows"]}
        self.assertIsNone(by["bl__b"]["deltas"]["wer"])
        self.assertLess(by["bl__a"]["deltas"]["wer"], 0)
        self.assertEqual([x["label"] for x in summary["results"]][-1], "bl__b__repeat")  # the *baseline* is repeated
        r = self.sweep({"name": "bad", "baseline": "zz", "configs": [{"name": "a"}]})
        self.assertEqual(r.returncode, 2)
        self.assertIn("'baseline'", r.stderr)

    def test_explain_and_help_document_the_spec_and_config_keys(self):
        explain = self.cli("sweep", "--explain", config=False)
        self.assertEqual(explain.returncode, 0)
        for out in (explain.stdout, self.bench("sweep", "-h", config=False).stdout):
            for text in ("name", "base", "configs", "matrix", "baseline", "repeats", "repeat_baseline", "cooldown_seconds",
                         "allow_new_keys", "Example", '"configs": [{"name": "default"}'):
                self.assertIn(text, out)
        self.assertIn("--explain", self.bench("sweep", "-h", config=False).stdout)
        self.assertEqual(self.cli("sweep", config=False).returncode, 2)  # neither --spec nor --explain
        top = self.bench("--help", config=False).stdout
        for key in ("primary_metric", "criteria", "timing_keys", "holdout_split", "provenance", "allow_new_keys", "history_root",
                    "items.json", "diff.json", "metrics.log", "prev/", "130", "BENCH_FIXTURE_<NAME>", "--force", "fixture_drift",
                    "rank_split", "public_metrics", "BENCH_SHOW_HOLDOUT", "BENCH_HOLDOUT_SPLIT", "BENCH_RANK_SPLIT", "hidden_metrics",
                    "rescored_provenance", "all.<metric>"):
            self.assertIn(key, top)
        for sub in ("run", "metrics", "compare", "items", "init", "stats", "determinism"):
            self.assertEqual(self.bench(sub, "-h", config=False).returncode, 0)

    def test_repeats_in_a_sweep(self):
        self.use_param_scorer(timing_keys=["t"], primary_metric={"name": "q", "goal": "max"}, criteria=None)
        spec = {"name": "rp", "repeats": 2, "baseline": "stable",
                "configs": [{"name": "stable", "m": {"q": 0.5}, "noise": True}, {"name": "noisy", "rand": True, "noise": True}]}
        r = self.sweep(spec)
        self.assertEqual(r.returncode, 0, r.stderr)
        summary, md = self.summary()
        self.assertEqual([x["label"] for x in summary["results"]], ["rp__stable-r1", "rp__stable-r2", "rp__noisy-r1", "rp__noisy-r2"])
        self.assertEqual([x["repeat"] for x in summary["results"]], [1, 2, 1, 2])
        groups = {x["config"]: x["group"] for x in summary["results"]}
        self.assertEqual(len(set(groups.values())), 2)
        self.assertEqual({self.run_json(x["run_id"])["group"] for x in summary["results"] if x["config"] == "stable"}, {groups["stable"]})
        self.assertEqual(summary["non_deterministic"], {"noisy": ["q"]})
        self.assertIn("WARNING: quality metrics vary between repeats of `noisy`: q", md)
        self.assertIn("(!)", md)
        self.assertIn("±", md)  # timing key as mean±sd
        cfg_rows = {row["label"]: row for row in summary["configs"]["rows"]}
        self.assertIn("±", str(cfg_rows["rp__stable"]["values"]["t"]))
        self.assertEqual(cfg_rows["rp__stable"]["values"]["q"], 0.5)  # constant quality metric shown plainly
        self.assertIn("(!)", str(cfg_rows["rp__noisy"]["values"]["q"]))
        self.assertEqual(sum(1 for row in summary["configs"]["rows"] if row["rank"]), 2)
        self.assertIn("## Configs", md)
        self.assertIn("## Runs", md)

    def test_every_history_write_is_atomic_and_private(self):
        self.assertEqual(self.sweep({"name": "p", "repeats": 2, "configs": [{"name": "a"}]}).returncode, 0)
        self.cli("metrics", "--all")
        self.no_tmp_files()
        self.assert_private(self.hist)


class RepeatTests(FeatureCase):
    def test_run_repeat_makes_a_group_and_stats_summarize_it(self):
        r = self.cli("run", "--label", "rr", "--repeat", "3")
        self.assertEqual(r.returncode, 0, r.stderr)
        ids = self.run_ids()
        self.assertEqual([self.run_json(i)["label"] for i in ids], ["rr-r1", "rr-r2", "rr-r3"])
        groups = {self.run_json(i)["group"] for i in ids}
        self.assertEqual(len(groups), 1)
        (group,) = groups
        self.assertIn("group: " + group, r.stdout)
        self.assertNotIn("group", self.run_json(self.run_ok("single")))
        for args in (["--group", group], ["--group", group[:12]], ["--label-glob", "rr-r*"], ["rr-r1", "rr-r2", "rr-r3"]):
            out = self.tmp / "stats.json"
            r = self.cli("stats", *args, "--json-out", str(out))
            self.assertEqual(r.returncode, 0, (args, r.stderr))
            st = json.loads(out.read_text())
            self.assertEqual(st["metrics"]["wer"]["n"], 3, args)
            self.assertAlmostEqual(st["metrics"]["wer"]["mean"], 0.2)
            self.assertAlmostEqual(st["metrics"]["wer"]["sd"], 0.0)
            self.out_unlink(out)
        self.assertIn("| wer | 3 |", r.stdout)
        self.assertEqual(self.cli("stats", "--group", "nosuch").returncode, 2)
        self.assertEqual(self.cli("stats").returncode, 2)
        r = self.cli("run", "--label", "dry", "--repeat", "2", "--dry-run")
        self.assertEqual(r.stdout.count("DRY RUN"), 2)
        self.assertEqual(self.cli("run", "--label", "z", "--repeat", "0").returncode, 2)

    def out_unlink(self, path):
        Path(path).unlink()

    def test_stats_math(self):
        rows = [{"run_id": str(i), "label": str(i), "status": "success", "metrics": m}
                for i, m in enumerate([{"a": 1, "b": 10, "c": "x", "t": 5.0}, {"a": 3, "b": 10, "t": 7.0}, {"a": 5, "b": 10, "t": 6.0}])]
        st = statslib.compute_stats(rows, ["t"])["metrics"]
        self.assertEqual(sorted(st), ["a", "b", "t"])  # strings are not numeric metrics
        self.assertEqual((st["a"]["n"], st["a"]["mean"], st["a"]["min"], st["a"]["max"]), (3, 3.0, 1, 5))
        self.assertAlmostEqual(st["a"]["sd"], 2.0)
        self.assertAlmostEqual(st["a"]["cv_pct"], 100 * 2.0 / 3.0)
        self.assertEqual((st["b"]["sd"], st["b"]["cv_pct"]), (0.0, 0.0))
        self.assertTrue(st["t"]["timing"] and not st["a"]["timing"])
        one = statslib.compute_stats(rows[:1], [])["metrics"]["a"]
        self.assertEqual((one["n"], one["sd"], one["cv_pct"]), (1, None, None))
        zero = statslib.compute_stats([{"run_id": "z", "label": "z", "status": "success", "metrics": {"a": 0}},
                                       {"run_id": "y", "label": "y", "status": "success", "metrics": {"a": 0}}], [])
        self.assertIsNone(zero["metrics"]["a"]["cv_pct"])

    def test_determinism_ignores_timing_keys_and_lists_differences(self):
        self.use_param_scorer(timing_keys=["t"])
        a = self.run_ok("a", m={"q": 1, "r": 2}, noise=True)
        b = self.run_ok("b", m={"q": 1, "r": 2}, noise=True)
        self.assertNotEqual(self.metrics_json(a)["t"], self.metrics_json(b)["t"])
        r = self.cli("determinism", a, b)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("DETERMINISTIC", r.stdout)
        self.assertIn("timing keys ignored: t", r.stdout)
        c = self.run_ok("c", m={"q": 1, "r": 3}, noise=True)
        d = self.run_ok("d", m={"q": 1, "r": 2, "extra": 1}, noise=True)
        out = self.tmp / "det.json"
        r = self.cli("determinism", a, c, d, "--json-out", str(out))
        self.assertEqual(r.returncode, 1)
        self.assertIn("NOT DETERMINISTIC", r.stdout)
        res = json.loads(out.read_text())
        self.assertEqual(sorted(res["differing"]), ["extra", "r"])  # q is identical, t is a timing key
        self.assertIsNone(res["differing"]["extra"][a])
        self.assertEqual(self.cli("determinism", a).returncode, 2)  # needs two runs
        self.assertEqual(self.cli("determinism", "--label-glob", "[ab]").returncode, 0)
        self.update_config(timing_keys=[])
        self.assertEqual(self.cli("determinism", a, b).returncode, 1)  # without timing_keys the clock differs


# ---------------------------------------------------------------- items, splits, holdout

class ItemsTests(FeatureCase):
    SPLITS = {"i1": "train", "i2": "train", "i3": "holdout", "i5": "train", "i7": "train"}
    A = {"i1": 0.9, "i2": 0.4, "i3": 1.0, "i4": 0.2, "i7": 0.6}
    B = {"i1": 0.7, "i2": 0.8, "i3": 1.0, "i5": 0.5, "i7": 0.45}

    def setUp(self):
        super().setUp()
        self.use_param_scorer()

    def run_items(self, label, scores, **extra):
        return self.run_ok(label, m={"f1": 0.5}, items=items_doc(scores, self.SPLITS), **extra)

    def test_split_aggregates_are_added_without_clobbering(self):
        a = self.run_items("a", self.A)
        m = self.metrics_json(a)
        self.assertEqual(m["f1"], 0.5)
        self.assertAlmostEqual(m["train.score_mean"], (0.9 + 0.4 + 0.6) / 3)
        self.assertAlmostEqual(m["train.pass_rate"], 2 / 3)
        self.assertEqual((m["train.n"], m["holdout.n"], m["holdout.pass_rate"], m["holdout.score_mean"]), (3, 1, 1.0, 1.0))
        self.assertEqual(sorted(k for k in m if k.endswith(".n")), ["holdout.n", "train.n"])  # null split: no aggregate
        self.assertEqual(self.index_rows()["a"]["metrics"]["train.n"], 3)
        c = self.run_ok("c", m={"train.n": 99}, items=items_doc(self.A, self.SPLITS))
        self.assertEqual(self.metrics_json(c)["train.n"], 99)  # the script's own key wins
        self.assertIn("train.score_mean", self.metrics_json(c))
        self.assertEqual(itemslib.split_aggregates({"x": {"split": None, "score": 1}}), {})

    def test_invalid_items_json_fails_the_metrics_stage(self):
        bad = {"schema_version": 1, "items": {"i1": {"split": "train", "score": "high", "pass": "yes"}}}
        r = self.cli("run", "--label", "bad", *self.set_args(m={"f1": 1}, items=bad))
        self.assertEqual(r.returncode, 1)
        run = self.run_json(self.only_run())
        self.assertEqual((run["status"], run["error_code"]), ("metrics_failed", "items_invalid"))
        for doc in (None, [], {"schema_version": 2, "items": {}}, {"schema_version": 1, "items": []},
                    {"schema_version": 1, "items": {"a": {"split": 3}}}, {"schema_version": 1, "items": {"a": {"score": float("nan")}}}):
            self.assertTrue(itemslib.validate_items(doc), doc)
        self.assertEqual(itemslib.validate_items({"schema_version": 1, "items": {"a": {"split": None, "score": 1, "pass": None}}}), [])

    def test_item_comparison_categories_sorting_and_privacy(self):
        a, b = self.run_items("a", self.A), self.run_items("b", self.B)
        out = self.tmp / "items.json"
        md_out = self.tmp / "items.md"
        r = self.cli("items", a, b, "--json-out", str(out), "--markdown-out", str(md_out), "--show-holdout")
        self.assertEqual(r.returncode, 0, r.stderr)
        res = json.loads(out.read_text())
        self.assertEqual([e["item"] for e in res["regressed"]], ["i1", "i7"])  # worst delta first; i7 also flipped to fail
        self.assertEqual([e["item"] for e in res["improved"]], ["i2"])
        self.assertEqual([e["item"] for e in res["unchanged"]], ["i3"])
        self.assertEqual([e["item"] for e in res["only_in_a"]], ["i4"])
        self.assertEqual([e["item"] for e in res["only_in_b"]], ["i5"])
        self.assertAlmostEqual(res["regressed"][0]["delta"], -0.2)
        self.assertEqual((res["regressed"][1]["pass_a"], res["regressed"][1]["pass_b"]), (True, False))
        self.assertEqual(res["counts"], {"regressed": 2, "improved": 1, "unchanged": 1, "only_in_a": 1, "only_in_b": 1})
        self.assertEqual(md_out.read_text().strip(), r.stdout.strip())
        for text in (r.stdout, out.read_text()):
            self.assertNotIn("CONTENT-", text)  # per-item detail never leaves the run dir
        self.assertIn("## Regressed", r.stdout)
        # threshold: i1 (-0.2) is inside 0.25, but flips decide for i7 and i2
        res = self.items_json(a, b, "--threshold", "0.25", "--show-holdout")
        self.assertEqual(([e["item"] for e in res["regressed"]], [e["item"] for e in res["improved"]]), (["i7"], ["i2"]))
        self.assertEqual([e["item"] for e in res["unchanged"]], ["i1", "i3"])
        res = self.items_json(a, b, "--split", "train")
        self.assertEqual(sorted(e["item"] for k in ("regressed", "improved", "unchanged", "only_in_a", "only_in_b") for e in res[k]),
                         ["i1", "i2", "i5", "i7"])
        self.assertEqual(self.cli("items", a, b, "--threshold", "-1").returncode, 2)
        none = self.run_ok("none")
        self.assertEqual(self.cli("items", a, none).returncode, 2)
        self.assertIn("no items.json", self.cli("items", a, none).stderr)

    def items_json(self, a, b, *args):
        out = self.tmp / ("items-%d.json" % len(list(self.tmp.glob("items-*"))))
        self.assertEqual(self.cli("items", a, b, "--json-out", str(out), *args).returncode, 0)
        return json.loads(out.read_text())

    def test_holdout_split_is_hidden_from_item_comparison_by_default(self):
        self.update_config(holdout_split="holdout")
        a, b = self.run_items("a", self.A), self.run_items("b", self.B)
        res = self.items_json(a, b)
        self.assertEqual(res["hidden_holdout_items"], 1)
        self.assertNotIn("i3", json.dumps(res["unchanged"]))
        self.assertEqual(self.items_json(a, b, "--show-holdout")["hidden_holdout_items"], 0)
        explicit = self.items_json(a, b, "--split", "holdout")
        self.assertEqual([e["item"] for e in explicit["unchanged"]], ["i3"])


class HoldoutGuardTests(FeatureCase):
    def setUp(self):
        super().setUp()
        self.use_param_scorer(holdout_split="holdout", primary_metric={"name": "f1", "goal": "max"}, criteria=None,
                              index_columns=None)
        # overall f1 favors B (inflated by its holdout items); the tuning split favors A
        self.a = self.run_ok("a", m={"f1": 0.8, "train.f1": 0.9, "holdout.f1": 0.5})
        self.b = self.run_ok("b", m={"f1": 0.9, "train.f1": 0.7, "holdout.f1": 0.95})

    @staticmethod
    def table_header(text):
        return next(l for l in text.splitlines() if l.startswith("| "))

    def test_compare_hides_holdout_columns_and_ranks_by_train(self):
        out = self.tmp / "c.json"
        r = self.cli("compare", self.b, self.a, "--json-out", str(out))
        comp = json.loads(out.read_text())
        self.assertNotIn("holdout.f1", comp["columns"])
        self.assertIn("train.f1", comp["columns"])
        self.assertEqual(comp["sort"]["key"], "train.f1")
        self.assertEqual([x["run_id"] for x in comp["rows"]], [self.a, self.b])
        self.assertNotIn("holdout.f1", self.table_header(r.stdout))
        r = self.cli("compare", self.b, self.a, "--show-holdout", "--json-out", str(self.tmp / "c2.json"))
        self.assertIn("holdout.f1", self.table_header(r.stdout))
        r = self.cli("compare", self.a, self.b, "--columns", "holdout.f1")
        self.assertEqual(r.returncode, 2)
        self.assertIn("--show-holdout", r.stderr)
        self.assertEqual(self.cli("compare", self.a, self.b, "--sort", "holdout.f1").returncode, 2)
        self.assertEqual(self.cli("compare", self.a, self.b, "--sort", "holdout.f1", "--show-holdout").returncode, 0)

    def test_sweep_ranking_uses_train_and_prints_the_reminder(self):
        spec = {"name": "hg", "configs": [{"name": "x", "m": {"f1": 0.8, "train.f1": 0.9, "holdout.f1": 0.5}},
                                          {"name": "y", "m": {"f1": 0.9, "train.f1": 0.7, "holdout.f1": 0.95}}]}
        self.assertEqual(self.cli("sweep", "--spec", self.write_spec(spec)).returncode, 0)
        summary = json.loads(next((self.hist / "sweeps").glob("*/summary.json")).read_text())
        self.assertEqual(summary["ranking"]["key"], "train.f1")
        self.assertEqual([r["label"] for r in summary["comparison"]["rows"]], ["hg__x", "hg__y"])
        self.assertNotIn("holdout.f1", summary["comparison"]["columns"])
        md = next((self.hist / "sweeps").glob("*/summary.md")).read_text()
        self.assertIn("Look at it only for the final chosen config", md)
        self.assertIn("--show-holdout", md)
        self.assertNotIn("holdout.f1", self.table_header(md))

    def test_warning_when_no_train_metric_exists_and_index_hides_holdout(self):
        c = self.run_ok("c", m={"f1": 0.6, "holdout.f1": 0.4})
        r = self.cli("compare", c)
        self.assertIn("WARNING: holdout_split is set but no `train.f1` metric", r.stdout)
        index_md = (self.hist / "index.md").read_text()
        self.assertNotIn("holdout.f1", self.table_header(index_md))
        self.assertIn("Holdout split `holdout`", index_md)
        self.assertEqual(self.index_rows()["a"]["metrics"]["holdout.f1"], 0.5)  # index.json keeps everything
        stats = self.cli("stats", self.a, self.b).stdout
        self.assertNotIn("| holdout.f1", stats)
        self.assertIn("holdout.f1", self.cli("stats", self.a, self.b, "--show-holdout").stdout)


# ---------------------------------------------------------------- init templates

class InitTemplateTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="bench-init-")
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def init(self, rel, *args, cwd=None):
        return subprocess.run([sys.executable, str(SCRIPTS / "bench.py"), "init", rel, *args], cwd=str(cwd or self.tmp),
                              capture_output=True, text=True, timeout=60)

    def project_bench(self, proj, *args, cwd=None, timeout=120):
        """Run the init'ed project's OWN copy of bench.py."""
        return subprocess.run([sys.executable, str(proj / "bench.py"), *args], cwd=str(cwd or proj), capture_output=True,
                              text=True, timeout=timeout)

    def test_init_copies_toolkit_without_absolute_paths_and_refuses_overwrite(self):
        for template in ("text", "items"):
            with self.subTest(template):
                proj = self.tmp / ("p-" + template)
                r = self.init(proj.name, "--template", template)
                self.assertEqual(r.returncode, 0, r.stderr)
                for name in ["bench.py", "textdiff.py", "calibrate.py", "bench.json", "bench_stub_driver.py", "bench_stub_metrics.py"] + [
                        p.name for p in SCRIPTS.glob("benchlib_*.py")]:
                    self.assertTrue((proj / name).exists(), name)
                    self.assertTrue((proj / name).read_text() == (SCRIPTS / name).read_text() if (SCRIPTS / name).exists() else True)
                for f in proj.rglob("*"):
                    if f.is_file():
                        self.assertNotIn(str(SCRIPTS), f.read_text(), f)  # nothing points back to the skill
                before = (proj / "bench.json").read_text()
                r = self.init(proj.name, "--template", template)
                self.assertEqual(r.returncode, 2)
                self.assertIn("refusing to overwrite", r.stderr)
                self.assertEqual((proj / "bench.json").read_text(), before)
        self.assertEqual(self.init("bad", "--template", "nope").returncode, 2)
        keys = json.loads((self.tmp / "p-items" / "bench.json").read_text())
        for key in ("primary_metric", "criteria", "holdout_split", "timing_keys", "provenance"):
            self.assertIn(key, keys)

    def test_printed_hints_run_as_printed_from_the_users_cwd(self):
        r = self.init("nested/dir", "--template", "items")
        self.assertEqual(r.returncode, 0, r.stderr)
        hints = [l.strip() for l in r.stdout.splitlines() if l.strip().startswith("python3 ")]
        self.assertGreaterEqual(len(hints), 4)
        self.assertTrue(all("nested/dir/bench.py" in h for h in hints))
        for hint in hints[:2]:  # `run --label baseline` and `run --label no-date --set ...`
            argv = shlex.split(hint)
            argv[0] = sys.executable
            done = subprocess.run(argv, cwd=str(self.tmp), capture_output=True, text=True, timeout=60)
            self.assertEqual(done.returncode, 0, (hint, done.stderr))
        for hint in hints[2:4]:  # compare / items of the two runs just made
            argv = shlex.split(hint)
            argv[0] = sys.executable
            done = subprocess.run(argv, cwd=str(self.tmp), capture_output=True, text=True, timeout=60)
            self.assertEqual(done.returncode, 0, (hint, done.stderr))
        hint = hints[-1]
        self.assertIn("sweep --explain", hint)
        r = self.init("here", "--template", "text")
        self.assertIn("here/bench.py run --label baseline --config here/bench.json", r.stdout)

    def test_items_template_end_to_end(self):
        self.assertEqual(self.init("proj", "--template", "items").returncode, 0)
        proj = self.tmp / "proj"
        pb = lambda *a, **k: self.project_bench(proj, *a, **k)
        r = pb("run", "--label", "base")
        self.assertEqual(r.returncode, 0, r.stderr)
        base = next(i for i in os.listdir(proj / "history" / "runs") if i.endswith("_base"))
        m = json.loads((proj / "history" / "runs" / base / "metrics.json").read_text())
        for key in ("latency_ms", "train.precision", "train.recall", "train.f1", "train.exact_rate", "holdout.f1",
                    "train.score_mean", "train.pass_rate", "train.n", "holdout.n"):
            self.assertIn(key, m)
        self.assertEqual([k for k in m if "." not in k], ["latency_ms"])  # no unsplit quality metric next to the splits
        self.assertEqual((m["train.n"], m["holdout.n"]), (5, 3))
        diff = (proj / "history" / "runs" / base / "diff.md").read_text()
        self.assertIn("FAIL", diff)
        self.assertIn("customer", diff)
        self.assertNotIn("item-06", diff)  # the holdout items are left out of the per-run reports
        items = json.loads((proj / "history" / "runs" / base / "items.json").read_text())["items"]
        self.assertEqual({i["split"] for i in items.values()}, {"train", "holdout"})
        self.assertEqual(pb("run", "--label", "nodate", "--set", "extract_date=false").returncode, 0)
        self.assertEqual(pb("run", "--label", "typo", "--set", "extract_dat=false").returncode, 2)  # unknown key
        spec = self.tmp / "spec.json"
        spec.write_text(json.dumps({"name": "ex", "baseline": "default", "repeats": 2,
                                    "configs": [{"name": "default"}, {"name": "nodate", "extract_date": False}],
                                    "matrix": {"lowercase_customer": [True]}}))
        r = pb("sweep", "--spec", str(spec))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("best by train.f1: default", r.stdout)
        (sweep_dir,) = list((proj / "history" / "sweeps").iterdir())
        md = (sweep_dir / "summary.md").read_text()
        self.assertNotIn("holdout.f1", md)
        self.assertIn("(best)", md)
        self.assertIn("±", md)  # latency_ms is a timing key
        self.assertNotIn("WARNING: quality metrics vary", md)  # the stub is deterministic
        r = pb("compare", "base", "nodate")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("holdout.f1", r.stdout)
        self.assertIn("holdout.f1", pb("compare", "base", "nodate", "--show-holdout").stdout)
        r = pb("items", "base", "nodate")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("## Regressed", r.stdout)
        self.assertEqual(pb("stats", "--label-glob", "ex__default-r*").returncode, 0)
        r = pb("determinism", "--label-glob", "ex__default-r*")
        self.assertEqual(r.returncode, 0, r.stdout)  # latency_ms differs but is a timing key
        # fixture drift: add an item, then the rescore refuses until --force
        (proj / "fixtures" / "items" / "item-09.json").write_text(json.dumps({"id": "item-09", "input": "Order 1 from Zed Zed, total 1.00 EUR"}))
        r = pb("metrics", "--run", "base")
        self.assertEqual(r.returncode, 2)
        self.assertIn("--force", r.stderr)
        r = pb("metrics", "--run", "base", "--force")
        self.assertEqual(r.returncode, 0, r.stderr)
        run = json.loads((proj / "history" / "runs" / base / "run.json").read_text())
        self.assertTrue(run["fixture_drift"])
        self.assertTrue(run["provenance"]["paths"][0]["sha256"])
        self.assertEqual(pb("index").returncode, 0)
        out = subprocess.run([sys.executable, str(proj / "bench.py"), "compare", "base", "nodate"], cwd=str(proj),
                             capture_output=True, text=True)
        self.assertIn("WARNING: compared runs used different fixtures", out.stdout)

    def test_text_template_end_to_end(self):
        self.assertEqual(self.init("proj").returncode, 0)  # default template = text
        proj = self.tmp / "proj"
        pb = lambda *a: self.project_bench(proj, *a)
        self.assertEqual(pb("run", "--label", "base", "--repeat", "2").returncode, 0)
        self.assertEqual(pb("run", "--label", "dropped", "--set", "drop_every=3").returncode, 0)
        spec = self.tmp / "spec.json"
        spec.write_text(json.dumps({"name": "tx", "configs": [{"name": "a"}], "matrix": {"drop_every": [2, 3]}}))
        r = pb("sweep", "--spec", str(spec))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("best by wer: a", r.stdout)
        r = pb("compare", "base-r1", "dropped")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("| pass |", r.stdout)
        self.assertEqual(pb("items", "base-r1", "dropped").returncode, 2)  # the text template writes no items.json
        self.assertIn("no items.json", pb("items", "base-r1", "dropped").stderr)
        self.assertEqual(pb("stats", "--label-glob", "base-r*").returncode, 0)
        self.assertEqual(pb("determinism", "--label-glob", "base-r*").returncode, 0)
        (proj / "fixtures" / "reference.txt").write_text("another reference\n")
        self.assertEqual(pb("metrics", "--run", "dropped").returncode, 2)
        self.assertEqual(pb("metrics", "--run", "dropped", "--force").returncode, 0)
        self.assertEqual(pb("metrics", "--all", "--force").returncode, 0)
        self.assertIn("fixture_drift", json.dumps(json.loads(next((proj / "history" / "runs").glob("*_dropped/run.json")).read_text())))


# ---------------------------------------------------------------- misc

class OutputRobustnessTests(FeatureCase):
    def test_closed_stdout_does_not_traceback(self):
        self.run_ok("a")
        cmd = [sys.executable, str(SCRIPTS / "bench.py"), "compare", "--config", str(self.conf), "a"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        proc.stdout.close()  # like `| head -0`
        err = proc.stderr.read()
        proc.stderr.close()
        self.assertEqual(proc.wait(timeout=30), 0)
        self.assertNotIn("Traceback", err)
        self.assertNotIn("BrokenPipe", err)
        proc = subprocess.Popen([sys.executable, str(SCRIPTS / "bench.py"), "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        proc.stdout.close()
        err = proc.stderr.read()
        proc.stderr.close()
        self.assertEqual(proc.wait(timeout=30), 0)
        self.assertNotIn("Traceback", err)

    def test_say_survives_a_broken_pipe(self):
        class Broken:
            def write(self, _):
                raise BrokenPipeError()
            flush = write
        with mock.patch.object(util, "close_stdout") as closed, mock.patch.object(sys, "stdout", Broken()):
            util.say("hello")
        closed.assert_called_once()
        with mock.patch.object(util, "_stdout_closed", True), contextlib.redirect_stdout(io.StringIO()) as out:
            util.say("dropped")
        self.assertEqual(out.getvalue(), "")

    def test_write_atomic_leaves_no_temp_file_and_is_private(self):
        target = self.tmp / "f.json"
        util.write_json(target, {"a": 1})
        util.write_json(target, {"a": 2})
        self.assertEqual(json.loads(target.read_text()), {"a": 2})
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
        self.assertEqual(sorted(p.name for p in self.tmp.glob(".f.json*")), [])
        self.assertEqual(util.tail_lines(self.tmp / "missing", 5), [])


if __name__ == "__main__":
    unittest.main()
