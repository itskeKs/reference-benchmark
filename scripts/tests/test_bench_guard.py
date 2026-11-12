"""Tests for the holdout guard (unsplit metrics, rank_split, public_metrics, BENCH_SHOW_HOLDOUT), the classification
helpers, code-identity warnings, group resolution, override type checks, sweep ties / inert parameters and init's
.gitignore. Same conventions as test_bench_features.py: in-process bench.main(), temp dirs only."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchlib_classify as clf  # noqa: E402
import benchlib_config as cfglib  # noqa: E402
import benchlib_runs as runslib  # noqa: E402
import benchlib_util as util  # noqa: E402
from test_bench_features import PARAM_SCORER, FeatureCase, items_doc  # noqa: E402

ENV_SCORER = PARAM_SCORER + '''
env = {k: os.environ.get(k, "unset") for k in ("BENCH_SHOW_HOLDOUT", "BENCH_HOLDOUT_SPLIT", "BENCH_RANK_SPLIT")}
json.dump({**m, **{"env_" + k.lower(): v for k, v in env.items()}}, open(d + "/metrics.json", "w"))
'''
WARNING = "bench: warning: holdout_split 'holdout' is set"


def hashlib_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table_header(text):
    return next(l for l in text.splitlines() if l.startswith("| "))


# ---------------------------------------------------------------- classification helpers

class ClassifyTests(unittest.TestCase):
    PAIRS = [("a", "a"), ("a", "a"), ("a", "b"), ("b", "b"), ("b", "c"), ("c", "c")]

    def test_accuracy_per_class_and_macro_f1_by_hand(self):
        s = clf.summarize(self.PAIRS)
        self.assertAlmostEqual(s["accuracy"], 4 / 6)
        a, b, c = (s["per_class"][k] for k in "abc")
        self.assertEqual((a["tp"], a["fp"], a["fn"], a["support"]), (2, 0, 1, 3))
        self.assertAlmostEqual(a["precision"], 1.0)
        self.assertAlmostEqual(a["recall"], 2 / 3)
        self.assertAlmostEqual(a["f1"], 0.8)
        self.assertAlmostEqual(b["f1"], 0.5)
        self.assertAlmostEqual(c["precision"], 0.5)
        self.assertAlmostEqual(c["f1"], 2 / 3)
        self.assertAlmostEqual(s["macro_f1"], (0.8 + 0.5 + 2 / 3) / 3)
        self.assertAlmostEqual(s["macro_precision"], (1 + 0.5 + 0.5) / 3)
        self.assertAlmostEqual(s["macro_recall"], (2 / 3 + 0.5 + 1) / 3)
        self.assertEqual(s["labels"], ["a", "b", "c"])

    def test_missing_predictions_count_as_wrong_but_are_no_class(self):
        s = clf.summarize([("a", "a"), ("a", None), ("b", "b")])
        self.assertAlmostEqual(s["accuracy"], 2 / 3)
        self.assertEqual(s["labels"], ["a", "b"])  # "(missing)" is not averaged
        self.assertAlmostEqual(s["per_class"]["a"]["recall"], 0.5)
        self.assertAlmostEqual(s["per_class"]["a"]["precision"], 1.0)
        md = clf.confusion_markdown([("a", "a"), ("a", None), ("b", "b")])
        self.assertIn("(missing)", table_header(md))

    def test_class_without_predictions_scores_zero_and_empty_input_is_none(self):
        s = clf.summarize([("a", "a"), ("b", "a")])  # b is never predicted: P = R = F1 = 0
        self.assertEqual(s["per_class"]["b"]["f1"], 0.0)
        self.assertAlmostEqual(s["macro_f1"], (2 / 3 + 0.0) / 2)
        empty = clf.summarize([])
        self.assertEqual((empty["n"], empty["accuracy"], empty["macro_f1"]), (0, None, None))

    def test_labels_are_compared_as_strings_and_non_strings_are_encoded(self):
        self.assertEqual(clf.label_of(None), None)
        self.assertEqual(clf.label_of(3), "3")
        self.assertEqual(clf.accuracy([("1", 1), ("x", "y")]), 0.5)  # the int 1 is the label "1"

    def test_confusion_matrix_markdown_and_folding(self):
        md = clf.confusion_markdown(self.PAIRS)
        lines = md.splitlines()
        self.assertEqual(lines[0], "| gold \\ predicted | a | b | c |")
        self.assertEqual(lines[2], "| a | 2 | 1 |  |")
        self.assertEqual(lines[4], "| c |  |  | 1 |")
        many = [("c%02d" % i, "c%02d" % i) for i in range(30)] + [("c00", "c00")] * 5
        folded = clf.confusion(many, max_labels=10)
        self.assertTrue(folded["folded"])
        self.assertEqual(len(folded["rows"]), 11)
        self.assertEqual(folded["matrix"]["c00"]["c00"], 6)
        self.assertIn("(other)", clf.confusion_markdown(many, max_labels=10))
        table = clf.per_class_markdown(clf.summarize(self.PAIRS))
        self.assertIn("| a | 3 | 1.000 | 0.667 | 0.800 |", table)


# ---------------------------------------------------------------- holdout guard: unsplit metrics

class UnsplitGuardTests(FeatureCase):
    METRICS = {"accuracy": 0.91, "macro_f1": 0.88, "train.f1": 0.7, "holdout.f1": 0.99, "latency_ms": 5, "all.f1": 0.9}

    def setUp(self):
        super().setUp()
        self.use_param_scorer(holdout_split="holdout", primary_metric={"name": "f1", "goal": "max"}, criteria=None,
                              index_columns=None, timing_keys=["latency_ms"])

    def test_unsplit_keys_are_hidden_everywhere_and_a_one_time_warning_names_them(self):
        r = self.cli("run", "--label", "a", "--repeat", "2", *self.set_args(m=self.METRICS))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stderr.count(WARNING), 1)  # the index is rebuilt after every run: still one warning
        for key in ("accuracy", "all.f1", "macro_f1"):
            self.assertIn(key, r.stderr)
        index_md = (self.hist / "index.md").read_text()
        header = table_header(index_md)
        for key in ("accuracy", "macro_f1", "all.f1", "holdout.f1"):
            self.assertNotIn(key, header)
        self.assertIn("train.f1", header)
        self.assertIn("latency_ms", header)  # timing keys are never hidden
        self.assertIn("`accuracy`", index_md)  # the note names what is hidden
        row = self.index_rows()["a-r1"]
        self.assertEqual(row["metrics"]["accuracy"], 0.91)  # index.json stays complete ...
        self.assertEqual(row["hidden_metrics"], ["accuracy", "all.f1", "holdout.f1", "macro_f1"])  # ... and marks what is hidden
        a = self.run_for("a-r1")
        cmp_ = self.cli("compare", a, self.run_for("a-r2"))
        for key in ("accuracy", "macro_f1", "all.f1", "holdout.f1"):
            self.assertNotIn(key, table_header(cmp_.stdout))
        self.assertIn("NOTE: unsplit metric(s)", cmp_.stdout)
        self.assertEqual(cmp_.stderr.count(WARNING), 1)
        shown = self.cli("compare", a, self.run_for("a-r2"), "--show-holdout")
        for key in ("accuracy", "macro_f1", "all.f1", "holdout.f1"):
            self.assertIn(key, table_header(shown.stdout))
        self.assertNotIn(WARNING, shown.stderr)
        stats = self.cli("stats", "--group", "a")
        self.assertNotIn("| accuracy", stats.stdout)
        self.assertIn("| train.f1", stats.stdout)
        self.assertIn("NOTE: unsplit metric(s)", stats.stdout)
        self.assertIn("| accuracy", self.cli("stats", "--group", "a", "--show-holdout").stdout)
        for args in (["--columns", "accuracy"], ["--sort", "macro_f1"]):
            denied = self.cli("compare", a, *args)
            self.assertEqual(denied.returncode, 2, args)
            self.assertIn("--show-holdout", denied.stderr)
        self.assertEqual(self.cli("compare", a, "--sort", "macro_f1", "--show-holdout").returncode, 0)

    def test_public_metrics_allowlist_accepts_names_and_globs(self):
        self.update_config(public_metrics=["accuracy", "pub.*"])
        self.run_ok("a", m={**self.METRICS, "pub.x": 1, "pub.y": 2})
        header = table_header((self.hist / "index.md").read_text())
        for key in ("accuracy", "pub.x", "pub.y"):
            self.assertIn(key, header)
        for key in ("macro_f1", "all.f1", "holdout.f1"):
            self.assertNotIn(key, header)
        self.assertEqual(self.index_rows()["a"]["hidden_metrics"], ["all.f1", "holdout.f1", "macro_f1"])
        self.update_config(public_metrics=["holdout.*"])  # the holdout namespace itself can never be allowlisted
        self.cli("index")
        self.assertNotIn("holdout.f1", table_header((self.hist / "index.md").read_text()))
        for bad in ("accuracy", [1], [""]):
            self.update_config(public_metrics=bad)
            self.assertEqual(self.cli("index").returncode, 2, bad)

    def test_sweep_summary_hides_unsplit_metrics_and_notes_them(self):
        spec = {"name": "g", "configs": [{"name": "x", "m": self.METRICS}, {"name": "y", "m": {**self.METRICS, "train.f1": 0.5}}]}
        r = self.cli("sweep", "--spec", self.write_spec(spec))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stderr.count(WARNING), 1)
        summary = json.loads(next((self.hist / "sweeps").glob("*/summary.json")).read_text())
        self.assertEqual(summary["ranking"]["key"], "train.f1")
        self.assertEqual(summary["hidden_metrics"], ["accuracy", "all.f1", "macro_f1"])
        md = next((self.hist / "sweeps").glob("*/summary.md")).read_text()
        for key in ("accuracy", "macro_f1", "all.f1", "holdout.f1"):
            self.assertNotIn(key, table_header(md))
        self.assertIn("NOTE: unsplit metric(s)", md)

    def test_determinism_verdict_counts_hidden_keys_but_masks_their_values(self):
        a = self.run_ok("a", m={"train.f1": 0.5, "accuracy": 0.91})
        b = self.run_ok("b", m={"train.f1": 0.5, "accuracy": 0.42})
        r = self.cli("determinism", a, b)
        self.assertEqual(r.returncode, 1)  # the hidden key still decides
        self.assertIn("accuracy", r.stdout)
        self.assertNotIn("0.91", r.stdout)
        self.assertIn("(hidden)", r.stdout)
        self.assertIn("0.91", self.cli("determinism", a, b, "--show-holdout").stdout)

    def test_without_holdout_split_nothing_is_hidden(self):
        self.update_config(holdout_split=None)
        self.run_ok("a", m=self.METRICS)
        header = table_header((self.hist / "index.md").read_text())
        for key in ("accuracy", "macro_f1", "all.f1", "holdout.f1"):
            self.assertIn(key, header)
        self.assertNotIn("hidden_metrics", self.index_rows()["a"])

    def test_known_splits_come_from_items_json_aggregates(self):
        splits = {"i1": "train", "i2": "dev", "i3": "holdout", "i4": "holdout"}
        a = self.run_ok("a", m={"dev.f1": 0.6, "train.f1": 0.9, "holdout.f1": 0.4, "accuracy": 0.7, "test.f1": 0.3},
                        items=items_doc({"i1": 0.9, "i2": 0.8, "i3": 0.7, "i4": 0.6}, splits))
        rows = [runslib.load_run(self.hist / "runs" / a)]
        guard = runslib.Guard(cfglib.load_config(self.conf), rows)
        self.assertEqual(guard.splits, {"train", "dev"})
        self.assertEqual([guard.reason(k) for k in ("dev.f1", "train.f1", "holdout.f1", "accuracy", "test.f1", "all.f1")],
                         [None, None, "holdout", "unsplit", "unsplit", "unsplit"])  # `test` is no known split: allowlist it


# ---------------------------------------------------------------- rank_split and criteria

class RankSplitTests(FeatureCase):
    def setUp(self):
        super().setUp()
        self.use_param_scorer(holdout_split="holdout", rank_split="dev", primary_metric={"name": "f1", "goal": "max"},
                              criteria=None, index_columns=None)
        # train favors A (it was fitted on train!), dev favors B
        self.a = self.run_ok("a", m={"train.f1": 0.99, "dev.f1": 0.6, "holdout.f1": 0.5})
        self.b = self.run_ok("b", m={"train.f1": 0.7, "dev.f1": 0.8, "holdout.f1": 0.9})

    def test_compare_and_sweep_rank_on_the_rank_split(self):
        out = self.tmp / "c.json"
        r = self.cli("compare", self.a, self.b, "--json-out", str(out))
        comp = json.loads(out.read_text())
        self.assertEqual(comp["sort"]["key"], "dev.f1")
        self.assertEqual([x["run_id"] for x in comp["rows"]], [self.b, self.a])
        header = table_header(r.stdout)
        self.assertIn("train.f1", header)  # other non-holdout splits stay visible
        self.assertNotIn("holdout.f1", header)
        spec = {"name": "rs", "configs": [{"name": "x", "m": {"train.f1": 0.99, "dev.f1": 0.6}},
                                          {"name": "y", "m": {"train.f1": 0.7, "dev.f1": 0.8}}]}
        r = self.cli("sweep", "--spec", self.write_spec(spec))
        self.assertIn("best by dev.f1: y", r.stdout)

    def test_criteria_use_the_rank_split_with_a_fallback_warning(self):
        self.update_config(criteria={"f1": ">=0.75", "recall": ">=0.5", "latency_ms": "<=10"}, timing_keys=["latency_ms"])
        ok = self.run_ok("ok", m={"f1": 0.99, "dev.f1": 0.8, "recall": 0.9, "dev.recall": 0.7, "latency_ms": 3})
        cr = self.run_json(ok)["criteria_result"]
        self.assertTrue(cr["pass"])
        self.assertEqual(cr["details"]["f1"], {"criterion": ">=0.75", "value": 0.8, "pass": True, "metric": "dev.f1"})
        self.assertEqual(cr["details"]["recall"]["metric"], "dev.recall")
        self.assertNotIn("metric", cr["details"]["latency_ms"])  # timing keys are used as they are, silently
        self.assertNotIn("warnings", cr)
        bad = self.run_ok("bad", m={"f1": 0.99, "dev.f1": 0.3, "recall": 0.9, "dev.recall": 0.7, "latency_ms": 3})
        self.assertFalse(self.run_json(bad)["criteria_result"]["pass"])  # the inflated unsplit f1 would have passed
        r = self.cli("run", "--label", "fallback", *self.set_args(m={"f1": 0.9, "recall": 0.9, "dev.recall": 0.9, "latency_ms": 3}))
        cr = self.run_json(self.run_for("fallback"))["criteria_result"]
        self.assertTrue(cr["pass"])  # falls back to the plain key ...
        self.assertNotIn("metric", cr["details"]["f1"])
        self.assertIn("no `dev.f1` metric", cr["details"]["f1"]["warning"])  # ... and records why
        self.assertIn("may include holdout items", cr["warnings"][0])
        self.assertIn("no `dev.f1` metric", r.stderr)
        self.update_config(criteria={"train.f1": ">=0.95", "holdout.f1": ">=0.4"})
        done = self.run_ok("explicit", m={"train.f1": 0.97, "dev.f1": 0.1, "holdout.f1": 0.5})
        details = self.run_json(done)["criteria_result"]["details"]
        self.assertEqual((details["train.f1"]["value"], details["holdout.f1"]["value"]), (0.97, 0.5))  # prefixed names as is

    def test_rank_split_without_holdout_split_and_validation(self):
        self.update_config(holdout_split=None, criteria={"f1": ">=0.7"})
        c = self.run_ok("c", m={"f1": 0.1, "dev.f1": 0.9, "train.f1": 0.2})
        self.assertEqual(self.run_json(c)["criteria_result"]["details"]["f1"]["metric"], "dev.f1")
        self.assertEqual(self.cli("compare", self.a, self.b).stdout.count("Ranked by `dev.f1`"), 1)
        header = table_header((self.hist / "index.md").read_text())
        self.assertIn("holdout.f1", header)  # nothing is hidden without a holdout split
        for bad, text in (({"rank_split": "holdout", "holdout_split": "holdout"}, "must differ"), ({"rank_split": ""}, "split name"),
                          ({"rank_split": 3}, "split name"),
                          ({"holdout_split": "holdout", "primary_metric": {"name": "holdout.f1", "goal": "max"}}, "holdout split")):
            self.update_config(**{"rank_split": None, "holdout_split": None, "primary_metric": {"name": "f1", "goal": "max"}, **bad})
            r = self.cli("index")
            self.assertEqual(r.returncode, 2, bad)
            self.assertIn(text, r.stderr)

    def test_missing_rank_split_metric_warns_in_compare(self):
        c = self.run_ok("c", m={"f1": 0.6})
        r = self.cli("compare", c)
        self.assertIn("WARNING: holdout_split is set but no `dev.f1` metric", r.stdout)


# ---------------------------------------------------------------- BENCH_SHOW_HOLDOUT

class ShowHoldoutEnvTests(FeatureCase):
    def setUp(self):
        super().setUp()
        self.update_config(metrics=[sys.executable, self.script("env_scorer.py", ENV_SCORER)], index_columns=None,
                           holdout_split="holdout", rank_split="dev")

    def test_env_is_set_for_the_metrics_command_only_on_metrics_show_holdout(self):
        with mock.patch.dict(os.environ, {"BENCH_SHOW_HOLDOUT": "1", "BENCH_HOLDOUT_SPLIT": "stale", "BENCH_RANK_SPLIT": "stale"}):
            a = self.run_ok("a", m={"x": 1})
            m = self.metrics_json(a)
            self.assertEqual((m["env_bench_show_holdout"], m["env_bench_holdout_split"], m["env_bench_rank_split"]),
                             ("unset", "holdout", "dev"))  # inherited values never reach the command
            self.assertEqual(self.cli("metrics", "--run", a, "--show-holdout").returncode, 0)
            self.assertEqual(self.metrics_json(a)["env_bench_show_holdout"], "1")
            self.assertTrue(self.run_json(a)["show_holdout"])
            self.assertEqual(self.cli("metrics", "--run", a).returncode, 0)
            self.assertEqual(self.metrics_json(a)["env_bench_show_holdout"], "unset")
            self.assertNotIn("show_holdout", self.run_json(a))
        self.assertEqual(self.cli("metrics", "--run", a, "--env", "BENCH_SHOW_HOLDOUT=1").returncode, 2)  # BENCH_* is reserved
        self.update_config(holdout_split=None, rank_split=None)
        self.assertEqual(self.cli("metrics", "--run", a).returncode, 0)
        m = self.metrics_json(a)
        self.assertEqual((m["env_bench_holdout_split"], m["env_bench_rank_split"]), ("unset", "unset"))

    def test_dry_run_lists_the_bench_variables(self):
        r = self.cli("run", "--label", "d", "--dry-run")
        self.assertIn("BENCH_HOLDOUT_SPLIT=holdout", r.stdout)
        self.assertIn("BENCH_RANK_SPLIT=dev", r.stdout)
        self.assertNotIn("BENCH_SHOW_HOLDOUT", r.stdout)


# ---------------------------------------------------------------- code identity (provenance)

class CodeIdentityTests(FeatureCase):
    def setUp(self):
        super().setUp()
        self.use_param_scorer(primary_metric=None, criteria=None)
        self.scorer = Path(self.script("param_scorer.py", PARAM_SCORER))
        self.driver = self.proj / "bench_stub_driver.py"

    def test_compare_warns_when_the_driver_or_metrics_script_differ(self):
        a = self.run_ok("a", m={"q": 1})
        b = self.run_ok("b", m={"q": 1})
        self.assertNotIn("different code", self.cli("compare", a, b).stdout)
        self.driver.write_text(self.driver.read_text() + "\n# v2\n")
        c = self.run_ok("c", m={"q": 1})
        out = self.cli("compare", a, c).stdout
        self.assertIn("WARNING: compared runs used different code", out)
        self.assertIn("driver script", out)
        self.assertNotIn("metrics script", out)
        self.scorer.write_text(PARAM_SCORER + "\n# v2\n")
        d = self.run_ok("d", m={"q": 1})
        out = self.cli("compare", c, d).stdout
        self.assertIn("metrics script", out)
        self.assertNotIn("driver script", out)
        self.assertEqual(self.cli("compare", a, b).stdout.count("different code"), 0)

    def test_provenance_paths_are_part_of_the_code_identity(self):
        sut = self.proj / "sut"
        sut.mkdir()
        (sut / "m.py").write_text("v1\n")
        self.update_config(provenance=["sut"])
        a = self.run_ok("a", m={"q": 1})
        (sut / "m.py").write_text("v2\n")
        b = self.run_ok("b", m={"q": 1})
        self.assertIn("provenance paths", self.cli("compare", a, b).stdout)

    def test_runs_without_provenance_are_ignored(self):
        a = self.run_ok("a", m={"q": 1})
        run_json = self.hist / "runs" / a / "run.json"
        run = json.loads(run_json.read_text())
        del run["provenance"]
        run_json.write_text(json.dumps(run))
        self.driver.write_text(self.driver.read_text() + "\n# v2\n")
        b = self.run_ok("b", m={"q": 1})
        self.assertNotIn("different code", self.cli("compare", a, b).stdout)

    def test_sweep_summary_warns_about_an_older_run_with_different_code(self):
        old = self.run_ok("old", m={"q": 1})
        self.driver.write_text(self.driver.read_text() + "\n# v2\n")
        spec = {"name": "cw", "configs": [{"name": "x", "m": {"q": 1}}]}
        r = self.cli("sweep", "--spec", self.write_spec(spec), "--include-run", old)
        self.assertIn("WARNING: compared runs used different code", r.stdout)
        (summary_md,) = (self.hist / "sweeps").glob("*/summary.md")
        self.assertIn("different code", summary_md.read_text())
        spec["name"] = "cw2"
        r = self.cli("sweep", "--spec", self.write_spec(spec, "s2.json"))
        self.assertNotIn("different code", r.stdout)  # all runs of one sweep share their code

    def test_rescore_keeps_the_original_hash_and_records_the_new_one(self):
        a = self.run_ok("a", m={"q": 1})
        original = self.run_json(a)["provenance"]["metrics"]
        self.assertEqual(self.cli("metrics", "--run", a).returncode, 0)  # same script: nothing to record
        self.assertNotIn("rescored_provenance", self.run_json(a))
        self.scorer.write_text(PARAM_SCORER + "\n# v2\n")
        self.assertEqual(self.cli("metrics", "--run", a).returncode, 0)
        run = self.run_json(a)
        self.assertEqual(run["provenance"]["metrics"], original)  # never silently overwritten
        (new,) = [e for e in run["rescored_provenance"]["metrics"] if e["path_name"] == "param_scorer.py"]
        self.assertEqual(new["sha256"], hashlib_sha(self.scorer))
        self.assertNotEqual(new["sha256"], [e for e in original if e["path_name"] == "param_scorer.py"][0]["sha256"])
        b = self.run_ok("b", m={"q": 1})  # fresh run with the new scorer
        self.assertNotIn("metrics script", self.cli("compare", a, b).stdout)  # a is scored by the same code now
        self.scorer.write_text(PARAM_SCORER)  # back to the recorded script: the note disappears
        self.assertEqual(self.cli("metrics", "--run", a).returncode, 0)
        self.assertNotIn("rescored_provenance", self.run_json(a))
        self.assertIn("metrics script", self.cli("compare", a, b).stdout)

    def test_rescore_of_a_run_without_recorded_provenance(self):
        a = self.run_ok("a", m={"q": 1})
        run = self.run_json(a)
        del run["provenance"]
        (self.hist / "runs" / a / "run.json").write_text(json.dumps(run))
        self.assertEqual(self.cli("metrics", "--run", a).returncode, 0)
        run = self.run_json(a)
        self.assertNotIn("provenance", run)
        self.assertTrue(run["rescored_provenance"]["metrics"])


# ---------------------------------------------------------------- --group resolution

class GroupResolutionTests(FeatureCase):
    def test_base_label_and_prefix_resolve_like_run_ids(self):
        self.assertEqual(self.cli("run", "--label", "det", "--repeat", "2").returncode, 0)
        group = self.run_json(self.run_ids()[0])["group"]
        for command in ("stats", "determinism"):
            for token in ("det", group, group[:12]):
                r = self.cli(command, "--group", token)
                self.assertIn(r.returncode, (0, 1), (command, token, r.stderr))
                self.assertNotIn("bench: error", r.stderr, (command, token))
        self.assertIn("| wer | 2 |", self.cli("stats", "--group", "det").stdout)
        self.assertEqual(self.cli("determinism", "--group", "det").returncode, 0)
        r = self.cli("metrics", "--group", "det")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("re-scored 2 run(s)", r.stdout)
        self.assertEqual(self.cli("stats", "--group", "nosuch").returncode, 2)

    def test_two_groups_with_one_label_are_ambiguous_unless_latest(self):
        for _ in range(2):
            self.assertEqual(self.cli("run", "--label", "det", "--repeat", "2").returncode, 0)
        r = self.cli("stats", "--group", "det")
        self.assertEqual(r.returncode, 2)
        self.assertIn("ambiguous group 'det'", r.stderr)
        self.assertIn("--latest", r.stderr)
        out = self.tmp / "s.json"
        r = self.cli("stats", "--group", "det", "--latest", "--json-out", str(out))
        self.assertEqual(r.returncode, 0, r.stderr)
        groups = sorted({self.run_json(i)["group"] for i in self.run_ids()})
        newest = [i for i in self.run_ids() if self.run_json(i)["group"] == groups[-1]]
        self.assertEqual([x["run_id"] for x in json.loads(out.read_text())["runs"]], newest)
        self.assertIn("matches 2 groups", self.cli("determinism", "--group", "20").stderr)  # shared timestamp prefix

    def test_sweep_repeat_groups_resolve_by_sweep_and_config_name(self):
        spec = {"name": "sw", "repeats": 2, "configs": [{"name": "a"}, {"name": "b", "drop_every": 3}]}
        self.assertEqual(self.cli("sweep", "--spec", self.write_spec(spec)).returncode, 0)
        r = self.cli("stats", "--group", "sw__a")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("sw__a-r1", r.stdout)
        self.assertNotIn("sw__b", r.stdout)

    def test_resolve_group_unit(self):
        known = ["20261112T105549123Z_det", "20261112T105549999Z_det", "20261113T000000000Z_other"]
        self.assertEqual(runslib.resolve_group(known, "other"), known[2])
        self.assertEqual(runslib.resolve_group(known, "20261113"), known[2])
        self.assertEqual(runslib.resolve_group(known, "det", latest=True), known[1])
        with self.assertRaises(util.UsageError):
            runslib.resolve_group(known, "det")
        with self.assertRaises(util.UsageError) as cm:
            runslib.resolve_group(known, "othr")
        self.assertIn("other", str(cm.exception))


# ---------------------------------------------------------------- override type checks

class OverrideTypeTests(FeatureCase):
    DEFAULTS = {"drop_every": 0, "ratio": 0.5, "flag": False, "name": "x", "tags": [], "opts": {}, "nothing": None}

    def setUp(self):
        super().setUp()
        self.update_config(defaults=self.DEFAULTS, allow_new_keys=False)

    def test_mismatches_exit_2_and_nothing_runs(self):
        for text, needle in (("drop_every=abc", "expected int"), ("drop_every=1.5", "expected int"), ("drop_every=true", "expected int"),
                             ("drop_every=null", "expected int"), ("ratio=high", "expected float"), ("flag=1", "write true / false"),
                             ("flag=True", "expected bool"), ("tags=1", "expected list"), ("opts=[]", "expected dict")):
            with self.subTest(text):
                r = self.cli("run", "--label", "t", "--set", text)
                self.assertEqual(r.returncode, 2, r.stderr)
                self.assertIn(needle, r.stderr)
                self.assertIn(text.split("=")[0], r.stderr)
        self.assertEqual(self.run_ids(), [])
        r = self.cli("run", "--label", "t", "--set", "drop_every=x", "--set", "flag=2")
        self.assertIn("drop_every", r.stderr)
        self.assertIn("flag", r.stderr)  # all mismatches at once

    def test_accepted_values(self):
        for text in ("drop_every=3", "ratio=1", "ratio=0.25", "flag=true", "name=hello", "tags=[1, \"a\"]", "opts={\"k\": 1}",
                     "nothing=5", "nothing=\"s\"", "nothing=[1]"):
            r = self.cli("run", "--label", "ok", "--set", text)
            self.assertEqual(r.returncode, 0, (text, r.stderr))
        self.assertEqual(self.run_json(self.run_ids()[1])["config"]["ratio"], 1)  # an int is accepted where a float is expected

    def test_string_defaults_take_the_raw_text(self):
        for raw, expected in (("3", "3"), ("true", "true"), ("null", "null"), ("[1]", "[1]")):
            self.assertEqual(self.cli("run", "--label", "s", "--set", "name=" + raw).returncode, 0)
            self.assertEqual(self.run_json(self.run_ids()[-1])["config"]["name"], expected)
        self.assertEqual(cfglib.parse_sets(["name=3", "ratio=3"], self.DEFAULTS), {"name": "3", "ratio": 3})

    def test_new_keys_are_untyped(self):
        r = self.cli("run", "--label", "n", "--set", "brand_new=[1]", "--allow-new-keys")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.update_config(allow_new_keys=True)
        self.assertEqual(self.cli("run", "--label", "m", "--set", "also_new=true", "--set", "drop_every=2").returncode, 0)
        self.assertEqual(self.cli("run", "--label", "bad", "--set", "drop_every=two").returncode, 2)  # known keys stay typed

    def test_sweep_overrides_are_checked_up_front(self):
        for where, spec in {
            "base": {"name": "s", "base": {"drop_every": "2"}, "configs": [{"name": "a"}]},
            "config": {"name": "s", "configs": [{"name": "a"}, {"name": "b", "ratio": "x"}]},
            "matrix": {"name": "s", "matrix": {"flag": [True, 1]}},
            "str default": {"name": "s", "matrix": {"name": [1, 2]}},
        }.items():
            with self.subTest(where):
                r = self.cli("sweep", "--spec", self.write_spec(spec))
                self.assertEqual(r.returncode, 2, r.stderr)
                self.assertIn("wrong type", r.stderr)
                self.assertFalse(self.hist.exists())
        ok = {"name": "s", "base": {"ratio": 1}, "configs": [{"name": "a", "drop_every": 2}], "matrix": {"flag": [True, False]}}
        self.assertEqual(self.cli("sweep", "--spec", self.write_spec(ok)).returncode, 0)

    def test_type_helpers(self):
        cfg = {"defaults": self.DEFAULTS}
        self.assertEqual(cfglib.override_type_errors(cfg, {"ratio": 1, "tags": [], "unknown": object()}), [])
        self.assertEqual(len(cfglib.override_type_errors(cfg, {"drop_every": True, "flag": 0})), 2)
        self.assertEqual([cfglib.type_name(v) for v in (None, True, 1, 1.5, "s", [], {})],
                         ["null", "bool", "int", "float", "str", "list", "dict"])


# ---------------------------------------------------------------- sweep ties and inert parameters

class SweepTieTests(FeatureCase):
    def sweep(self, spec, *extra):
        return self.cli("sweep", "--spec", self.write_spec(spec), *extra)

    def summary(self):
        d = max((self.hist / "sweeps").iterdir(), key=lambda p: (p / "summary.json").stat().st_mtime_ns)  # the newest sweep
        return json.loads((d / "summary.json").read_text()), (d / "summary.md").read_text()

    def test_best_line_names_ties_and_summary_flags_inert_parameters(self):
        spec = {"name": "ti", "configs": [{"name": "kw"}, {"name": "a", "marker": 1}, {"name": "b", "marker": 2},
                                          {"name": "c", "marker": 3}, {"name": "worse", "drop_every": 3}]}
        r = self.sweep(spec)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("best by wer: kw (tied with 3 others: a, b, c)", r.stdout)
        self.assertIn("possibly inert parameter(s): `marker`", r.stdout)
        summary, md = self.summary()
        (inert,) = summary["possibly_inert"]
        self.assertEqual((inert["configs"], inert["keys"]), (["kw", "a", "b", "c"], ["marker"]))
        self.assertIn("NOTE: possibly inert parameter(s): `marker` (configs kw, a, b, c have identical results", md)
        self.assertNotIn("worse", next(l for l in md.splitlines() if "possibly inert" in l))

    def test_one_tie_uses_the_singular_and_distinct_results_are_not_flagged(self):
        r = self.sweep({"name": "one", "configs": [{"name": "kw"}, {"name": "twin", "marker": 1}, {"name": "worse", "drop_every": 3}]})
        self.assertIn("best by wer: kw (tied with 1 other: twin)", r.stdout)
        r = self.sweep({"name": "none", "configs": [{"name": "kw"}, {"name": "worse", "drop_every": 3}]})
        self.assertIn("best by wer: kw\n", r.stdout)
        self.assertNotIn("inert", r.stdout)
        summary, md = self.summary()
        self.assertEqual(summary["possibly_inert"], [])
        self.assertNotIn("inert", md)

    def test_many_ties_are_truncated(self):
        configs = [{"name": "c%02d" % i, "marker": i} for i in range(12)]
        r = self.sweep({"name": "many", "configs": configs})
        line = next(l for l in r.stdout.splitlines() if l.startswith("best by"))
        self.assertIn("tied with 11 others: c01, c02", line)
        self.assertTrue(line.endswith(", ...)"), line)

    def test_repeats_and_repeat_baseline_do_not_cause_false_flags(self):
        r = self.sweep({"name": "rp", "repeats": 2, "repeat_baseline": True, "configs": [{"name": "kw"}, {"name": "worse", "drop_every": 3}]})
        self.assertEqual(r.returncode, 0, r.stderr)
        summary, _ = self.summary()
        self.assertEqual(summary["possibly_inert"], [])  # kw__repeat has the very same config: nothing differs
        r = self.sweep({"name": "rp2", "repeats": 2, "configs": [{"name": "kw"}, {"name": "same", "marker": 1}]})
        summary, _ = self.summary()
        self.assertEqual(summary["possibly_inert"][0]["keys"], ["marker"])
        self.assertIn("tied with 1 other: same", r.stdout)

    def test_inert_detection_ignores_hidden_holdout_metrics(self):
        self.use_param_scorer(holdout_split="holdout", primary_metric={"name": "f1", "goal": "max"}, criteria=None, index_columns=None)
        spec = {"name": "hg", "configs": [{"name": "x", "m": {"train.f1": 0.5, "holdout.f1": 0.1}},
                                          {"name": "y", "m": {"train.f1": 0.5, "holdout.f1": 0.9}, "marker": 1}]}
        self.sweep(spec)
        summary, _ = self.summary()
        self.assertEqual(summary["possibly_inert"][0]["configs"], ["x", "y"])  # identical on the visible metrics; holdout is not consulted


# ---------------------------------------------------------------- init: .gitignore

class InitGitignoreTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="bench-gi-")
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def init(self, name, template="text"):
        return subprocess.run([sys.executable, str(SCRIPTS / "bench.py"), "init", name, "--template", template], cwd=str(self.tmp),
                              capture_output=True, text=True, timeout=60)

    def test_init_writes_a_gitignore_and_uses_the_readme_label(self):
        r = self.init("fresh")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual((self.tmp / "fresh" / ".gitignore").read_text().splitlines(), ["history/", "__pycache__/"])
        self.assertIn(".gitignore: added history/, __pycache__/", r.stdout)
        self.assertIn("run --label baseline --config fresh/bench.json", r.stdout)
        self.assertNotIn("smoke", r.stdout)

    def test_existing_gitignore_only_gets_the_missing_lines(self):
        outputs = {}
        for name, before, after in (("a", "node_modules/\nhistory\n", "node_modules/\nhistory\n__pycache__/\n"),
                                    ("b", "*.log", "*.log\nhistory/\n__pycache__/\n"),
                                    ("c", "history/\n__pycache__/\n", "history/\n__pycache__/\n")):
            (self.tmp / name).mkdir()
            (self.tmp / name / ".gitignore").write_text(before)
            r = self.init(name)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual((self.tmp / name / ".gitignore").read_text(), after, name)
            outputs[name] = r.stdout
        self.assertIn(".gitignore: added __pycache__/", outputs["a"])
        self.assertNotIn(".gitignore:", outputs["c"])  # nothing to add, nothing said


# ---------------------------------------------------------------- end to end: classification + holdout guard

LABEL_DRIVER = '''import json, os
from pathlib import Path
run_dir = Path(os.environ["BENCH_RUN_DIR"])
cfg = json.loads(Path(os.environ["BENCH_CONFIG"]).read_text())
outputs = {}
for path in sorted(Path(os.environ["BENCH_FIXTURE_ITEMS"]).glob("*.json")):
    item = json.loads(path.read_text())
    words = set(item["input"].split())
    if words & {"good", "great", "fun"}:
        outputs[item["id"]] = "pos"
    elif words & {"bad", "awful", "sad"}:
        outputs[item["id"]] = "neg"
    else:
        outputs[item["id"]] = "neu" if cfg.get("use_neutral", True) else "pos"
tmp = run_dir / ".result.json.tmp"
tmp.write_text(json.dumps({"schema_version": 1, "status": "success", "outputs": outputs, "latency_ms": 1.5}))
os.replace(tmp, run_dir / "result.json")
'''
LABEL_ITEMS = [  # (id, split, text, gold): the rule classifier is perfect on train and poor on holdout
    ("t1", "train", "good great", "pos"), ("t2", "train", "great fun", "pos"), ("t3", "train", "bad awful", "neg"),
    ("t4", "train", "awful bad", "neg"), ("t5", "train", "meh okay", "neu"), ("t6", "train", "okay meh", "neu"),
    ("t7", "train", "good fun", "pos"), ("t8", "train", "bad sad", "neg"),
    ("h1", "holdout", "not good", "neg"), ("h2", "holdout", "not bad", "pos"), ("h3", "holdout", "fine", "neu"),
    ("h4", "holdout", "so so good", "neu"),
]


class ClassificationHoldoutEndToEndTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="bench-clf-")
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.proj = self.tmp / "proj"
        self.run_py(str(SCRIPTS / "bench.py"), "init", str(self.proj), "--template", "items", cwd=self.tmp)
        for f in list((self.proj / "fixtures" / "items").iterdir()) + list((self.proj / "fixtures" / "gold").iterdir()):
            f.unlink()
        splits = {}
        for iid, split, text, gold in LABEL_ITEMS:
            (self.proj / "fixtures" / "items" / (iid + ".json")).write_text(json.dumps({"id": iid, "input": text}))
            (self.proj / "fixtures" / "gold" / (iid + ".json")).write_text(json.dumps({"id": iid, "expected": gold}))
            splits.setdefault(split, []).append(iid)
        (self.proj / "fixtures" / "splits.json").write_text(json.dumps(splits))
        (self.proj / "bench_stub_driver.py").write_text(LABEL_DRIVER)
        conf = json.loads((self.proj / "bench.json").read_text())
        conf.update(defaults={"use_neutral": True}, index_columns=None)  # show EVERY metric key the guard lets through
        (self.proj / "bench.json").write_text(json.dumps(conf))

    def run_py(self, *args, cwd=None):
        r = subprocess.run([sys.executable, *args], cwd=str(cwd or self.proj), capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, (args, r.stderr))
        return r

    def pb(self, *args, ok=True):
        r = subprocess.run([sys.executable, str(self.proj / "bench.py"), *args], cwd=str(self.proj), capture_output=True, text=True,
                           timeout=120)
        if ok:
            self.assertEqual(r.returncode, 0, (args, r.stderr))
        return r

    def run_dir(self, label):
        (d,) = [p for p in (self.proj / "history" / "runs").iterdir() if p.name.endswith("_" + label)]
        return d

    def test_scalar_labels_with_a_holdout_never_leak_the_holdout_unless_asked(self):
        r = self.pb("run", "--label", "base")
        self.pb("run", "--label", "no-neutral", "--set", "use_neutral=false")
        d = self.run_dir("base")
        metrics = json.loads((d / "metrics.json").read_text())
        self.assertEqual(metrics["train.accuracy"], 1.0)
        self.assertAlmostEqual(metrics["train.macro_f1"], 1.0)
        self.assertEqual(metrics["holdout.accuracy"], 0.25)
        unsplit = [k for k in metrics if "." not in k and k != "latency_ms"]
        self.assertEqual(unsplit, [])  # the stub emits split-prefixed quality metrics only
        self.assertFalse([k for k in metrics if k.startswith("all.")])
        self.assertNotIn("warning", r.stderr)
        # per-run reports leave the holdout out by default
        for name in ("metrics.md", "diff.md"):
            text = (d / name).read_text()
            for needle in ("h1", "h4", "holdout |", "| holdout"):
                self.assertNotIn(needle, text, name)
        self.assertIn("Confusion matrix", (d / "metrics.md").read_text())
        self.assertIn("| train | 8 |", (d / "metrics.md").read_text())
        # index.md and compare: split-prefixed non-holdout metrics only
        index_header = table_header((self.proj / "history" / "index.md").read_text())
        compare = self.pb("compare", "base", "no-neutral").stdout
        for text in (index_header, table_header(compare)):
            self.assertIn("train.accuracy", text)
            self.assertIn("train.macro_f1", text)
            self.assertNotIn("holdout", text)
            self.assertNotIn(" accuracy ", text + " ")
            self.assertNotIn("all.", text)
        # criteria (f1 >= 0.8, exact_rate >= 0.5) are evaluated on train.*
        cr = json.loads((d / "run.json").read_text())["criteria_result"]
        self.assertTrue(cr["pass"])
        self.assertEqual({k: v["metric"] for k, v in cr["details"].items()}, {"f1": "train.f1", "exact_rate": "train.exact_rate"})
        bad = json.loads((self.run_dir("no-neutral") / "run.json").read_text())["criteria_result"]
        self.assertEqual(bad["details"]["exact_rate"]["value"], 0.75)
        self.assertFalse(bad["pass"])  # f1 of the degraded train split < 0.8
        # sweeps rank on the rank split and show no holdout
        spec = self.tmp / "spec.json"
        spec.write_text(json.dumps({"name": "sw", "configs": [{"name": "neutral"}, {"name": "plain", "use_neutral": False}]}))
        sweep = self.pb("sweep", "--spec", str(spec))
        self.assertIn("best by train.f1: neutral", sweep.stdout)
        (sweep_dir,) = list((self.proj / "history" / "sweeps").iterdir())
        self.assertNotIn("holdout", table_header((sweep_dir / "summary.md").read_text()))
        # raw data stays complete: index.json has every key and marks the hidden ones
        rows = {r["label"]: r for r in json.loads((self.proj / "history" / "index.json").read_text())}
        self.assertEqual(rows["base"]["metrics"]["holdout.accuracy"], 0.25)
        self.assertIn("holdout.accuracy", rows["base"]["hidden_metrics"])
        # --show-holdout reveals the holdout in compare ...
        shown = table_header(self.pb("compare", "base", "no-neutral", "--show-holdout").stdout)
        self.assertIn("holdout.accuracy", shown)
        # ... and `metrics --show-holdout` makes the metrics command write holdout details and all.* too
        self.pb("metrics", "--run", "base", "--show-holdout")
        metrics = json.loads((d / "metrics.json").read_text())
        self.assertAlmostEqual(metrics["all.accuracy"], 9 / 12)
        text = (d / "metrics.md").read_text() + (d / "diff.md").read_text()
        for needle in ("holdout", "h1", "h4", "all"):
            self.assertIn(needle, text)
        self.assertTrue(json.loads((d / "run.json").read_text())["show_holdout"])
        # all.* is an unsplit key: still hidden by default, visible with --show-holdout
        compare = self.pb("compare", "base", "no-neutral")
        self.assertNotIn("all.", table_header(compare.stdout))
        self.assertIn("all.accuracy", compare.stderr)  # ... and the one-time warning names it
        self.assertIn("all.accuracy", table_header(self.pb("compare", "base", "no-neutral", "--show-holdout").stdout))
        # a plain re-score removes the holdout details again
        self.pb("metrics", "--run", "base")
        self.assertNotIn("h1", (d / "diff.md").read_text())
        self.assertNotIn("all.accuracy", (d / "metrics.json").read_text())
        # items.json is raw per-item data and always complete
        items = json.loads((d / "items.json").read_text())["items"]
        self.assertEqual(sum(1 for i in items.values() if i["split"] == "holdout"), 4)

    def test_dev_split_as_rank_split_in_the_items_template(self):
        splits = {"train": ["t1", "t2", "t3", "t4"], "dev": ["t5", "t6", "t7", "t8"], "holdout": ["h1", "h2", "h3", "h4"]}
        (self.proj / "fixtures" / "splits.json").write_text(json.dumps(splits))
        conf = json.loads((self.proj / "bench.json").read_text())
        conf.update(rank_split="dev")
        (self.proj / "bench.json").write_text(json.dumps(conf))
        self.pb("run", "--label", "base")
        self.pb("run", "--label", "no-neutral", "--set", "use_neutral=false")
        metrics = json.loads((self.run_dir("base") / "metrics.json").read_text())
        for split in splits:
            self.assertIn(split + ".accuracy", metrics)
        self.assertEqual((metrics["train.accuracy"], metrics["dev.accuracy"], metrics["holdout.accuracy"]), (1.0, 1.0, 0.25))
        cr = json.loads((self.run_dir("base") / "run.json").read_text())["criteria_result"]
        self.assertEqual({k: v["metric"] for k, v in cr["details"].items()}, {"f1": "dev.f1", "exact_rate": "dev.exact_rate"})
        compare = self.pb("compare", "base", "no-neutral").stdout
        self.assertIn("Ranked by `dev.f1`", compare)
        header = table_header(compare)
        self.assertIn("train.accuracy", header)  # every non-holdout split stays visible
        self.assertIn("dev.accuracy", header)
        self.assertNotIn("holdout", header)
        self.assertEqual(self.pb("compare", "base", "no-neutral", "--sort", "holdout.f1", ok=False).returncode, 2)
        report = (self.run_dir("base") / "metrics.md").read_text()
        self.assertIn("| dev | 4 |", report)
        self.assertNotIn("| holdout |", report)

    def test_dict_gold_still_uses_the_extraction_scoring(self):
        self.run_py(str(SCRIPTS / "bench.py"), "init", str(self.tmp / "dictproj"), "--template", "items", cwd=self.tmp)
        out = subprocess.run([sys.executable, str(self.tmp / "dictproj" / "bench.py"), "run", "--label", "x"],
                             cwd=str(self.tmp / "dictproj"), capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        (d,) = list((self.tmp / "dictproj" / "history" / "runs").iterdir())
        m = json.loads((d / "metrics.json").read_text())
        self.assertNotIn("train.accuracy", m)  # classification metrics only for plain-string gold
        self.assertIn("train.f1", m)
        self.assertNotIn("Confusion", (d / "metrics.md").read_text())


if __name__ == "__main__":
    unittest.main()
