"""End-to-end tests for bench.py using `bench.py init` and its stub driver / stub metrics.

Everything runs in temp dirs with the CLI as a subprocess (so umask, signals and exit codes are
real); a few unit-level checks import bench directly. No file is left behind and no stub driver
survives a test (killed-process checks use the pid the stub writes to pid.txt).
"""
import contextlib
import hashlib
import io
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
import bench  # noqa: E402

BENCH = str(SCRIPTS / "bench.py")
GUARD_PY = '''import pathlib, sys
p = pathlib.Path(sys.argv[1]); codes = sys.argv[2:]
n = int(p.read_text()) if p.exists() else 0
p.write_text(str(n + 1))
sys.exit(int(codes[n]) if n < len(codes) else 0)
'''
FAKE_DRIVER = '''import json, os
open(os.environ["BENCH_RUN_DIR"] + "/result.json", "w").write(json.dumps(
    {"schema_version": 1, "status": "success", "output": "the quick brown fox jumped over the lazy dog today"}))
'''


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class BenchCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="bench-test-")
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.proj = self.tmp / "proj"
        self.hist = self.tmp / "history"  # not created until a command needs it
        self.assertEqual(self.bench("init", str(self.proj), config=False).returncode, 0)
        self.conf = self.proj / "bench.json"
        self.update_config(history_root=str(self.hist))

    # ---- helpers
    def update_config(self, **changes):
        cfg = json.loads(self.conf.read_text())
        cfg.update(changes)
        self.conf.write_text(json.dumps({k: v for k, v in cfg.items() if v is not None}))

    def bench(self, command, *args, config=True, timeout=60):
        cmd = [sys.executable, BENCH, command] + (["--config", str(self.conf)] if config else []) + list(args)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def run_label(self, label, *extra):
        return self.bench("run", "--label", label, *extra)

    def run_ids(self):
        runs = self.hist / "runs"
        return sorted(p.name for p in runs.iterdir()) if runs.is_dir() else []

    def run_json(self, run_id):
        return json.loads((self.hist / "runs" / run_id / "run.json").read_text())

    def run_for(self, label):
        ids = [i for i in self.run_ids() if self.run_json(i)["label"] == label]
        self.assertEqual(len(ids), 1, (label, ids))
        return ids[0]

    def only_run(self):
        ids = self.run_ids()
        self.assertEqual(len(ids), 1, ids)
        return ids[0]

    def write_spec(self, spec, name="spec.json"):
        path = self.tmp / name
        path.write_text(json.dumps(spec))
        return str(path)

    def assert_private(self, root):
        for dirpath, dirnames, filenames in os.walk(root):
            for name in dirnames:
                self.assertEqual(stat.S_IMODE(os.stat(os.path.join(dirpath, name)).st_mode), 0o700, os.path.join(dirpath, name))
            for name in filenames:
                self.assertEqual(stat.S_IMODE(os.stat(os.path.join(dirpath, name)).st_mode), 0o600, os.path.join(dirpath, name))
        self.assertEqual(stat.S_IMODE(os.stat(root).st_mode), 0o700)

    def interrupt_when_driver_runs(self, args, pid_glob, sig):
        """Start bench.py, send `sig` once a stub driver has written its pid; -> (returncode, pid, pid_file)."""
        cmd = [sys.executable, BENCH, args[0], "--config", str(self.conf)] + list(args[1:])
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        pid_file = None
        try:
            deadline = time.monotonic() + 20
            while pid_file is None and time.monotonic() < deadline:
                for f in (self.hist / "runs").glob(pid_glob):
                    if f.read_text().strip():
                        pid_file = f
                time.sleep(0.05)
            self.assertIsNotNone(pid_file, "driver never started")
            pid = int(pid_file.read_text())
            proc.send_signal(sig)
            proc.communicate(timeout=30)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        if pid_alive(pid):  # only reached when the test is failing: do not leave the stub running
            self.addCleanup(os.kill, pid, signal.SIGKILL)
        return proc.returncode, pid, pid_file

    def script(self, name, text):
        path = self.tmp / name
        path.write_text(text)
        return str(path)


class InitAndConfigTests(BenchCase):
    def test_init_refuses_to_overwrite(self):
        before = self.conf.read_text()
        r = self.bench("init", str(self.proj), config=False)
        self.assertEqual(r.returncode, 2)
        self.assertIn("refusing to overwrite", r.stderr)
        self.assertEqual(self.conf.read_text(), before)
        for name in ("bench_stub_driver.py", "bench_stub_metrics.py", "fixtures/input.txt", "fixtures/reference.txt"):
            self.assertTrue((self.proj / name).exists(), name)

    def test_usage_and_config_errors_exit_2(self):
        self.assertEqual(self.run_label("bad label").returncode, 2)
        self.assertEqual(self.run_label("-lead").returncode, 2)
        self.assertEqual(self.run_label("x", "--set", "novalue").returncode, 2)
        self.assertEqual(self.run_label("x", "--retries", "-1").returncode, 2)
        self.assertEqual(self.bench("run", "--label", "x", "--config", str(self.tmp / "nope.json"), config=False).returncode, 2)
        (self.tmp / "nohist.json").write_text("{}")
        self.assertEqual(self.bench("index", "--config", str(self.tmp / "nohist.json"), config=False).returncode, 2)
        (self.proj / "fixtures" / "input.txt").unlink()
        r = self.run_label("x")
        self.assertEqual(r.returncode, 2)
        self.assertIn("fixture", r.stderr)
        self.assertEqual(self.run_ids(), [])

    def test_help_documents_the_contract(self):
        out = self.bench("--help", config=False).stdout
        for text in ("BENCH_RUN_DIR", "BENCH_CONFIG", "BENCH_FIXTURE_<NAME>", "result.json", "75", "metrics_failed"):
            self.assertIn(text, out)

    def test_set_values_are_json_or_string(self):
        self.assertEqual(self.run_label("sets", "--set", "drop_every=3", "--set", "name=foo", "--set", "flag=true",
                                        "--set", "xs=[1, 2]").returncode, 0)
        self.assertEqual(self.run_json(self.only_run())["config"],
                         {"drop_every": 3, "name": "foo", "flag": True, "xs": [1, 2]})


class RunTests(BenchCase):
    def test_dry_run_creates_nothing(self):
        before = sorted(str(p) for p in self.tmp.rglob("*"))
        r = self.run_label("dry", "--dry-run", "--set", "drop_every=2")
        self.assertEqual(r.returncode, 0, r.stderr)
        for text in ("DRY RUN", "_dry", "BENCH_RUN_DIR=", "BENCH_CONFIG=", "BENCH_FIXTURE_INPUT=", "BENCH_FIXTURE_REFERENCE=",
                     '"drop_every": 2', "bench_stub_driver.py"):
            self.assertIn(text, r.stdout)
        self.assertFalse(self.hist.exists())
        self.assertEqual(sorted(str(p) for p in self.tmp.rglob("*")), before)

    def test_success_run_records_everything(self):
        r = self.run_label("ok")
        self.assertEqual(r.returncode, 0, r.stderr)
        rid = self.only_run()
        self.assertRegex(rid, r"^\d{8}T\d{6}Z_ok$")
        run = self.run_json(rid)
        self.assertEqual((run["schema_version"], run["run_id"], run["label"], run["suite"]), (1, rid, "ok", "starter"))
        self.assertEqual((run["status"], run["error_code"], run["attempt"], run["attempts_allowed"]), ("success", None, 1, 1))
        self.assertEqual(run["config"], {"drop_every": 0})
        for name in ("input", "reference"):
            path = self.proj / "fixtures" / ("%s.txt" % name)
            self.assertEqual(run["fixtures"][name], {"path_name": path.name, "bytes": path.stat().st_size,
                                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        self.assertTrue(run["repo"] is None or "commit" in run["repo"])
        self.assertEqual(set(run["host"]), {"platform", "python"})
        self.assertIsInstance(run["duration_ms"], int)
        self.assertTrue(run["started_at"].endswith("Z") and run["finished_at"].endswith("Z"))
        raw = (self.hist / "runs" / rid / "run.json").read_text()
        self.assertNotIn("quick", raw)  # no output content in run.json
        self.assertNotIn(str(self.proj), raw)  # fixtures are recorded by basename only
        run_dir = self.hist / "runs" / rid
        for name in ("config.json", "driver.log", "result.json", "metrics.json", "metrics.md", "diff.md", "run.json"):
            self.assertTrue((run_dir / name).exists(), name)
        self.assertAlmostEqual(json.loads((run_dir / "metrics.json").read_text())["wer"], 0.2)
        index = json.loads((self.hist / "index.json").read_text())
        self.assertEqual([(row["run_id"], row["status"]) for row in index], [(rid, "success")])
        self.assertEqual(index[0]["metrics"]["S"], 1)
        self.assertIn(rid, (self.hist / "index.md").read_text())
        self.assert_private(self.hist)

    def test_fixture_hash_changes_with_content(self):
        self.run_label("a")
        (self.proj / "fixtures" / "input.txt").write_text("something else\n")
        self.run_label("b")
        a, b = (self.run_json(i)["fixtures"]["input"]["sha256"] for i in self.run_ids())
        self.assertNotEqual(a, b)

    def test_run_id_collision_gets_numeric_suffix(self):
        cfg = bench.load_config(self.conf)
        with mock.patch.object(bench, "stamp", return_value="20260101T000000Z"), contextlib.redirect_stdout(io.StringIO()):
            for _ in range(3):
                self.assertEqual(bench.execute_run(cfg, "lab", {})["status"], "success")
        self.assertEqual(self.run_ids(), ["20260101T000000Z_lab", "20260101T000000Z_lab-2", "20260101T000000Z_lab-3"])
        self.assertEqual(self.run_json("20260101T000000Z_lab-3")["run_id"], "20260101T000000Z_lab-3")

    def test_system_error_is_a_result_not_retried(self):
        r = self.run_label("err", "--set", "fail=error", "--retries", "2")
        self.assertEqual(r.returncode, 1)
        rid = self.only_run()  # no retry
        run = self.run_json(rid)
        self.assertEqual((run["status"], run["error_code"]), ("error", "stub_error"))
        self.assertFalse((self.hist / "runs" / rid / "metrics.json").exists())
        self.assertTrue((self.hist / "runs" / rid / "result.json").exists())

    def test_driver_crash_without_result(self):
        r = self.run_label("crash", "--set", "fail=crash")
        self.assertEqual(r.returncode, 1)
        run = self.run_json(self.only_run())
        self.assertEqual((run["status"], run["error_code"]), ("error", "driver_exit_3"))

    def test_missing_binary_and_invalid_results(self):
        self.assertEqual(self.run_label("nobin", "--driver", str(self.tmp / "does-not-exist")).returncode, 1)
        self.assertEqual(self.run_json(self.run_for("nobin"))["error_code"], "driver_exit_127")
        self.assertIn("cannot start", (self.hist / "runs" / self.run_for("nobin") / "driver.log").read_text())
        self.assertEqual(self.run_label("empty", "--driver", "python3 -c pass").returncode, 1)
        self.assertEqual(self.run_json(self.run_for("empty"))["error_code"], "no_result")
        bad = self.script("bad_driver.py", 'import os; open(os.environ["BENCH_RUN_DIR"] + "/result.json", "w").write(\'{"status": "success"}\')')
        self.assertEqual(self.run_label("invalid", "--driver", "python3 " + bad).returncode, 1)
        self.assertEqual(self.run_json(self.run_for("invalid"))["error_code"], "invalid_result")

    def test_driver_override_with_canned_result(self):
        fake = self.script("fake_driver.py", FAKE_DRIVER)
        r = self.run_label("fake", "--driver", "%s %s" % (sys.executable, fake))
        self.assertEqual(r.returncode, 0, r.stderr)
        run_dir = self.hist / "runs" / self.only_run()
        self.assertEqual(json.loads((run_dir / "metrics.json").read_text())["wer"], 0.0)

    def test_killed_is_retried_and_each_attempt_has_its_own_dir(self):
        r = self.run_label("flaky", "--set", "fail=killed", "--set", "fail_attempts=1", "--retries", "1")
        self.assertEqual(r.returncode, 0, r.stderr)
        first, second = self.run_ids()
        self.assertNotEqual(first, second)
        a, b = self.run_json(first), self.run_json(second)
        self.assertEqual((a["status"], a["attempt"], a["attempts_allowed"], a["error_code"]), ("killed", 1, 2, "driver_exit_75"))
        self.assertEqual((b["status"], b["attempt"], b["attempts_allowed"]), ("success", 2, 2))
        self.assertFalse((self.hist / "runs" / first / "metrics.json").exists())
        self.assertEqual([row["status"] for row in json.loads((self.hist / "index.json").read_text())], ["killed", "success"])

    def test_killed_without_retries_left_exits_3(self):
        self.assertEqual(self.run_label("k0", "--set", "fail=killed").returncode, 3)
        self.assertEqual(len(self.run_ids()), 1)
        r = self.run_label("k1", "--set", "fail=killed", "--retries", "1")
        self.assertEqual(r.returncode, 3)
        self.assertEqual(len(self.run_ids()), 3)  # k0 + both attempts of k1

    def test_timeout_kills_the_driver_and_is_not_retried(self):
        t0 = time.monotonic()
        r = self.run_label("slow", "--set", "fail=sleep:30", "--timeout-seconds", "1", "--retries", "2")
        self.assertLess(time.monotonic() - t0, 20)
        self.assertEqual(r.returncode, 1)
        rid = self.only_run()  # timeout is a result: no retries
        run = self.run_json(rid)
        self.assertEqual((run["status"], run["error_code"]), ("timeout", "driver_timeout"))
        pid = int((self.hist / "runs" / rid / "pid.txt").read_text())
        if pid_alive(pid):  # only reached when the test is failing: do not leave the stub running
            self.addCleanup(os.kill, pid, signal.SIGKILL)
        self.assertFalse(pid_alive(pid))
        self.assertTrue((self.hist / "runs" / rid / "driver.log").exists())
        self.assertFalse((self.hist / "runs" / rid / "metrics.json").exists())

    def test_ctrl_c_and_sigterm_stop_the_driver_and_record_the_run(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(sig=sig.name):
                label = "int-" + sig.name.lower()
                rc, pid, pid_file = self.interrupt_when_driver_runs(
                    ["run", "--label", label, "--set", "fail=sleep:30"], "*_%s/pid.txt" % label, sig)
                self.assertEqual(rc, 130)
                self.assertFalse(pid_alive(pid))
                run = self.run_json(pid_file.parent.name)
                self.assertEqual((run["status"], run["error_code"]), ("killed", "interrupted"))
        index = json.loads((self.hist / "index.json").read_text())
        self.assertEqual([r["status"] for r in index], ["killed", "killed"])


class GuardTests(BenchCase):
    def test_busy_guard_never_starts_the_driver(self):
        self.update_config(guard=[sys.executable, "-c", "import sys; sys.exit(1)"], guard_max_wait_seconds=0.5,
                           guard_poll_seconds=0.1)
        r = self.run_label("busy")
        self.assertEqual(r.returncode, 4)
        rid = self.only_run()
        run = self.run_json(rid)
        self.assertEqual((run["status"], run["error_code"]), ("busy", "guard_timeout"))
        self.assertFalse((self.hist / "runs" / rid / "driver.log").exists())
        self.assertFalse((self.hist / "runs" / rid / "result.json").exists())
        self.assertFalse((self.hist / "stub_counter.txt").exists())  # the stub driver never ran

    def test_busy_is_retried(self):
        self.update_config(guard=[sys.executable, "-c", "import sys; sys.exit(1)"], guard_max_wait_seconds=0.2,
                           guard_poll_seconds=0.1)
        self.assertEqual(self.run_label("busy2", "--retries", "1").returncode, 4)
        self.assertEqual([self.run_json(i)["attempt"] for i in self.run_ids()], [1, 2])

    def test_guard_is_polled_until_it_passes(self):
        counter = self.tmp / "guard-count"
        guard = self.script("guard.py", GUARD_PY)
        self.update_config(guard=[sys.executable, guard, str(counter), "1", "1"], guard_poll_seconds=0.1,
                           guard_max_wait_seconds=10)
        self.assertEqual(self.run_label("polled").returncode, 0)
        self.assertEqual(counter.read_text(), "3")  # two failures, then a pass (quiet = 0 -> a single good check)

    def test_guard_must_stay_quiet_for_the_whole_period(self):
        counter = self.tmp / "guard-count"
        guard = self.script("guard.py", GUARD_PY)
        # ok, ok, FAIL, then ok forever: the quiet period restarts at the first good check after the failure.
        # Without the restart the guard would pass at check 4 (~0.3 s); with it, not before ~0.55 s / check 5.
        self.update_config(guard=[sys.executable, guard, str(counter), "0", "0", "1"], guard_poll_seconds=0.1,
                           guard_quiet_seconds=0.25, guard_max_wait_seconds=10)
        self.assertEqual(self.run_label("quiet").returncode, 0)
        self.assertGreaterEqual(int(counter.read_text()), 5)
        self.assertGreaterEqual(self.run_json(self.only_run())["duration_ms"], 500)


class MetricsTests(BenchCase):
    def test_metrics_failure_keeps_result_and_rescore_repairs_it(self):
        good = json.loads(self.conf.read_text())["metrics"]
        self.update_config(metrics=[sys.executable, "-c", "import sys; sys.exit(7)"])
        r = self.run_label("m")
        self.assertEqual(r.returncode, 1)
        rid = self.only_run()
        run_dir = self.hist / "runs" / rid
        run = self.run_json(rid)
        self.assertEqual((run["status"], run["error_code"]), ("metrics_failed", "metrics_exit_7"))
        self.assertTrue((run_dir / "result.json").exists())
        self.assertFalse((run_dir / "metrics.json").exists())
        self.assertEqual(json.loads((self.hist / "index.json").read_text())[0]["status"], "metrics_failed")

        self.update_config(metrics=good)  # "fix" the metrics command, then re-score only that stage
        calls = (self.hist / "stub_counter.txt").read_text()
        r = self.bench("metrics", "--run", rid)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual((self.hist / "stub_counter.txt").read_text(), calls)  # the driver did not run again
        run = self.run_json(rid)
        self.assertEqual((run["status"], run["error_code"]), ("success", None))
        self.assertIn("rescored_at", run)
        self.assertAlmostEqual(json.loads((run_dir / "metrics.json").read_text())["wer"], 0.2)
        row = json.loads((self.hist / "index.json").read_text())[0]
        self.assertEqual(row["status"], "success")
        self.assertAlmostEqual(row["metrics"]["wer"], 0.2)
        self.assertEqual(self.bench("rescore", "--run", rid).returncode, 0)  # alias
        self.assert_private(self.hist)

    def test_metrics_without_output_file_or_with_nested_values_fail(self):
        self.update_config(metrics=[sys.executable, "-c", "pass"])
        self.run_label("nofile")
        self.assertEqual(self.run_json(self.run_for("nofile"))["error_code"], "metrics_missing")
        nested = "import os; open(os.environ['BENCH_RUN_DIR'] + '/metrics.json', 'w').write('{\"a\": {\"b\": 1}}')"
        self.update_config(metrics=[sys.executable, "-c", nested])
        self.run_label("nested")
        self.assertEqual(self.run_json(self.run_for("nested"))["error_code"], "metrics_invalid")

    def test_rescore_needs_a_successful_result(self):
        self.run_label("err", "--set", "fail=error")
        self.assertEqual(self.bench("metrics", "--run", self.only_run()).returncode, 2)
        self.assertEqual(self.bench("metrics", "--run", "nope").returncode, 2)


class IndexTests(BenchCase):
    def test_index_tolerates_partial_and_legacy_dirs(self):
        self.run_label("real")
        runs = self.hist / "runs"
        (runs / "junk").mkdir()
        legacy = runs / "20200101T000000Z_legacy"
        legacy.mkdir()
        (legacy / "config.json").write_text(json.dumps({"drop_every": "a|b\nc"}))
        (legacy / "metrics.json").write_text(json.dumps({"wer": 0.5}))
        corrupt = runs / "20200101T000001Z_corrupt"
        corrupt.mkdir()
        (corrupt / "run.json").write_text("{not json")
        (corrupt / "metrics.json").write_text("[1, 2]")
        r = self.bench("index")
        self.assertEqual(r.returncode, 0, r.stderr)
        rows = json.loads((self.hist / "index.json").read_text())
        self.assertEqual([x["run_id"] for x in rows], sorted(x["run_id"] for x in rows))
        by_id = {x["run_id"]: x for x in rows}
        self.assertEqual(len(rows), 4)
        self.assertEqual((by_id["junk"]["label"], by_id["junk"]["status"]), ("junk", "unknown"))
        leg = by_id["20200101T000000Z_legacy"]
        self.assertEqual((leg["label"], leg["status"], leg["metrics"]["wer"]), ("legacy", "unknown", 0.5))
        self.assertEqual(by_id["20200101T000001Z_corrupt"]["status"], "unknown")
        md = (self.hist / "index.md").read_text()
        self.assertIn("a\\|b<br>c", md)  # | and newlines are escaped in table cells
        self.assertEqual(sum(1 for line in md.splitlines() if line.startswith("| ")), 5)  # header + 4 rows

    def test_default_columns_are_the_sorted_union(self):
        self.update_config(index_columns=None)
        self.run_label("one", "--set", "zeta=1")
        self.run_label("two", "--set", "alpha=2")
        header = next(l for l in (self.hist / "index.md").read_text().splitlines() if l.startswith("| run_id"))
        cols = [c.strip() for c in header.strip("|").split("|")]
        self.assertEqual(cols, ["run_id", "label", "status", "attempt", "alpha", "drop_every", "zeta",
                                "D", "I", "N_hyp", "N_ref", "S", "wer"])

    def test_empty_history_and_column_name_clash(self):
        self.assertEqual(self.bench("index").returncode, 0)
        self.assertEqual(json.loads((self.hist / "index.json").read_text()), [])
        self.update_config(index_columns={"config": ["wer"], "metrics": ["wer"]})
        self.run_label("clash", "--set", "wer=9")
        header = next(l for l in (self.hist / "index.md").read_text().splitlines() if l.startswith("| run_id"))
        self.assertIn("config.wer", header)
        self.assertIn("metrics.wer", header)

    def test_md_table_escaping(self):
        text = bench.md_table(["a|b", "c"], [["x|y", None], ["line1\nline2", 0.123456789]])
        self.assertIn("a\\|b", text)
        self.assertIn("x\\|y | – |", text)
        self.assertIn("line1<br>line2 | 0.123457 |", text)


class CompareTests(BenchCase):
    def setUp(self):
        super().setUp()
        self.run_label("a")
        self.run_label("b", "--set", "drop_every=3")
        self.a, self.b = self.run_for("a"), self.run_for("b")

    def test_deltas_are_relative_to_the_first_run(self):
        out = self.tmp / "cmp.json"
        md_out = self.tmp / "cmp.md"
        r = self.bench("compare", self.a, self.b, "--json-out", str(out), "--markdown-out", str(md_out))
        self.assertEqual(r.returncode, 0, r.stderr)
        comp = json.loads(out.read_text())
        self.assertEqual(comp["baseline"], self.a)
        base, other = comp["rows"]
        self.assertIsNone(base["deltas"]["wer"])
        wer_a, wer_b = base["values"]["wer"], other["values"]["wer"]
        self.assertAlmostEqual(other["deltas"]["wer"], wer_b - wer_a)
        self.assertGreater(other["deltas"]["wer"], 0)
        self.assertEqual(comp["config_differences"], [{"key": "drop_every", "values": {self.a: 0, self.b: 3}}])
        self.assertIn("Δwer", r.stdout)
        self.assertIn("+0.3", r.stdout)
        self.assertEqual(md_out.read_text().strip(), r.stdout.strip())
        for p in (out, md_out):
            self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o600)

    def test_baseline_is_whatever_is_listed_first(self):
        out = self.tmp / "rev.json"
        self.bench("compare", self.b, self.a, "--json-out", str(out))
        comp = json.loads(out.read_text())
        self.assertEqual(comp["baseline"], self.b)
        self.assertLess(comp["rows"][1]["deltas"]["wer"], 0)

    def test_columns_option_and_unknown_run(self):
        out = self.tmp / "cols.json"
        self.assertEqual(self.bench("compare", self.a, self.b, "--columns", "wer,drop_every", "--json-out", str(out)).returncode, 0)
        comp = json.loads(out.read_text())
        self.assertEqual(comp["columns"], ["drop_every", "wer"])
        self.assertEqual(comp["delta_columns"], ["wer"])
        r = self.bench("compare", self.a, "20990101T000000Z_nope")
        self.assertEqual(r.returncode, 2)
        self.assertIn("unknown run id", r.stderr)

    def test_no_config_differences(self):
        r = self.bench("compare", self.a, self.a)
        self.assertEqual(r.returncode, 0)
        self.assertIn("None.", r.stdout)


class SweepTests(BenchCase):
    def sweep(self, spec, *extra):
        return self.bench("sweep", "--spec", self.write_spec(spec), *extra)

    def summary(self):
        dirs = sorted((self.hist / "sweeps").iterdir())
        self.assertEqual(len(dirs), 1)
        return json.loads((dirs[0] / "summary.json").read_text()), dirs[0]

    def test_matrix_expansion_and_summary(self):
        spec = {"name": "sw", "base": {"drop_every": 0, "extra": "x"}, "configs": [{"name": "base"}],
                "matrix": {"drop_every": [2, 3]}, "cooldown_seconds": 0}
        r = self.sweep(spec)
        self.assertEqual(r.returncode, 0, r.stderr)
        summary, sweep_dir = self.summary()
        self.assertEqual([x["config"] for x in summary["results"]], ["base", "drop_every-2", "drop_every-3"])
        self.assertEqual([x["label"] for x in summary["results"]], ["sw__base", "sw__drop_every-2", "sw__drop_every-3"])
        self.assertTrue(all(x["status"] == "success" for x in summary["results"]))
        for res in summary["results"]:  # attribution by the exact run id the run function returned
            self.assertEqual(self.run_json(res["run_id"])["label"], res["label"])
        self.assertEqual(self.run_json(summary["results"][1]["run_id"])["config"], {"drop_every": 2, "extra": "x"})
        comp = summary["comparison"]
        self.assertEqual(comp["baseline"], summary["results"][0]["run_id"])
        self.assertIsNone(comp["rows"][0]["deltas"]["wer"])
        self.assertIsNotNone(comp["rows"][2]["deltas"]["wer"])
        self.assertEqual([d["key"] for d in comp["config_differences"]], ["drop_every"])
        md = (sweep_dir / "summary.md").read_text()
        self.assertIn("sw__drop_every-3", md)
        self.assertIn("Failed configs: none", md)
        self.assert_private(self.hist)
        self.assertEqual(len(json.loads((self.hist / "index.json").read_text())), 3)

    def test_matrix_names_are_sanitized_cartesian_products(self):
        spec = bench.parse_sweep_spec(self.write_spec({"name": "m", "matrix": {"model": ["a b", "c/d"], "threads": [1, 2.5]}}))
        names = [n for n, _ in spec["entries"]]
        self.assertEqual(names, ["model-a-b_threads-1", "model-a-b_threads-2.5", "model-c-d_threads-1", "model-c-d_threads-2.5"])
        self.assertTrue(all(bench.LABEL_RE.match(n) for n in names))
        self.assertEqual(spec["entries"][1][1], {"model": "a b", "threads": 2.5})

    def test_validation_happens_before_anything_runs(self):
        bad_specs = {
            "duplicate": {"name": "s", "configs": [{"name": "ok"}, {"name": "ok"}]},
            "label": {"name": "s", "configs": [{"name": "ok"}, {"name": "bad name"}]},
            "empty": {"name": "s", "configs": []},
            "sweep name": {"name": "bad name!", "configs": [{"name": "ok"}]},
            "matrix clash": {"name": "s", "configs": [{"name": "drop_every-2"}], "matrix": {"drop_every": [2]}},
            "empty matrix list": {"name": "s", "configs": [{"name": "ok"}], "matrix": {"drop_every": []}},
            "unknown key": {"name": "s", "configs": [{"name": "ok"}], "repeat_basline": True},
            "negative cooldown": {"name": "s", "configs": [{"name": "ok"}], "cooldown_seconds": -1},
        }
        for title, spec in bad_specs.items():
            with self.subTest(title):
                r = self.sweep(spec)
                self.assertEqual(r.returncode, 2, r.stdout)
                self.assertIn("invalid sweep spec", r.stderr)
                self.assertFalse(self.hist.exists())  # nothing ran, not even the valid first config
        self.assertEqual(self.bench("sweep", "--spec", str(self.tmp / "missing.json")).returncode, 2)

    def test_failing_config_does_not_stop_the_sweep_or_inherit_older_runs(self):
        self.assertEqual(self.run_label("fs__bad").returncode, 0)  # an older, successful run with the same label
        old = self.only_run()
        spec = {"name": "fs", "configs": [{"name": "ok1"}, {"name": "bad", "fail": "error"}, {"name": "ok2"}]}
        r = self.sweep(spec)
        self.assertEqual(r.returncode, 1)
        summary, sweep_dir = self.summary()
        self.assertEqual([x["status"] for x in summary["results"]], ["success", "error", "success"])
        bad = summary["results"][1]
        self.assertNotEqual(bad["run_id"], old)
        self.assertEqual((bad["error_code"], self.run_json(bad["run_id"])["status"]), ("stub_error", "error"))
        row = summary["comparison"]["rows"][1]
        self.assertEqual(row["status"], "error")
        self.assertIsNone(row["values"]["wer"])  # no metrics borrowed from the older successful run
        self.assertIn("Failed configs: bad", (sweep_dir / "summary.md").read_text())
        self.assertEqual(len(self.run_ids()), 4)

    def test_repeat_baseline_and_include_run(self):
        self.assertEqual(self.run_label("old").returncode, 0)
        old = self.only_run()
        spec = {"name": "rp", "repeat_baseline": True, "configs": [{"name": "a"}, {"name": "b", "drop_every": 2}]}
        r = self.sweep(spec, "--include-run", old)
        self.assertEqual(r.returncode, 0, r.stderr)
        summary, _ = self.summary()
        self.assertEqual([x["label"] for x in summary["results"]], ["rp__a", "rp__b", "rp__a__repeat"])
        self.assertEqual(summary["include_runs"], [old])
        rows = summary["comparison"]["rows"]
        self.assertEqual([x["run_id"] for x in rows][0], old)
        self.assertEqual(len(rows), 4)
        self.assertEqual(summary["comparison"]["baseline"], old)
        self.assertEqual(len(self.run_ids()), 4)
        n = len(self.run_ids())
        r = self.sweep(spec, "--include-run", "20990101T000000Z_nope")
        self.assertEqual(r.returncode, 2)
        self.assertEqual(len(self.run_ids()), n)

    def test_cooldown_is_between_runs_only(self):
        spec = {"name": "cd", "cooldown_seconds": 7, "repeat_baseline": True,
                "configs": [{"name": "a"}, {"name": "b"}]}
        sleeps = []
        args = bench.build_parser().parse_args(["sweep", "--config", str(self.conf), "--spec", self.write_spec(spec)])
        with mock.patch.object(bench, "_sleep", side_effect=sleeps.append), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(args.func(args), 0)
        self.assertEqual(sleeps, [7, 7])  # 3 runs -> 2 gaps, none after the last
        spec["cooldown_seconds"] = 0
        sleeps.clear()
        args = bench.build_parser().parse_args(["sweep", "--config", str(self.conf), "--spec", self.write_spec(spec, "s2.json")])
        with mock.patch.object(bench, "_sleep", side_effect=sleeps.append), contextlib.redirect_stdout(io.StringIO()):
            args.func(args)
        self.assertEqual(sleeps, [])

    def test_retries_timeout_driver_and_dry_run_pass_through(self):
        spec = {"name": "pt", "configs": [{"name": "flaky", "fail": "killed", "fail_attempts": 1}]}
        r = self.sweep(spec, "--retries", "1")
        self.assertEqual(r.returncode, 0, r.stderr)
        summary, _ = self.summary()
        res = summary["results"][0]
        self.assertEqual(len(self.run_ids()), 2)
        self.assertEqual(res["run_id"], self.run_ids()[-1])  # the retry's run id, not the killed attempt's
        self.assertEqual(res["status"], "success")

        before = sorted(str(p) for p in self.tmp.rglob("*"))
        r = self.sweep({"name": "dr", "configs": [{"name": "a"}, {"name": "b"}]}, "--dry-run")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.count("DRY RUN"), 2)
        self.assertIn("dr__b", r.stdout)
        self.assertEqual(sorted(str(p) for p in self.tmp.rglob("*")), before)

        fake = self.script("fake_driver.py", FAKE_DRIVER)
        r = self.sweep({"name": "fk", "configs": [{"name": "a", "fail": "error"}]}, "--driver", "%s %s" % (sys.executable, fake))
        self.assertEqual(r.returncode, 0, r.stderr)  # the override driver ignores the stub-only "fail" key

    def test_ctrl_c_stops_the_sweep_but_keeps_a_summary(self):
        spec = self.write_spec({"name": "ci", "configs": [{"name": "a"}, {"name": "b", "fail": "sleep:30"}, {"name": "c"}]})
        rc, pid, _ = self.interrupt_when_driver_runs(["sweep", "--spec", spec], "*_ci__b/pid.txt", signal.SIGINT)
        self.assertEqual(rc, 130)
        self.assertFalse(pid_alive(pid))
        summary, sweep_dir = self.summary()
        self.assertEqual([(x["config"], x["status"]) for x in summary["results"]], [("a", "success"), ("b", "killed")])
        self.assertEqual(len(self.run_ids()), 2)  # "c" never started
        self.assertIn("Failed configs: b", (sweep_dir / "summary.md").read_text())

    def test_timeout_in_a_sweep_counts_as_failure(self):
        r = self.sweep({"name": "to", "configs": [{"name": "slow", "fail": "sleep:30"}, {"name": "fast"}]},
                       "--timeout-seconds", "1")
        self.assertEqual(r.returncode, 1)
        summary, _ = self.summary()
        self.assertEqual([x["status"] for x in summary["results"]], ["timeout", "success"])
        pid = int((self.hist / "runs" / summary["results"][0]["run_id"] / "pid.txt").read_text())
        if pid_alive(pid):
            self.addCleanup(os.kill, pid, signal.SIGKILL)
        self.assertFalse(pid_alive(pid))


if __name__ == "__main__":
    unittest.main()
