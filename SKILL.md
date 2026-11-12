---
name: reference-benchmark
description: >
  Build and run a reproducible reference benchmark that measures how well a pipeline,
  model, prompt or algorithm performs against a fixed fixture and a reference
  ("ground truth"), then compares configurations over time. Use when the user wants to
  test, measure, benchmark, A/B-compare, tune or regression-check output quality of
  something like transcription/diarization, OCR, extraction, classification, LLM
  summaries/prompts, ranking or on-device performance — "wie gut ist X", "messen",
  "Referenz-Benchmark", "Qualität vergleichen", "Parameter-Sweep". Not for plain unit tests.
---

# Reference benchmark

Goal: replace "it looks better" with a number you can reproduce, a diff that shows
**where** it is wrong, and a history that shows what changed between runs.

Reply in the user's language. Write the harness and its docs in the project's language
conventions.

## Core loop

```
fixture + reference ──► driver (system under test, one config) ──► result.json
                                                                      │
history/<run_id>/ ◄── run.json (config, fixture hashes, status) ◄─────┤
        │                                                             ▼
        ├── metrics.json / metrics.md ◄── metrics script (deterministic)
        ├── diff.md  (per-segment: where it is wrong)
        └── index.md (one row per run)  ◄── sweep (config matrix, one change at a time)
```

## Phase 0 — Decide what the benchmark must answer (ask, don't assume)

Settle with the user, in one short round:

1. **Decision question.** "Is variant B good enough to ship?", "Which of these 3 models?",
   "Did the upgrade regress?" A benchmark without a decision is a hobby.
2. **System under test and how to drive it headlessly.** The real code path, not a
   re-implementation. Prefer an existing debug/diagnostic entry point; if none exists,
   add a minimal one behind a debug flag (file in → result file out, no UI interaction).
3. **Fixture(s).** Representative input; synthetic/fictional when the domain is
   sensitive. Note the size and why it is representative.
4. **Reference and its quality.** Hand-verified gold, or a "silver" reference produced by
   another system? With silver, every metric means *agreement with that system* — say so
   in every report header.
5. **Pass criteria**, written down before the first run (e.g. "recall ≥ 80 %, zero
   invented facts, ≤ 5 min on device").
6. **Where history lives.** Outside the repo when fixtures or outputs are sensitive
   (`~/<project>-benchmarks/<suite>/`, dirs 0700, files 0600). Never commit real data.

If an earlier benchmark for the same question exists, extend it — do not build a second one.

**No reference yet?** Start with a reference-free A/B diagnostic: run the real pipeline
twice on the same input with one component swapped, reuse every shared stage (so the
variants differ only in the swapped part), and report a cheap difference score (e.g.
1 − Dice over token multisets) plus input health stats and severity-ranked findings.
That is enough to locate *which stage* is wrong; build the reference benchmark once you
need to quantify *how much better* a fix is.

## Phase 1 — Pin the fixture and reference

- Store fixture + reference paths in a config; record **name, bytes, sha256** of each in
  every `run.json`, so a changed fixture can never silently invalidate comparisons.
- Write the **reference parser and normalizer** first and unit-test it: case, punctuation,
  number words, whitespace, label names. Document every rule in the report header.
- For generated-text tasks: write the fact checklist / expected structure per fixture
  **before** reading any outputs. Keep a **holdout** split that is never used for tuning.

## Phase 2 — Driver

- One command runs one config end to end and writes a machine-readable `result.json`
  (outputs + per-stage timings + environment telemetry). No interactive steps.
- `result.json` carries `schema_version` and `status: success|error` (with
  `error.code/message`) so "the system failed" is distinguishable from "the harness
  failed". Validate it strictly. Write it atomically (temp file + rename) and poll for
  the final name only.
- Every later stage (metrics, diff, index) is a pure function of files on disk and can be
  re-run offline from the raw result — never re-run the expensive driver to fix a metric.
- If the system emits arbitrary labels (`spk0`, `cluster_3`), either have it emit the
  reference's labels or add an explicit, reported mapping step in the metrics.
- Config is explicit and complete (every tunable the run used, including defaults), passed
  as a file/flags — never "whatever the app currently has".
- Collect environment before/after: device/model, OS, app/build version, repo commit +
  dirty flag, thermal/battery/memory where relevant.
- **Liveness + timeout:** detect a dead or hung run (process gone, no progress for N s)
  and mark it `timeout`/`killed` instead of waiting forever.
- **Exclusive-resource guard:** if a device, GPU or service can be used by others
  (other sessions, CI, other agents), check it is quiet before starting, and on
  interference wait and **retry** (`--retries`), recording `attempt` and a `killed_by_hint`.
- Retry only **transient, externally caused** failures (killed by someone else, resource
  busy). A system `error` or a `timeout` is a result, not noise — record it, don't retry.
  On kill/timeout save evidence (logs around the event) next to the run.
- If the driver changes shared state (device settings, env, services), save the original
  and restore it exactly once in a cleanup trap, also on Ctrl-C.
- `--dry-run` prints what would execute and touches nothing.
- **Stub driver** (`--driver <cmd>`): a fake that writes a canned `result.json`, so the
  harness can be tested and dry-run without the real system.

## Phase 3 — Metrics (deterministic, explain themselves)

- Pick from [references/metrics-catalog.md](references/metrics-catalog.md): one primary,
  a few diagnostic, plus cost metrics. Report error components separately (S/D/I, not just WER).
- When the reference lacks information for a formal metric (e.g. no segment end times →
  no real DER), compute an honest **proxy** and label it a proxy.
- Metrics read only `result.json` + reference → same inputs, same numbers. Output
  `metrics.json` (flat, for the index) and `metrics.md` (for humans).
- **Automatic scorers and LLM judges must be calibrated** before you trust them: score a
  set of runs by hand, compare (MAE, bias, max error, per-item accuracy), tune thresholds
  with cross-validation, and publish the calibration table. Keep the human score as the
  tie-breaker for close calls.

## Phase 4 — Diff: where, not how often

Aggregate numbers say *that* something is worse; the diff says *where*. Produce `diff.md`
per run: aligned per segment (turn, page, record, query), segment-level metric, inline
markup (`[-deleted-]`, `[+inserted+]`, `[ref→hyp]`), and highlighted error regions (e.g.
swap regions). Read the diff before proposing any fix — it often shows the error is not
where the aggregate suggests.

## Phase 5 — History, index, compare

- Each run gets `run_id = <UTC timestamp>_<label>` and a directory with `run.json`,
  `result.json`, `metrics.*`, `diff.*`, raw telemetry. Statuses: `success`, `error`
  (system reported failure), `timeout`, `killed`, `busy`, `metrics_failed`. Every attempt
  gets its own dir with `attempt`/`attempts_allowed`; failed runs stay in history.
- `index` rebuilds `index.json` + `index.md` from the run dirs (derived state; tolerate
  partial or legacy dirs) after **every** attempt: one row per run, columns = config keys
  + metric keys + status/attempt. `compare A B …` shows deltas against the first run.
- Manifests, index and sweep summaries contain **no content** (no transcript text, no
  outputs) — only config, hashes, numbers. Content stays in the per-run artifacts.

## Phase 6 — Sweeps

- A sweep spec (JSON) names the sweep and lists configs as overrides of a base config
  (optionally a small cartesian matrix). Labels become `<sweep>__<config>`.
- **Change one variable at a time** relative to a baseline; re-run the baseline inside the
  sweep when conditions (device, temperature, build) may have drifted.
- Run configs sequentially on a shared resource, with a cooldown between them; a failing
  config does not abort the sweep. Attribute output to a config by the run id that run
  actually created (never "latest dir with a matching name"), so a failed config never
  inherits an older run.
- Don't change a production default because one fixture improved; a sweep winner is a
  candidate for a broader check, not a decision.
- Check determinism once: run the baseline twice. Identical quality → one run per config
  is enough; otherwise repeat and report spread.
- Long sweeps run in the background as a direct child of the shell; never let a verifier
  or helper agent start the real driver (it may kill the running sweep on abort) — agents
  get dry-run/stub only.

## Phase 7 — Test the harness itself

Unit-test parser, normalizer, alignment, metrics on tiny hand-made cases with known
answers; test driver/history/sweep end to end with the stub driver. Do this before
trusting the first real number.

## Phase 8 — Report and remember

- Per finding: config A vs B, primary metric delta, what the diff shows, recommendation,
  and whether it passes the Phase-0 criteria. Inert parameters ("no change at all") are
  findings too — they close a search direction.
- Record durable results (what was measured, on which fixture/commit, what was decided,
  dead ends) in the project's docs or memory, so nobody repeats a dead experiment.

## Bundled tools (optional, stdlib Python ≥ 3.9)

- `scripts/bench.py` — generic harness wrapping any project driver + metrics command via a
  `bench.json` config. Subcommands: `init` (starter config + stub driver/metrics + sample
  fixtures), `run` (`--label`, `--set k=v`, `--retries`, `--timeout-seconds`, `--driver`,
  `--dry-run`), `sweep --spec`, `metrics --run ID` (re-score offline), `index`, `compare`.
  Contract: the driver gets `BENCH_RUN_DIR`, `BENCH_CONFIG`, `BENCH_FIXTURE_<NAME>` and
  writes `result.json`; exit 75 = transient/killed (retried). The metrics command writes a
  flat `metrics.json` (+ optional `metrics.md`, `diff.md`). Full contract: `bench.py -h`.
- `scripts/textdiff.py` — exact token alignment (S/D/I, WER; bit-parallel, fast on 10k+
  tokens), per-segment reports, attributed WER with label mapping and mismatch regions,
  top errors, inline diff markup. Importable or CLI (`textdiff.py REF HYP --markdown-out F`).
- Self-test: `python3 -m unittest discover -s scripts/tests` (67 tests, ~30 s).

Quick start in a project: `python3 <skill>/scripts/bench.py init bench/`, then replace the
stub driver with one that runs the real system and the stub metrics with task metrics.

Use them when the project has no harness yet; when it has one, follow and extend the
project's own. Copy the scripts into the project (e.g. `scripts/bench/`) rather than
calling them from the skill directory, so the project stays self-contained.

## Cleanup

Stop drivers, emulators, preview servers and background sweeps you started; delete stub
outputs and scratch runs that are not part of the history; keep the history directory.
