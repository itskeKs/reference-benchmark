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
5. **Primary metric and pass criteria**, written down before the first run (e.g.
   "recall ≥ 80 %, zero invented facts, ≤ 5 min on device"). With `bench.py` these go
   into `bench.json` as `primary_metric` and `criteria`, so every run gets a pass column.
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
- Many small cases (invoices, queries, notes) are **items** of one fixture set: one
  directory of inputs, one of gold, and a `splits.json` assigning each item to `train` or
  `holdout`. Tune on train; look at holdout only for the finally chosen config.
- If the system **learns from gold** (a classifier, a fitted ranker), its train score is
  memorized fit, not quality: rank on a `dev` split or cross-validation instead, and make
  sure the driver only reads gold for training items — never for holdout.
- Report quality metrics per split (`train.f1`, `holdout.f1`). A metric over all items
  silently includes holdout and leaks it.
- Make sure every tunable can actually change the outcome on your fixture: an option
  that never triggers (e.g. date order when no day is above 12) looks "inert" only
  because the fixture lacks the case. Add the case.

## Phase 2 — Driver

For a pure function (a parser, a ranker, a prompt against a local model), the driver is a
few lines that run it over the fixture and write `result.json`; skip the device-oriented
points below (guard, liveness, telemetry, state restore). They matter for phones,
GPUs, shared services and long runs.

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
- For many-item fixtures, also write a per-item result (`items.json`: id, split, score,
  pass, detail) so runs can be compared item by item, not just in aggregate.
- **Automatic scorers and LLM judges must be calibrated** before you trust them:
  1. Split items into folds **before** looking at any output (deterministic seed).
  2. Hand-score a blinded sample — the rater must not see which run or config produced
     an output, nor the automatic score. Aim for ≥ 30 scored units (item × run); say
     which unit you use (per item-run is usually right; per run hides compensating errors).
  3. Compare auto vs. human: MAE, bias (sign matters — a scorer that is always generous
     inflates every run), max abs error, share within a tolerance, correlation, and
     agreement for pass/fail verdicts.
  4. Tune scorer rules/thresholds on one fold only, report the other fold as held out.
  5. Calibrate screens for forbidden content separately: keyword screens over-fire on
     quoted or negated context ("he did *not* mention salary"), so measure their
     precision too, not just recall.
  Publish the calibration table next to the results; keep the human score as the
  tie-breaker for close calls.

## Phase 4 — Diff: where, not how often

Aggregate numbers say *that* something is worse; the diff says *where*. Produce `diff.md`
per run: aligned per segment (turn, page, record, query), segment-level metric, inline
markup (`[-deleted-]`, `[+inserted+]`, `[ref→hyp]`), and highlighted error regions (e.g.
swap regions). For structured outputs: per record, which field is wrong, with expected vs.
actual and the source context. Read the diff before proposing any fix — it often shows the
error is not where the aggregate suggests.

Between two runs, compare **per item**: which items regressed, improved or flipped
pass/fail. An unchanged average can hide a regression on exactly the items that matter.

## Phase 5 — History, index, compare

- Each run gets `run_id = <UTC timestamp>_<label>` and a directory with `run.json`,
  `result.json`, `metrics.*`, `diff.*`, raw telemetry. Statuses: `success`, `error`
  (system reported failure), `timeout`, `killed`, `busy`, `metrics_failed`. Every attempt
  gets its own dir with `attempt`/`attempts_allowed`; failed runs stay in history.
- `index` rebuilds `index.json` + `index.md` from the run dirs (derived state; tolerate
  partial or legacy dirs) after **every** attempt: one row per run, columns = config keys
  + metric keys + status/attempt. `compare A B …` shows deltas against the first run.
- Re-scoring an old run never rewrites what it was run on: if a fixture changed since,
  refuse (or record the drift explicitly) and keep the previous metrics. Compare only
  runs with the same fixture fingerprint, or say loudly that they differ.
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
- Parameters that only apply to one variant (k1/b only for BM25) are not a cartesian
  product: list those combinations explicitly instead of sweeping a full matrix.
- Rank by the primary metric; on ties prefer the config closest to the current default,
  then the cheapest one. Report ties as ties.
- Don't change a production default because one fixture improved; a sweep winner is a
  candidate for a broader check, not a decision.
- Check determinism once: run the baseline twice. Identical quality → one run per config
  is enough; otherwise repeat and report spread. Timing is always noisy: repeat timing
  measurements (or time many iterations inside the driver) and report mean ± sd, never
  a single number; process wall time of the whole run is not a latency measurement.
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

## Bundled tools (optional, stdlib Python ≥ 3.9, no dependencies)

Use them when the project has no harness yet; when it has one, follow and extend the
project's own. Start with `init`, which **copies** the tools into the project so it stays
self-contained:

```
python3 <skill-dir>/scripts/bench.py init bench/ --template items   # or --template text
```

- `text` template: one input text vs. one reference text (WER-style, uses `textdiff.py`).
- `items` template: many items + gold + `splits.json`, per-item scoring (field-level
  P/R/F1 and exact match for records; accuracy, macro-F1, per-class P/R/F1 and a
  confusion matrix for single labels), `items.json`, per-item diff. Use it for extraction,
  classification, ranking, checklists — anything with many small cases.

Then replace `bench_stub_driver.py` with a driver for the real system and
`bench_stub_metrics.py` with task metrics (see the metrics catalog).

**Contract.** bench runs the driver and the metrics command with cwd = the `bench.json`
directory and env `BENCH_RUN_DIR`, `BENCH_CONFIG` (merged config JSON), and
`BENCH_FIXTURE_<NAME>` per fixture. The driver writes `result.json`
(`schema_version`, `status: success|error`); exit 75 means transient/killed and is
retried. The metrics command writes a flat `metrics.json`, optionally `metrics.md`,
`diff.md`, `diff.json` and `items.json`. Full details: `bench.py -h`, `bench.py sweep --explain`.

**`bench.json` keys:** `suite`, `history_root`, `fixtures`, `driver`, `metrics`,
`defaults` (every tunable; `--set` and sweeps reject unknown keys unless
`allow_new_keys`, and values whose type differs from the default), `timeout_seconds`, `retries`, `guard` (+ `guard_quiet_seconds`,
`guard_poll_seconds`, `guard_max_wait_seconds`), `index_columns`, `provenance` (extra paths
to hash as code identity), `primary_metric` (`{"name", "goal": "max"|"min"}`), `criteria`
(`{"f1": ">=0.8", "invented": "==0"}` → pass column), `timing_keys` (excluded from the
determinism check, shown as mean ± sd), `holdout_split` (hides `holdout.*` and every
metric not namespaced to a visible split unless `--show-holdout`), `rank_split` (split
used for ranking and criteria, default `train`; use `dev`/`cv` for systems that fit on
gold), `public_metrics` (unsplit metrics that may stay visible, globs allowed).

**Split conventions.** Metrics scripts emit split-prefixed quality metrics (`train.f1`,
`dev.f1`, `holdout.f1`); `all.*` is reserved for aggregates over every item. Criteria and
sweep ranking use `<rank_split>.<metric>`. bench passes `BENCH_HOLDOUT_SPLIT` and
`BENCH_RANK_SPLIT` to the driver and metrics; a fitting driver reads gold only for
training ids. Metrics scripts omit holdout detail from `metrics.md`/`diff.md` unless
`BENCH_SHOW_HOLDOUT=1`, which bench sets only for `metrics RUN --show-holdout` — the final
check on the chosen config, followed by `compare … --show-holdout`.

**Commands** (run ids accept a unique prefix or a label; `--latest` picks the newest):

| Command | Purpose |
|---|---|
| `run --label L [--set k=v] [--repeat N] [--retries N] [--timeout-seconds S] [--driver CMD] [--dry-run]` | one config end to end |
| `sweep --spec F` / `sweep --explain` | run a config list and/or matrix; ranked summary |
| `metrics RUN… / --all / --label-glob P / --group G / --sweep NAME [--env K=V] [--force] [--show-holdout]` | re-score offline; previous results move to `prev/`; refuses on fixture drift without `--force` |
| `index` | rebuild `index.json` / `index.md` |
| `compare RUN… [--sort M] [--top N] [--columns …] [--show-holdout]` | ranked table, deltas vs. first run, warnings on fixture or code changes |
| `items A B [--split S] [--threshold X]` | per-item regressed / improved / unchanged |
| `stats --group G` / `determinism RUN… \| --group G` | mean, sd, cv for repeats / exit 1 if quality metrics differ (`--group` takes the run label) |

`scripts/textdiff.py`: exact token alignment (S/D/I, WER; bit-parallel, fast on 10k+
tokens), per-segment reports, attributed WER with label mapping and mismatch regions,
top errors, inline markup. Importable or CLI.

`scripts/calibrate.py`: `sheet` builds a blinded, shuffled hand-scoring CSV from
`items.json` of several runs (the key stays in a separate private file); `score` compares
auto vs. human scores (MAE, bias, max error, within-tolerance share, r, pass agreement),
optionally per fold.

Self-test: `python3 -m unittest discover -s scripts/tests` (167 tests, about 80 s).

## Cleanup

Stop drivers, emulators, preview servers and background sweeps you started; delete stub
outputs and scratch runs that are not part of the history; keep the history directory.
