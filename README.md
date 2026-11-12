# reference-benchmark

A [Claude Code](https://claude.com/claude-code) skill for building **reproducible
reference benchmarks**. It measures how well a pipeline, model, prompt or algorithm does
against fixed inputs and a reference ("ground truth"). It then compares configurations
over time, shows *where* the errors are, and keeps a history.

It works for any task where an output can be checked against an expectation:

- speech-to-text and speaker attribution, OCR, translation (text vs. reference text)
- field extraction and classification (structured output vs. gold records)
- search and ranking (results vs. relevance judgments)
- LLM summaries and reports (output vs. a fact checklist, with invented-content gates)
- performance on a device or service (latency, memory, thermal state)

The method came out of a real on-device speech pipeline benchmark. It was then tested on
field extraction, search ranking and fact-checklist scoring before release.

## What you get

- **`SKILL.md`**: the method Claude follows. It covers the decision question, fixture
  and reference, a headless driver, deterministic metrics, a per-segment or per-item diff,
  run history, sweeps, a determinism check, scorer calibration and reporting.
- **`references/metrics-catalog.md`**: which metrics fit which kind of output.
- **`scripts/bench.py`**: a generic harness with no dependencies. It wraps *your* driver
  and metrics commands. It handles run directories with config and fixture hashes,
  statuses, retries, timeouts, an exclusive-resource guard, an index, ranked comparisons,
  per-item diffs, train/dev/holdout splits with holdout hiding, repeats with mean ± sd,
  pass/fail criteria, and warnings when fixtures or code changed between runs. History files are
  private (0700/0600).
- **`scripts/textdiff.py`**: exact token alignment (S/D/I, WER), per-segment reports,
  attributed WER with label mapping, and inline diff markup.
- **`scripts/calibrate.py`**: blinded hand-scoring sheets and an auto-vs-human
  calibration report.

Everything is standard-library Python 3.9 or newer.

## Install

Clone into your personal skills directory so the skill is available in every project:

```bash
git clone https://github.com/itskeKs/reference-benchmark ~/.claude/skills/reference-benchmark
```

For a single project, clone it into `<project>/.claude/skills/reference-benchmark`
instead.

Claude loads the skill when you ask things like "how good is X", "benchmark these two
models", "did the upgrade regress quality" or "sweep these parameters". You can also
invoke it directly with `/reference-benchmark`.

## Use the harness without Claude

```bash
python3 ~/.claude/skills/reference-benchmark/scripts/bench.py init bench --template items
```

```bash
python3 bench/bench.py run --label baseline --config bench/bench.json
```

```bash
python3 bench/bench.py sweep --explain
```

`init` copies the tools into your project, together with a stub driver, stub metrics and
sample fixtures. Replace the stubs with your own driver and metrics. `python3 bench.py -h`
documents the full contract.

## Tests

```bash
python3 -m unittest discover -s scripts/tests
```

## License

MIT. See [LICENSE](LICENSE).
