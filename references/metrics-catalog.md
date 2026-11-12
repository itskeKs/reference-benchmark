# Metrics catalog

Pick one **primary metric** that answers the decision question, a few
**diagnostic metrics** that explain it, and **cost metrics** (time, memory,
money). Name every proxy as a proxy in the report header.

## Text output vs. a reference text (ASR, OCR, translation, rewrite)

| Metric | Definition | Notes |
|---|---|---|
| WER / CER | Levenshtein alignment over normalized tokens/characters: (S + D + I) / N_ref | Report S/D/I separately — they point at different causes. Normalize first (case, punctuation, number words) and document the rules. |
| Per-segment WER | WER per reference segment (turn, paragraph, page) | Feeds the diff; shows *where*. |
| Attributed WER | Alignment where a token only counts as correct if its attribute (speaker, field, language) also matches | Separates "wrong word" from "right word, wrong owner". |
| Attribute accuracy on matched tokens | Share of equal-aligned tokens whose attribute matches | Cheap proxy when the reference has no exact boundaries. |
| Segment-start accuracy | Attribute of the first hypothesis token at each reference segment start | Catches boundary/role swaps. |
| Swap regions | Maximal runs of consecutive attribute mismatches | Count + list; one swap region = one human-visible error. |

Hypothesis labels rarely share names with reference labels (`spk0` vs
`interviewer`): map them first, e.g. majority vote over aligned equal tokens, and
report the mapping.

## Generated text vs. a fact checklist (LLM summaries, notes, reports)

| Metric | Definition | Notes |
|---|---|---|
| Fact recall | (full + 0.5 × partial) / N_facts against a hand-written checklist per input | The checklist is the reference; write it before looking at outputs. |
| Invented content | Count of claims with no source (fabricated quotes, dates, numbers, blanket negations like "no X reported") | Usually a hard gate, not an average. |
| Guardrail violations | Regex/keyword screen for forbidden content classes | Screening hits go to a human; they are not verdicts. |
| Structure compliance | Required sections present, order, length limits | Deterministic. |
| Judge score | LLM-as-judge per rubric item | Only after calibration against human scores (see SKILL.md). |

## Structured output vs. expected structure (extraction, classification, tool calls)

| Metric | Definition |
|---|---|
| Field-level precision / recall / F1 | Per field and micro/macro averaged; normalize values (dates, whitespace, case) first |
| Exact-match rate | Whole record identical after normalization |
| Confusion matrix | For categorical fields; shows which classes collapse |
| Schema-valid rate | Output parses and validates |

## Ranking / retrieval

Recall@k, MRR, nDCG@k against labeled relevant items; report per-query
results so the diff can show which queries regress.

## Code / agent tasks

Pass rate on a fixed task set with hidden tests, plus per-task
pass/fail diff between runs (which tasks flipped).

## Cost and environment (always collect)

| Metric | Notes |
|---|---|
| Wall time, per-stage timings | Instrument stages before optimizing — the bottleneck is often not where you think. |
| Real-time factor / throughput | wall / input duration, items/s, tokens/s |
| Peak memory | |
| Thermal, battery, load before/after | Mobile and laptops throttle; record state so slow runs can be explained. |
| Token counts / API cost | For hosted models |

## Determinism check

Run the baseline twice. If quality metrics are byte-identical, single runs are
enough for quality and only timing needs repetitions. If they vary, report
mean ± spread over N runs and never compare single runs.
