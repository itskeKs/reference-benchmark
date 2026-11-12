#!/usr/bin/env python3
"""Token alignment, WER and inline diff markup (stdlib only, Python >= 3.9).

Importable (`import textdiff`) or CLI:

    textdiff.py REF_FILE HYP_FILE [--segments-by-line] [--json-out F] [--markdown-out F]

Pipeline: normalize() -> tokens, align() -> ops, counts()/render_inline()/top_errors().
Ops are tuples (kind, ref_index | None, hyp_index | None) with kind in
"equal" | "sub" | "del" | "ins".

align() is exact Levenshtein (unit costs) with a deterministic backtrace from the end
that prefers diagonal (equal/sub), then deletion, then insertion. It uses the bit-parallel
Myers/Hyyro algorithm on Python big ints: one (Pv, Mv) vertical-delta pair per hyp token
(about 2 * N_ref / 8 bytes each) instead of an N x M cell matrix; DP values needed by the
backtrace are recomputed from those bit vectors with popcounts. 10k x 10k tokens take well
under a second.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
import unicodedata

_TOKEN_RE = re.compile(r"[^\W_]+")  # letters/digits/umlauts, no underscore, no punctuation
try:
    _popcount = int.bit_count  # Python >= 3.10
except AttributeError:  # pragma: no cover - Python 3.9
    def _popcount(x):
        return bin(x).count("1")


def normalize(text, *, casefold=True, strip_punct=True):
    """NFKC, optional casefold, then tokens (`[^\\W_]+` or plain whitespace split)."""
    text = unicodedata.normalize("NFKC", text)
    if casefold:
        text = text.casefold()
    return _TOKEN_RE.findall(text) if strip_punct else text.split()


def align(ref, hyp):
    """Exact Levenshtein alignment of two token lists -> list of (kind, ref_i, hyp_j)."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return [("ins", None, j) for j in range(m)]
    if m == 0:
        return [("del", i, None) for i in range(n)]
    peq = {}  # token -> bitmask of ref positions holding it
    for i, tok in enumerate(ref):
        peq[tok] = peq.get(tok, 0) | (1 << i)
    full = (1 << n) - 1
    pv, mv = full, 0  # column 0: D(i, 0) = i, every vertical delta is +1
    cols = [(pv, mv)]  # cols[j] = vertical deltas of column j (bit i-1 = row i)
    for tok in hyp:
        eq = peq.get(tok, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | (~(xh | pv) & full)
        mh = pv & xh
        ph = ((ph << 1) | 1) & full  # row 0 boundary: D(0, j) = j
        mh = (mh << 1) & full
        pv = mh | (~(xv | ph) & full)
        mv = ph & xv
        cols.append((pv, mv))

    def dist(i, j):  # D(i, j) = j + sum of vertical deltas of column j over rows 1..i
        plus, minus = cols[j]
        low = (1 << i) - 1
        return j + _popcount(plus & low) - _popcount(minus & low)

    ops = []
    i, j, cur = n, m, dist(n, m)
    while i or j:
        if i and j:  # tie rule: diagonal first ...
            same = ref[i - 1] == hyp[j - 1]
            diag = dist(i - 1, j - 1)
            if diag + (not same) == cur:
                ops.append(("equal" if same else "sub", i - 1, j - 1))
                i, j, cur = i - 1, j - 1, diag
                continue
        if i:  # ... then deletion ...
            up = dist(i - 1, j)
            if up + 1 == cur:
                ops.append(("del", i - 1, None))
                i, cur = i - 1, up
                continue
        ops.append(("ins", None, j - 1))  # ... then insertion
        j, cur = j - 1, cur - 1
    ops.reverse()
    return ops


def _rate(errors, n_ref, n_hyp):
    if n_ref == 0:
        return 0.0 if n_hyp == 0 else 1.0
    return errors / n_ref


def counts(ops):
    """-> {S, D, I, N_ref, N_hyp, wer}."""
    kinds = Counter(op[0] for op in ops)
    s, d, i, eq = kinds["sub"], kinds["del"], kinds["ins"], kinds["equal"]
    n_ref, n_hyp = eq + s + d, eq + s + i
    return {"S": s, "D": d, "I": i, "N_ref": n_ref, "N_hyp": n_hyp, "wer": _rate(s + d + i, n_ref, n_hyp)}


def render_inline(ops, ref, hyp):
    """equal plain, `[ref→hyp]`, `[-w-]` (deleted), `[+w+]` (inserted)."""
    out = []
    for kind, i, j in ops:
        if kind == "equal":
            out.append(ref[i])
        elif kind == "sub":
            out.append("[%s→%s]" % (ref[i], hyp[j]))
        elif kind == "del":
            out.append("[-%s-]" % ref[i])
        else:
            out.append("[+%s+]" % hyp[j])
    return " ".join(out)


def _prepare(ref_segments, hyp):
    ref, owner = [], []
    for k, seg in enumerate(ref_segments):
        toks = normalize(seg["text"])
        ref.extend(toks)
        owner.extend([k] * len(toks))
    return ref, owner, (normalize(hyp) if isinstance(hyp, str) else list(hyp))


def _per_segment(ref_segments, ops, ref, hyp, owner):
    buckets = [[] for _ in ref_segments]
    last = 0  # insertions belong to the segment of the last seen ref token (first segment at the start)
    for op in ops:
        if op[1] is not None:
            last = owner[op[1]]
        buckets[last].append(op)
    report = []
    for seg, seg_ops in zip(ref_segments, buckets):
        c = counts(seg_ops)
        report.append({"id": seg["id"], "label": seg.get("label"), "S": c["S"], "D": c["D"], "I": c["I"],
                       "N_ref": c["N_ref"], "wer": c["wer"], "inline": render_inline(seg_ops, ref, hyp)})
    return report


def segment_report(ref_segments, hyp_text_or_tokens):
    """Align the concatenated reference once, attribute ops to ref segments.

    ref_segments: [{"id", "text", "label"?}]. -> [{id, label, S, D, I, N_ref, wer, inline}].
    """
    if not ref_segments:
        return []
    ref, owner, hyp = _prepare(ref_segments, hyp_text_or_tokens)
    return _per_segment(ref_segments, align(ref, hyp), ref, hyp, owner)


def _expand(pairs):
    """(label, word[, segment]) pairs -> parallel token / label / segment lists."""
    toks, labels, segs = [], [], []
    for pair in pairs:
        for tok in normalize(pair[1]):
            toks.append(tok)
            labels.append(pair[0])
            segs.append(pair[2] if len(pair) > 2 else None)
    return toks, labels, segs


def majority_label_map(ref_pairs, hyp_pairs):
    """Map each hyp label to the ref label it most often aligns with (equal ops only)."""
    rt, rl, _ = _expand(ref_pairs)
    ht, hl, _ = _expand(hyp_pairs)
    votes = defaultdict(Counter)
    for kind, i, j in align(rt, ht):
        if kind == "equal":
            votes[hl[j]][rl[i]] += 1
    return {h: min(c.items(), key=lambda kv: (-kv[1], str(kv[0])))[0] for h, c in votes.items()}


def attributed(ref_pairs, hyp_pairs, label_map=None):
    """Attributed WER, label accuracy on matched tokens, contiguous mismatch regions.

    Pairs are (label, word) or (ref only) (label, word, segment_id). Words are normalized
    (a word may yield several tokens). label_map maps hyp labels onto ref labels (see
    majority_label_map); unmapped labels compare as-is. An equal-aligned token whose label
    differs counts as a substitution in attributed_wer. A region is a maximal run of such
    tokens with the same (ref label, hyp label, segment); sub/del/ins ops inside it do not
    break it, a correctly labelled token or a label/segment change does.
    """
    rt, rl, rs = _expand(ref_pairs)
    ht, hl, _ = _expand(hyp_pairs)
    ops = align(rt, ht)
    base = counts(ops)
    lm = label_map or {}
    good = bad = 0
    regions, cur = [], None
    for kind, i, j in ops:
        if kind == "ins":
            continue
        if cur is not None and (rl[i], rs[i]) != cur["_owner"]:
            cur = None  # a ref token of another label/segment sits in between
        if kind != "equal":
            continue
        hyp_label = lm.get(hl[j], hl[j])
        if hyp_label == rl[i]:
            good += 1
            cur = None
            continue
        bad += 1
        if cur is None or cur["hyp_label"] != hyp_label:
            cur = {"_owner": (rl[i], rs[i]), "ref_label": rl[i], "hyp_label": hyp_label,
                   "ref_start": i, "ref_end": i, "tokens": []}
            regions.append(cur)
        cur["ref_end"] = i
        cur["tokens"].append(rt[i])
    for r in regions:
        r["n_tokens"] = len(r["tokens"])
        r["text"] = " ".join(r.pop("tokens"))
        del r["_owner"]
    errors = base["S"] + base["D"] + base["I"] + bad
    return {**base, "label_errors": bad, "label_accuracy": good / (good + bad) if good + bad else None,
            "attributed_wer": _rate(errors, base["N_ref"], base["N_hyp"]), "regions": regions,
            "label_map": dict(label_map) if label_map else None}


def top_errors(ops, ref, hyp, n=20):
    """Most frequent substitutions / deletions / insertions (count desc, then alphabetical)."""
    subs, dels, inss = Counter(), Counter(), Counter()
    for kind, i, j in ops:
        if kind == "sub":
            subs[(ref[i], hyp[j])] += 1
        elif kind == "del":
            dels[ref[i]] += 1
        elif kind == "ins":
            inss[hyp[j]] += 1

    def top(counter, make):
        return [make(k, c) for k, c in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]
    return {"substitutions": top(subs, lambda k, c: {"ref": k[0], "hyp": k[1], "count": c}),
            "deletions": top(dels, lambda k, c: {"word": k, "count": c}),
            "insertions": top(inss, lambda k, c: {"word": k, "count": c})}


def build_report(ref_segments, hyp_text_or_tokens, top_n=20):
    """One alignment -> {"summary": counts, "segments": per-segment, "top_errors": ...}."""
    ref, owner, hyp = _prepare(ref_segments, hyp_text_or_tokens)
    ops = align(ref, hyp)
    segments = _per_segment(ref_segments, ops, ref, hyp, owner) if ref_segments else []
    return {"summary": counts(ops), "segments": segments, "top_errors": top_errors(ops, ref, hyp, top_n)}


def _cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown(report, title="Text diff"):
    s = report["summary"]
    lines = ["# " + title, "", "| metric | value |", "|---|---|",
             "| WER | %.2f %% |" % (100 * s["wer"]), "| Substitutions (S) | %d |" % s["S"],
             "| Deletions (D) | %d |" % s["D"], "| Insertions (I) | %d |" % s["I"],
             "| Reference tokens | %d |" % s["N_ref"], "| Hypothesis tokens | %d |" % s["N_hyp"], ""]
    bad = [g for g in report["segments"] if g["S"] + g["D"] + g["I"]]
    lines += ["## Segments with errors (%d of %d)" % (len(bad), len(report["segments"])), "",
              "Markup: `[ref→hyp]` substitution, `[-w-]` missing in hypothesis, `[+w+]` extra in hypothesis.", ""]
    for g in bad:
        name = "%s (%s)" % (g["id"], g["label"]) if g["label"] else str(g["id"])
        lines += ["### %s: WER %.1f %% (S %d, D %d, I %d)" % (name, 100 * g["wer"], g["S"], g["D"], g["I"]),
                  "", g["inline"], ""]
    top = report["top_errors"]
    lines += ["## Top errors", ""]
    for title_, key, head in (("Substitutions", "substitutions", "ref → hyp"),
                              ("Deletions", "deletions", "word"), ("Insertions", "insertions", "word")):
        lines += ["### " + title_, "", "| count | %s |" % head, "|---|---|"]
        for e in top[key]:
            what = "%s → %s" % (e["ref"], e["hyp"]) if key == "substitutions" else e["word"]
            lines.append("| %d | %s |" % (e["count"], _cell(what)))
        lines.append("")
    return "\n".join(lines)


def write_atomic(path, text):
    """tmp file + rename, mode 0600."""
    path = Path(path)
    tmp = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(str(tmp), str(path))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Align a hypothesis text against a reference and report WER.")
    ap.add_argument("ref_file")
    ap.add_argument("hyp_file")
    ap.add_argument("--segments-by-line", action="store_true", help="each non-empty reference line is a segment")
    ap.add_argument("--json-out")
    ap.add_argument("--markdown-out")
    args = ap.parse_args(argv)
    try:
        ref_text = Path(args.ref_file).read_text(encoding="utf-8")
        hyp_text = Path(args.hyp_file).read_text(encoding="utf-8")
    except OSError as exc:
        print("textdiff: %s" % exc, file=sys.stderr)
        return 2
    if args.segments_by_line:
        segs = [{"id": str(n), "text": line} for n, line in enumerate(ref_text.splitlines(), 1) if line.strip()]
    else:
        segs = [{"id": "all", "text": ref_text}]
    report = build_report(segs, hyp_text)
    s = report["summary"]
    print("WER %.2f %%  (S %d, D %d, I %d, N_ref %d, N_hyp %d)" % (100 * s["wer"], s["S"], s["D"], s["I"],
                                                                 s["N_ref"], s["N_hyp"]))
    if args.json_out:
        write_atomic(args.json_out, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if args.markdown_out:
        write_atomic(args.markdown_out, render_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
