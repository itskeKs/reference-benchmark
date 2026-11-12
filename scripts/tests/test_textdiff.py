"""Tests for textdiff.py (hand-made cases with known answers + a naive-DP cross-check)."""
import contextlib
import io
import json
import os
import random
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import textdiff as td  # noqa: E402


def naive_align(ref, hyp):
    """Full-matrix Levenshtein with the documented tie rule (diagonal, deletion, insertion)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1)
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]) == d[i][j]:
            ops.append(("equal" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and d[i - 1][j] + 1 == d[i][j]:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    return ops[::-1]


def words(s):
    return s.split()


class NormalizeTests(unittest.TestCase):
    def test_default_rules(self):
        self.assertEqual(td.normalize("Hello, World! It's 42."), ["hello", "world", "it", "s", "42"])

    def test_underscore_and_whitespace(self):
        self.assertEqual(td.normalize("foo_bar\tbaz\n"), ["foo", "bar", "baz"])

    def test_umlauts_and_nfkc(self):
        self.assertEqual(td.normalize("Über Straße"), ["über", "strasse"])
        self.assertEqual(td.normalize("Über"), td.normalize("Über"))  # decomposed == composed
        self.assertEqual(td.normalize("ﬁsh"), ["fish"])  # NFKC ligature
        self.assertEqual(td.normalize("Größe 42"), ["grösse", "42"])  # casefold: ß -> ss, digits kept

    def test_flags(self):
        self.assertEqual(td.normalize("A, b", casefold=False), ["A", "b"])
        self.assertEqual(td.normalize("A, b", strip_punct=False), ["a,", "b"])


class AlignTests(unittest.TestCase):
    def test_perfect_match(self):
        ref = words("a b c")
        ops = td.align(ref, ref)
        self.assertEqual(ops, [("equal", 0, 0), ("equal", 1, 1), ("equal", 2, 2)])
        c = td.counts(ops)
        self.assertEqual((c["S"], c["D"], c["I"], c["N_ref"], c["N_hyp"], c["wer"]), (0, 0, 0, 3, 3, 0.0))

    def test_hand_case_known_counts(self):
        ref, hyp = words("the very cat sat on the mat"), words("the cat sit on the big mat")
        ops = td.align(ref, hyp)
        c = td.counts(ops)
        self.assertEqual((c["S"], c["D"], c["I"]), (1, 1, 1))
        self.assertEqual((c["N_ref"], c["N_hyp"]), (7, 7))
        self.assertAlmostEqual(c["wer"], 3 / 7)
        self.assertEqual(td.render_inline(ops, ref, hyp), "the [-very-] cat [sat→sit] on the [+big+] mat")

    def test_tie_rule_diagonal_first(self):
        # two optimal alignments of cost 2: (sub, sub) or (del, eq, ins); the diagonal wins
        self.assertEqual(td.align(["a", "b"], ["b", "a"]), [("sub", 0, 0), ("sub", 1, 1)])
        # sub b->c is preferred over del b + ins c; a is then deleted
        self.assertEqual(td.align(["a", "b"], ["c"]), [("del", 0, None), ("sub", 1, 0)])

    def test_tie_rule_deletion_before_insertion(self):
        # x y x vs y x y: both (ins y, eq, eq, del x) and (del x, eq, eq, ins y) cost 2;
        # the backtrace runs from the end and prefers deletion over insertion
        ops = td.align(words("x y x"), words("y x y"))
        self.assertEqual(ops, [("ins", None, 0), ("equal", 0, 1), ("equal", 1, 2), ("del", 2, None)])

    def test_empty_sides(self):
        self.assertEqual(td.align([], []), [])
        self.assertEqual(td.align([], ["a", "b"]), [("ins", None, 0), ("ins", None, 1)])
        self.assertEqual(td.align(["a"], []), [("del", 0, None)])
        self.assertEqual(td.counts([])["wer"], 0.0)
        self.assertEqual(td.counts(td.align([], ["a"]))["wer"], 1.0)  # N_ref == 0 with a hypothesis
        c = td.counts(td.align(["a", "b"], []))
        self.assertEqual((c["D"], c["wer"]), (2, 1.0))
        self.assertEqual(td.render_inline(td.align(["a"], []), ["a"], []), "[-a-]")

    def test_wer_can_exceed_one(self):
        c = td.counts(td.align(["a"], ["x", "y", "z"]))
        self.assertEqual((c["S"], c["I"]), (1, 2))
        self.assertAlmostEqual(c["wer"], 3.0)

    def test_matches_naive_dp_on_random_inputs(self):
        rng = random.Random(7)
        for _ in range(1500):
            k = rng.choice([2, 3, 5])
            ref = [str(rng.randrange(k)) for _ in range(rng.randrange(0, 14))]
            hyp = [str(rng.randrange(k)) for _ in range(rng.randrange(0, 14))]
            self.assertEqual(td.align(ref, hyp), naive_align(ref, hyp), (ref, hyp))
        for _ in range(20):
            ref = [str(rng.randrange(4)) for _ in range(rng.randrange(80, 160))]
            hyp = [str(rng.randrange(4)) for _ in range(rng.randrange(80, 160))]
            self.assertEqual(td.align(ref, hyp), naive_align(ref, hyp))

    def test_performance_3k_by_3k(self):
        rng = random.Random(1)
        vocab = ["w%d" % i for i in range(300)]
        ref = [rng.choice(vocab) for _ in range(3000)]
        similar = []  # ~15 % substitutions / deletions / insertions
        for w in ref:
            r = rng.random()
            if r < 0.05:
                similar.append(rng.choice(vocab))
            elif r < 0.10:
                continue
            elif r < 0.15:
                similar += [w, rng.choice(vocab)]
            else:
                similar.append(w)
        unrelated = [rng.choice(vocab) for _ in range(3000)]
        wers = []
        for hyp in (similar, unrelated):
            t0 = time.monotonic()
            ops = td.align(ref, hyp)
            self.assertLess(time.monotonic() - t0, 5.0)
            c = td.counts(ops)
            self.assertEqual((c["N_ref"], c["N_hyp"]), (len(ref), len(hyp)))
            wers.append(c["wer"])
        self.assertLess(wers[0], 0.3)
        self.assertGreater(wers[1], 0.8)


class SegmentTests(unittest.TestCase):
    SEGS = [{"id": "1", "text": "hello world", "label": "T"}, {"id": "2", "text": "how are you"}]

    def test_attribution_of_errors(self):
        rep = td.segment_report(self.SEGS, "hello big world how you")
        self.assertEqual([r["id"] for r in rep], ["1", "2"])
        self.assertEqual((rep[0]["S"], rep[0]["D"], rep[0]["I"]), (0, 0, 1))  # "big" follows ref token "hello"
        self.assertEqual((rep[1]["S"], rep[1]["D"], rep[1]["I"]), (0, 1, 0))  # "are" missing
        self.assertEqual(rep[0]["label"], "T")
        self.assertIsNone(rep[1]["label"])
        self.assertEqual(rep[0]["inline"], "hello [+big+] world")
        self.assertEqual(rep[1]["inline"], "how [-are-] you")
        self.assertAlmostEqual(rep[0]["wer"], 0.5)
        self.assertAlmostEqual(rep[1]["wer"], 1 / 3)

    def test_insertion_goes_to_last_seen_ref_token_or_first_segment(self):
        rep = td.segment_report(self.SEGS, "uh hello world um how are you")
        self.assertEqual(rep[0]["I"], 2)  # "uh" (before any ref token -> first segment) + "um" (after "world")
        self.assertEqual(rep[1]["I"], 0)
        rep = td.segment_report(self.SEGS, "hello world how are you again")
        self.assertEqual((rep[0]["I"], rep[1]["I"]), (0, 1))  # trailing insertion -> last ref token's segment

    def test_token_list_hypothesis_and_perfect_match(self):
        rep = td.segment_report(self.SEGS, ["hello", "world", "how", "are", "you"])
        self.assertTrue(all(r["wer"] == 0.0 and r["S"] + r["D"] + r["I"] == 0 for r in rep))

    def test_segment_with_no_tokens_and_empty_input(self):
        rep = td.segment_report([{"id": "a", "text": "..."}, {"id": "b", "text": "x y"}], "x y")
        self.assertEqual((rep[0]["N_ref"], rep[0]["wer"]), (0, 0.0))
        self.assertEqual(td.segment_report([], "anything"), [])

    def test_build_report_totals_match_segments(self):
        rep = td.build_report(self.SEGS, "hello big world how you")
        self.assertEqual(sum(g["S"] + g["D"] + g["I"] for g in rep["segments"]), 2)
        self.assertEqual((rep["summary"]["D"], rep["summary"]["I"], rep["summary"]["N_ref"]), (1, 1, 5))
        md = td.render_markdown(rep)
        self.assertIn("hello [+big+] world", md)
        self.assertIn("## Top errors", md)


class AttributedTests(unittest.TestCase):
    def test_label_map_makes_everything_match(self):
        ref = [("T", "how are you today"), ("C", "fine thanks"), ("T", "good")]
        hyp = [("a", "how are you today"), ("b", "fine thanks"), ("a", "good")]
        lm = td.majority_label_map(ref, hyp)
        self.assertEqual(lm, {"a": "T", "b": "C"})
        res = td.attributed(ref, hyp, lm)
        self.assertEqual((res["wer"], res["attributed_wer"], res["label_accuracy"], res["regions"]), (0.0, 0.0, 1.0, []))
        self.assertEqual(res["label_map"], lm)
        raw = td.attributed(ref, hyp)  # identity labels: nothing matches
        self.assertEqual(raw["label_errors"], 7)
        self.assertEqual(raw["attributed_wer"], 1.0)

    def test_swap_region_and_accuracy(self):
        ref = [("T", "how are you today"), ("C", "fine thanks"), ("T", "good")]
        hyp = [("a", "how are you today"), ("a", "fine thanks"), ("a", "good")]
        res = td.attributed(ref, hyp, {"a": "T"})
        self.assertEqual(res["label_errors"], 2)
        self.assertAlmostEqual(res["label_accuracy"], 5 / 7)
        self.assertAlmostEqual(res["attributed_wer"], 2 / 7)
        self.assertEqual(res["wer"], 0.0)
        self.assertEqual(len(res["regions"]), 1)
        reg = res["regions"][0]
        self.assertEqual((reg["ref_label"], reg["hyp_label"], reg["n_tokens"], reg["text"]), ("C", "T", 2, "fine thanks"))
        self.assertEqual((reg["ref_start"], reg["ref_end"]), (4, 5))

    def test_regions_split_by_correct_token_and_by_label_change(self):
        ref = [("T", "a b c d e")]
        hyp = [("T", "a"), ("C", "b"), ("T", "c"), ("C", "d e")]
        res = td.attributed(ref, hyp)
        self.assertEqual([r["text"] for r in res["regions"]], ["b", "d e"])  # "c" is correct and breaks the run
        res = td.attributed([("T", "a b"), ("C", "c d")], [("X", "a b c d")])
        self.assertEqual([(r["ref_label"], r["text"]) for r in res["regions"]], [("T", "a b"), ("C", "c d")])

    def test_substitution_inside_a_region_does_not_break_it(self):
        res = td.attributed([("T", "a b c d e")], [("C", "a b x d e")])
        self.assertEqual(res["S"], 1)
        self.assertEqual(res["label_errors"], 4)
        self.assertEqual(len(res["regions"]), 1)
        self.assertEqual(res["regions"][0]["text"], "a b d e")
        self.assertAlmostEqual(res["attributed_wer"], 5 / 5)

    def test_word_errors_and_label_errors_combine(self):
        ref = [("T", "one two three four")]
        hyp = [("T", "one three four five")]  # del "two", ins "five" (cost 2 < 3 substitutions)
        res = td.attributed(ref, hyp)
        self.assertEqual((res["S"], res["D"], res["I"], res["label_errors"]), (0, 1, 1, 0))
        self.assertAlmostEqual(res["attributed_wer"], 2 / 4)
        self.assertEqual(res["label_accuracy"], 1.0)

    def test_majority_map_picks_most_frequent_partner(self):
        ref = [("T", "alpha beta gamma delta"), ("C", "epsilon zeta")]
        hyp = [("s0", "alpha beta gamma delta epsilon"), ("s1", "zeta")]
        self.assertEqual(td.majority_label_map(ref, hyp), {"s0": "T", "s1": "C"})  # s0: 4 votes T vs 1 C
        self.assertEqual(td.majority_label_map([], []), {})

    def test_empty_inputs(self):
        res = td.attributed([], [])
        self.assertEqual((res["wer"], res["attributed_wer"], res["label_accuracy"]), (0.0, 0.0, None))


class TopErrorsTests(unittest.TestCase):
    def test_ordering_and_limit(self):
        ref, hyp = words("a a a c b"), words("q q q y z")
        ops = [("sub", i, i) for i in range(5)]  # (a->q) x3, (c->y), (b->z)
        top = td.top_errors(ops, ref, hyp)
        self.assertEqual(top["substitutions"], [{"ref": "a", "hyp": "q", "count": 3},
                                                {"ref": "b", "hyp": "z", "count": 1},
                                                {"ref": "c", "hyp": "y", "count": 1}])
        self.assertEqual(len(td.top_errors(ops, ref, hyp, n=2)["substitutions"]), 2)
        self.assertEqual((top["deletions"], top["insertions"]), ([], []))
        ref2, hyp2 = words("a b"), words("a b z y x")
        ins = td.top_errors(td.align(ref2, hyp2), ref2, hyp2, n=2)["insertions"]
        self.assertEqual([e["word"] for e in ins], ["x", "y"])  # limited to n, alphabetical on ties

    def test_count_desc_then_alphabetical(self):
        ref = words("x x y y z")
        ops = td.align(ref, [])
        self.assertEqual(td.top_errors(ops, ref, [])["deletions"],
                         [{"word": "x", "count": 2}, {"word": "y", "count": 2}, {"word": "z", "count": 1}])


class CliTests(unittest.TestCase):
    def test_cli_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "ref.txt").write_text("hello world\nhow are you\n", encoding="utf-8")
            (tmp / "hyp.txt").write_text("hello big world how you", encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = td.main([str(tmp / "ref.txt"), str(tmp / "hyp.txt"), "--segments-by-line",
                              "--json-out", str(tmp / "o.json"), "--markdown-out", str(tmp / "o.md")])
            self.assertEqual(rc, 0)
            self.assertIn("WER 40.00 %", out.getvalue())
            data = json.loads((tmp / "o.json").read_text(encoding="utf-8"))
            self.assertEqual([s["id"] for s in data["segments"]], ["1", "2"])
            md = (tmp / "o.md").read_text(encoding="utf-8")
            self.assertIn("how [-are-] you", md)
            for name in ("o.json", "o.md"):
                self.assertEqual(stat.S_IMODE(os.stat(tmp / name).st_mode), 0o600)
            self.assertEqual([p.name for p in tmp.iterdir() if p.name.endswith(".tmp")], [])
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(td.main([str(tmp / "missing.txt"), str(tmp / "hyp.txt")]), 2)


if __name__ == "__main__":
    unittest.main()
