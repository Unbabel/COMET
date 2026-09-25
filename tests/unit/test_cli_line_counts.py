# -*- coding: utf-8 -*-
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from comet.cli import compare as compare_cli
from comet.cli import score as score_cli

SRC = ["Hello.", "Good morning.", "Thank you.", "Goodbye."]
MT_1 = ["Bonjour.", "Bonjour !", "Merci.", "Au revoir !"]
MT_2 = ["Salut.", "Bon matin.", "Merci bien.", "Adieu."]
REF = ["Bonjour.", "Bonjour.", "Merci.", "Au revoir."]


class _Output(dict):
    """Stand-in for the Prediction returned by model.predict."""

    __getattr__ = dict.__getitem__


class _FakeModel:
    """Records the samples it is asked to score instead of running a model."""

    def __init__(self, requires_references=True):
        self.references_required = requires_references
        self.samples = []

    def eval(self):
        return self

    def half(self):
        return self

    def set_embedding_cache(self):
        pass

    def requires_references(self):
        return self.references_required

    def predict(self, samples, **kwargs):
        # Copy: score_command later adds a "COMET" key to these same dicts.
        self.samples.append([dict(sample) for sample in samples])
        return _Output(scores=[0.5] * len(samples), system_score=0.5)


class _ScoringStarted(Exception):
    pass


class TestCLILineCounts(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.ckpt = self.write("model.ckpt", "")

    def write(self, name, lines, newline_at_end=True):
        path = os.path.join(self.tmp, name)
        text = lines if isinstance(lines, str) else "\n".join(lines)
        if lines and newline_at_end:
            text += "\n"
        with open(path, "w", encoding="utf-8") as fp:
            fp.write(text)
        return path

    def run_score(self, args, model):
        argv = ["comet-score"] + args + ["--model", self.ckpt]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            score_cli, "load_from_checkpoint", return_value=model
        ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
            try:
                score_cli.score_command()
            except SystemExit as e:
                return e.code, err.getvalue()
        return None, err.getvalue()

    def run_compare(self, args, model):
        argv = ["comet-compare"] + args + ["--model", self.ckpt]
        reached = []

        def fake_score(cfg, systems):
            reached.append(systems)
            raise _ScoringStarted()

        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            compare_cli, "load_from_checkpoint", return_value=model
        ), mock.patch.object(
            compare_cli, "score", side_effect=fake_score
        ), redirect_stdout(
            io.StringIO()
        ), redirect_stderr(
            io.StringIO()
        ) as err:
            try:
                compare_cli.compare_command()
            except SystemExit as e:
                return e.code, err.getvalue(), reached
            except _ScoringStarted:
                pass
        return None, err.getvalue(), reached

    def assert_line_count_error(self, code, err, *expected):
        self.assertEqual(code, 2)
        self.assertIn("same number of lines", err)
        for name, n in expected:
            self.assertIn("{}: {} lines".format(name, n), err)

    # comet-score: aligned inputs must still be scored and paired row by row.

    def test_score_aligned(self):
        src, mt, ref = (
            self.write("src", SRC),
            self.write("mt", MT_1),
            self.write("ref", REF),
        )
        model = _FakeModel()
        code, _ = self.run_score(["-s", src, "-t", mt, "-r", ref], model)
        self.assertIsNone(code)
        self.assertEqual(
            model.samples,
            [[{"src": s, "mt": m, "ref": r} for s, m, r in zip(SRC, MT_1, REF)]],
        )

    def test_score_two_aligned_systems(self):
        # Two systems of four lines: a check against len(translations), the
        # number of systems, would reject this valid input.
        src, ref = self.write("src", SRC), self.write("ref", REF)
        mt1, mt2 = self.write("mt1", MT_1), self.write("mt2", MT_2)
        expected = [
            [{"src": s, "mt": m, "ref": r} for s, m, r in zip(SRC, mt, REF)]
            for mt in (MT_1, MT_2)
        ]
        for gpus, want in (("1", expected), ("2", [expected[0] + expected[1]])):
            model = _FakeModel()
            args = ["-s", src, "-t", mt1, mt2, "-r", ref, "--gpus", gpus]
            code, _ = self.run_score(args, model)
            self.assertIsNone(code)
            self.assertEqual(model.samples, want)

    def test_score_reference_free_aligned(self):
        src, mt = self.write("src", SRC), self.write("mt", MT_1)
        model = _FakeModel(requires_references=False)
        code, _ = self.run_score(["-s", src, "-t", mt], model)
        self.assertIsNone(code)
        self.assertEqual(
            model.samples, [[{"src": s, "mt": m} for s, m in zip(SRC, MT_1)]]
        )

    def test_score_counts_lines_as_read(self):
        # No trailing newline on the source, and a U+2028 inside a translation:
        # both files hold four segments as the CLI reads them, even though
        # counting "\n" or using str.splitlines() would disagree.
        mt_lines = ["Bonjour.", "Bonjour\u2028!", "Merci.", "Au revoir !"]
        src = self.write("src", SRC, newline_at_end=False)
        mt, ref = self.write("mt", mt_lines), self.write("ref", REF)
        model = _FakeModel()
        code, _ = self.run_score(["-s", src, "-t", mt, "-r", ref], model)
        self.assertIsNone(code)
        self.assertEqual([s["mt"] for s in model.samples[0]], mt_lines)

    def test_score_missing_references_still_rejected(self):
        src, mt = self.write("src", SRC), self.write("mt", MT_1)
        model = _FakeModel(requires_references=True)
        code, err = self.run_score(["-s", src, "-t", mt], model)
        self.assertEqual(code, 2)
        self.assertIn("requires -r/--references", err)
        self.assertEqual(model.samples, [])

    # comet-score: mismatched inputs must be rejected before scoring.

    def test_score_translation_shorter_than_source(self):
        src, ref = self.write("src", SRC), self.write("ref", REF)
        mt = self.write("mt", MT_1[:3])
        model = _FakeModel()
        code, err = self.run_score(["-s", src, "-t", mt, "-r", ref], model)
        self.assert_line_count_error(code, err, ("src", 4), ("mt", 3), ("ref", 4))
        self.assertEqual(model.samples, [])

    def test_score_reference_shorter_than_source(self):
        src, mt = self.write("src", SRC), self.write("mt", MT_1)
        ref = self.write("ref", REF[:3])
        model = _FakeModel()
        code, err = self.run_score(["-s", src, "-t", mt, "-r", ref], model)
        self.assert_line_count_error(code, err, ("src", 4), ("mt", 4), ("ref", 3))
        self.assertEqual(model.samples, [])

    def test_score_short_system_with_multiple_gpus(self):
        # With --gpus > 1 all systems are flattened before pairing, so a short
        # system shifts every later system against the wrong source lines.
        src, ref = self.write("src", SRC), self.write("ref", REF)
        mt1, mt2 = self.write("mt1", MT_1[:3]), self.write("mt2", MT_2[:3])
        model = _FakeModel()
        args = ["-s", src, "-t", mt1, mt2, "-r", ref, "--gpus", "2"]
        code, err = self.run_score(args, model)
        self.assert_line_count_error(code, err, ("mt1", 3), ("mt2", 3), ("ref", 4))
        self.assertEqual(model.samples, [])

    # comet-compare

    def test_compare_aligned(self):
        src, ref = self.write("src", SRC), self.write("ref", REF)
        mt1, mt2 = self.write("mt1", MT_1), self.write("mt2", MT_2)
        code, _, reached = self.run_compare(
            ["-s", src, "-t", mt1, mt2, "-r", ref], _FakeModel()
        )
        self.assertIsNone(code)
        self.assertEqual(
            reached,
            [
                [
                    {"src": SRC, "mt": MT_1, "ref": REF},
                    {"src": SRC, "mt": MT_2, "ref": REF},
                ]
            ],
        )

    def test_compare_reference_free_aligned(self):
        src = self.write("src", SRC)
        mt1, mt2 = self.write("mt1", MT_1), self.write("mt2", MT_2)
        model = _FakeModel(requires_references=False)
        code, _, reached = self.run_compare(["-s", src, "-t", mt1, mt2], model)
        self.assertIsNone(code)
        self.assertEqual(
            reached, [[{"src": SRC, "mt": MT_1}, {"src": SRC, "mt": MT_2}]]
        )

    def test_compare_single_system_still_rejected(self):
        src, mt, ref = (
            self.write("src", SRC),
            self.write("mt", MT_1),
            self.write("ref", REF),
        )
        with self.assertRaises(AssertionError):
            self.run_compare(["-s", src, "-t", mt, "-r", ref], _FakeModel())

    def test_compare_source_shorter_than_translations(self):
        src, ref = self.write("src", SRC[:3]), self.write("ref", REF)
        mt1, mt2 = self.write("mt1", MT_1), self.write("mt2", MT_2)
        code, err, reached = self.run_compare(
            ["-s", src, "-t", mt1, mt2, "-r", ref], _FakeModel()
        )
        self.assert_line_count_error(code, err, ("src", 3), ("mt1", 4), ("ref", 4))
        self.assertEqual(reached, [])

    def test_compare_translation_shorter_than_source(self):
        src, ref = self.write("src", SRC), self.write("ref", REF)
        mt1, mt2 = self.write("mt1", MT_1), self.write("mt2", MT_2[:3])
        code, err, reached = self.run_compare(
            ["-s", src, "-t", mt1, mt2, "-r", ref], _FakeModel()
        )
        self.assert_line_count_error(code, err, ("mt1", 4), ("mt2", 3))
        self.assertEqual(reached, [])


if __name__ == "__main__":
    unittest.main()
