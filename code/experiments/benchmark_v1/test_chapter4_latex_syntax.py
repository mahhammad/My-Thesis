"""Static guards for two recurring Chapter 4 TeX failures.

No TeX engine or dataset needed. This does NOT replace a complete XeLaTeX build:
it guards raw underscores in \texttt paths and multiple \label directives in
a single amsmath equation, the causes of the November 2026 errors.
"""
from pathlib import Path
import re
import unittest

CHAPTER = Path(__file__).resolve().parents[3] / "tex" / "chapter4.tex"


def braced_argument(src, opening_brace):
    """Return text and end position of an argument with nested {...} groups."""
    depth = 0
    start = opening_brace + 1
    for i in range(opening_brace, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i], i
    raise AssertionError("Unbalanced command argument")


class Chapter4LaTexGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = CHAPTER.read_text(encoding="utf-8")

    def test_no_raw_underscore_in_typewriter_arguments(self):
        violations = []
        for m in re.finditer(r"\\texttt\s*\{", self.src):
            argument, _ = braced_argument(self.src, m.end() - 1)
            if re.search(r"(?<!\\)_", argument):
                violations.append((self.src.count("\n", 0, m.start()) + 1, argument))
        self.assertFalse(violations, f"Raw '_' inside \\texttt: {violations}")

    def test_numbered_amsmath_equation_has_at_most_one_label(self):
        duplicates = []
        pattern = re.compile(r"\\begin\{equation\}([\s\S]*?)\\end\{equation\}")
        for match in pattern.finditer(self.src):
            labels = re.findall(r"\\label\{([^}]+)\}", match.group(1))
            if len(labels) > 1:
                duplicates.append((self.src.count("\n", 0, match.start()) + 1, labels))
        self.assertFalse(duplicates, f"Multiple equation labels: {duplicates}")

    def test_chapter4_reuses_chapter3_margin_loss_without_duplicate_equation(self):
        self.assertNotIn(r"\label{eq:ch4-margin-only-objective}", self.src)
        self.assertNotIn(r"\eqref{eq:ch4-margin-only-objective}", self.src)
        self.assertIn(r"\eqref{eq:ch3-js-sigmoid-balanced-objective}", self.src)
        self.assertNotIn(r"\label{eq:ch4-sigmoid-direct-training-loss}", self.src)


if __name__ == "__main__":
    unittest.main()
