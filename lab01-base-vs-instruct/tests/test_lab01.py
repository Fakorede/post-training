import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from checker import extract_answer, is_correct  # noqa: E402
from score import score  # noqa: E402


class CheckerTest(unittest.TestCase):
    def test_extract(self):
        self.assertEqual(extract_answer("12"), "12")
        self.assertEqual(extract_answer("7 + 5 = 12."), "12")
        self.assertEqual(extract_answer("The answer is 54, from 9 and 6"), "54")
        self.assertEqual(extract_answer("It is 1,200"), "1200")
        self.assertIsNone(extract_answer("twelve"))

    def test_correct(self):
        self.assertTrue(is_correct("Answer: 63", "63"))
        self.assertTrue(is_correct("63.0", "63"))
        self.assertFalse(is_correct("73", "63"))
        self.assertFalse(is_correct("", "63"))


class ScoreTest(unittest.TestCase):
    prompts = {"exact": [{"id": "e1", "answer": "12"}]}

    def test_disagreement(self):
        row = score(self.prompts, {"e1": ["13", "12"]}, {"e1": [1, 2]})[0]
        self.assertEqual((row["correct"], row["verdict"]), ([2], "DISAGREE"))

    def test_incomplete_ranking(self):
        with self.assertRaises(ValueError):
            score(self.prompts, {"e1": ["13", "12"]}, {"e1": []})


if __name__ == "__main__":
    unittest.main()
