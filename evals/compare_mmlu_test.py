#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from compare_mmlu import compare_results, load_results


def result(question_id: int, expected: str, predicted: str) -> dict[str, object]:
    return {
        "id": question_id,
        "expected": expected,
        "predicted": predicted,
        "correct": expected == predicted,
    }


class CompareMmluTest(unittest.TestCase):
    def test_flip_counts(self) -> None:
        baseline = {
            1: result(1, "A", "A"),
            2: result(2, "A", "B"),
            3: result(3, "A", "C"),
            4: result(4, "D", "D"),
        }
        variant = {
            1: result(1, "A", "B"),
            2: result(2, "A", "A"),
            3: result(3, "A", "D"),
            4: result(4, "D", "D"),
        }

        metrics = compare_results(baseline, variant)

        self.assertEqual(metrics["correct_to_incorrect"], 1)
        self.assertEqual(metrics["incorrect_to_correct"], 1)
        self.assertEqual(metrics["flips"], 2)
        self.assertEqual(metrics["flips_percent"], 50.0)
        self.assertEqual(metrics["wrong_to_wrong_changes"], 1)
        self.assertEqual(metrics["answer_changes"], 3)
        self.assertEqual(metrics["answer_changes_percent"], 75.0)
        self.assertEqual(metrics["accuracy_delta"], 0.0)

    def test_loads_prefixed_results_and_ignores_other_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            rows = [result(7, "B", "B"), result(8, "C", "A")]
            path.write_text(
                "startup noise\n"
                + "\n".join(f"MMLU_RESULT {json.dumps(row)}" for row in rows)
                + "\nMMLU_SUMMARY {}\n",
                encoding="utf-8",
            )

            loaded = load_results(path)

        self.assertEqual(set(loaded), {7, 8})
        self.assertTrue(loaded[7]["correct"])
        self.assertFalse(loaded[8]["correct"])

    def test_requires_matching_question_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "result IDs differ"):
            compare_results(
                {1: result(1, "A", "A")}, {2: result(2, "A", "A")}
            )

    def test_requires_matching_expected_answers(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected answers differ"):
            compare_results(
                {1: result(1, "A", "A")}, {1: result(1, "B", "B")}
            )


if __name__ == "__main__":
    unittest.main()
