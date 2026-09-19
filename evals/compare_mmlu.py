#!/usr/bin/env python3
"""Compare baseline and compressed gemma_mmlu outputs.

Each input is the stdout captured from gemma_mmlu and may contain unrelated
lines. Only lines beginning with ``MMLU_RESULT `` are parsed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


RESULT_PREFIX = "MMLU_RESULT "


def load_results(path: Path) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.startswith(RESULT_PREFIX):
                continue
            try:
                result = json.loads(line[len(RESULT_PREFIX) :])
                question_id = int(result["id"])
                result["correct"] = bool(result["correct"])
                result["expected"] = str(result["expected"])
                result["predicted"] = str(result["predicted"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid MMLU_RESULT: {error}"
                ) from error
            if question_id in results:
                raise ValueError(f"{path}:{line_number}: duplicate id {question_id}")
            results[question_id] = result

    if not results:
        raise ValueError(f"{path}: no {RESULT_PREFIX.strip()} lines found")
    return results


def compare_results(
    baseline: dict[int, dict[str, Any]], variant: dict[int, dict[str, Any]]
) -> dict[str, int | float]:
    baseline_ids = set(baseline)
    variant_ids = set(variant)
    if baseline_ids != variant_ids:
        missing = sorted(baseline_ids - variant_ids)
        extra = sorted(variant_ids - baseline_ids)
        raise ValueError(
            "result IDs differ: "
            f"missing from variant={missing[:10]}, extra in variant={extra[:10]}"
        )

    correct_to_incorrect = 0
    incorrect_to_correct = 0
    wrong_to_wrong_changes = 0
    answer_changes = 0
    baseline_correct = 0
    variant_correct = 0

    for question_id in sorted(baseline_ids):
        base = baseline[question_id]
        changed = variant[question_id]
        if base["expected"] != changed["expected"]:
            raise ValueError(
                f"id {question_id}: expected answers differ: "
                f"{base['expected']!r} != {changed['expected']!r}"
            )

        base_correct = base["correct"]
        changed_correct = changed["correct"]
        baseline_correct += int(base_correct)
        variant_correct += int(changed_correct)
        answer_changed = base["predicted"] != changed["predicted"]
        answer_changes += int(answer_changed)

        if base_correct and not changed_correct:
            correct_to_incorrect += 1
        elif not base_correct and changed_correct:
            incorrect_to_correct += 1
        elif not base_correct and not changed_correct and answer_changed:
            wrong_to_wrong_changes += 1

    samples = len(baseline_ids)
    flips = correct_to_incorrect + incorrect_to_correct
    return {
        "samples": samples,
        "baseline_correct": baseline_correct,
        "variant_correct": variant_correct,
        "baseline_accuracy": baseline_correct / samples,
        "variant_accuracy": variant_correct / samples,
        "accuracy_delta": (variant_correct - baseline_correct) / samples,
        "correct_to_incorrect": correct_to_incorrect,
        "incorrect_to_correct": incorrect_to_correct,
        "flips": flips,
        "flips_fraction": flips / samples,
        "flips_percent": 100.0 * flips / samples,
        "wrong_to_wrong_changes": wrong_to_wrong_changes,
        "answer_changes": answer_changes,
        "answer_changes_fraction": answer_changes / samples,
        "answer_changes_percent": 100.0 * answer_changes / samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and variant gemma_mmlu output streams."
    )
    parser.add_argument("baseline", type=Path, help="baseline gemma_mmlu stdout")
    parser.add_argument("variant", type=Path, help="variant gemma_mmlu stdout")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metrics = compare_results(
            load_results(args.baseline), load_results(args.variant)
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"MMLU_FLIPS {json.dumps(metrics, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
