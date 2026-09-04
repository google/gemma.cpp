#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from compare_models import (
    parse_entropy,
    parse_model_spec,
    parse_prefixed_json,
    render_table,
)


class CompareModelsTest(unittest.TestCase):
    def test_parse_model_spec_resolves_paths_and_environment(self) -> None:
        spec = parse_model_spec(
            {
                "name": "w8a8",
                "weights": "models/model.sbs",
                "args": ["--num_threads", 4],
                "env": {"GEMMA_MM_I8": 1},
            },
            Path("/work"),
        )

        self.assertEqual(spec.name, "w8a8")
        self.assertEqual(spec.weights, Path("/work/models/model.sbs"))
        self.assertEqual(spec.args, ("--num_threads", "4"))
        self.assertEqual(spec.env, {"GEMMA_MM_I8": "1"})

    def test_parse_prefixed_json_uses_last_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.log"
            path.write_text(
                'MMLU_SUMMARY {"answers":1}\n'
                'noise\nMMLU_SUMMARY {"answers":2}\n',
                encoding="utf-8",
            )
            parsed = parse_prefixed_json(path, "MMLU_SUMMARY ")

        self.assertEqual(parsed, {"answers": 2})

    def test_parse_entropy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "entropy.log"
            path.write_text(
                "Number of input tokens: 8\n"
                "Took 1.0 s [8.0 tokens / sec]\n"
                "Total cross entropy: 12.0 [cumulative: 12.0]\n",
                encoding="utf-8",
            )
            parsed = parse_entropy(path)

        self.assertEqual(parsed["tokens"], 8)
        self.assertEqual(parsed["total_bits"], 12.0)
        self.assertEqual(parsed["bits_per_token"], 1.5)
        self.assertEqual(parsed["tokens_per_second"], 8.0)

    def test_render_table(self) -> None:
        root = {
            "name": "root",
            "mmlu": {"accuracy": 0.5},
            "entropy": {
                "total_bits": 20.0,
                "bits_per_token": 2.0,
                "tokens_per_second": 10.0,
            },
            "flips": None,
            "kl": None,
            "peak_rss_kib": 1024,
        }
        target = {
            "name": "target",
            "mmlu": {"accuracy": 0.75},
            "entropy": {
                "total_bits": 22.0,
                "bits_per_token": 2.2,
                "tokens_per_second": 12.0,
            },
            "flips": {"flips_percent": 25.0},
            "kl": {"mean": 0.01, "p95": 0.03},
            "peak_rss_kib": 2048,
        }

        table = render_table([root, target])

        self.assertIn("| root | 2.0000 | +0.000% | 10.00 | +0.0%", table)
        self.assertIn(
            "| target | 2.2000 | +10.000% | 12.00 | +20.0% | "
            "75.0% | 25.00% | 0.01 | 0.03 | 2.0 MiB |",
            table,
        )

    def test_rejects_unsafe_report_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid model name"):
            parse_model_spec(
                {"name": "../target", "weights": "model.sbs"}, Path("/work")
            )

    def test_rejects_string_model_args(self) -> None:
        with self.assertRaisesRegex(ValueError, "args must be an array"):
            parse_model_spec(
                {"name": "target", "weights": "model.sbs", "args": "--foo"},
                Path("/work"),
            )


if __name__ == "__main__":
    unittest.main()
