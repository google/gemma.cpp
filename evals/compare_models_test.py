#!/usr/bin/env python3
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from compare_models import (
    parse_entropy,
    parse_model_spec,
    parse_prefixed_json,
    render_table,
    main,
    RunMetrics,
    run_command,
    run_evaluation,
)


class CompareModelsTest(unittest.TestCase):
    def test_parse_model_spec_resolves_paths_and_environment(self) -> None:
        spec = parse_model_spec(
            {
                "name": "target.v1",
                "weights": "models/model.sbs",
                "args": ["--num_threads", 4],
                "env": {"EXAMPLE_MODE": 1},
            },
            Path("/work"),
        )

        self.assertEqual(spec.name, "target.v1")
        self.assertEqual(spec.weights, Path("/work/models/model.sbs"))
        self.assertEqual(spec.args, ("--num_threads", "4"))
        self.assertEqual(spec.env, {"EXAMPLE_MODE": "1"})

    def test_parse_prefixed_json_rejects_duplicate_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.log"
            path.write_text("MMLU_SUMMARY {}\nMMLU_SUMMARY {}\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                parse_prefixed_json(path, "MMLU_SUMMARY ")

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
            "mmlu_inference_seconds": 8.0,
            "mmlu_wall_seconds": 12.0,
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
            "mmlu_inference_seconds": 4.0,
            "mmlu_wall_seconds": 20.0,
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

        self.assertIn("| 20.000 | 4.000 | 2.000x |", table)
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

    def test_rejects_names_and_owned_options(self) -> None:
        for name in (".", "..", "", "../x", "a/b", "a|b", None):
            with self.subTest(name=name), self.assertRaises(ValueError):
                parse_model_spec({"name": name, "weights": "x"}, Path("/tmp"))
        for option in ("--input", "--reference_out=x", "--weights", "--max_questions"):
            with self.subTest(option=option), self.assertRaises(ValueError):
                parse_model_spec(
                    {"name": "ok", "weights": "x", "args": [option]}, Path("/tmp")
                )

    def test_child_process_success_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "out"
            err = Path(directory) / "err"
            metrics = run_command(
                [sys.executable, "-c", "import os; print(os.environ['EVAL_TEST'])"],
                {"EVAL_TEST": "value"},
                out,
                err,
            )
            self.assertEqual(out.read_text().strip(), "value")
            self.assertGreater(metrics.wall_seconds, 0)
            with self.assertRaisesRegex(RuntimeError, r"failed \(3\)"):
                run_command([sys.executable, "-c", "raise SystemExit(3)"], {}, out, err)

    def test_interrupt_terminates_child(self) -> None:
        child = Mock()
        child.poll.return_value = None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with patch("compare_models.subprocess.Popen", return_value=child), patch(
                "compare_models._read_rss_kib", side_effect=KeyboardInterrupt
            ), self.assertRaises(KeyboardInterrupt):
                run_command(["fake"], {}, path / "out", path / "err")
        child.terminate.assert_called_once()
        child.wait.assert_called_once_with(timeout=5)

    def test_runner_report_and_dotted_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "weights.sbs").touch()
            (base / "mmlu.json").write_text("{}")
            (base / "gemma_mmlu").touch()
            config = {
                "mmlu": "mmlu.json",
                "max_questions": 1,
                "root": {"name": "root.v1", "weights": "weights.sbs"},
                "targets": [
                    {"name": "target.v1", "weights": "weights.sbs"},
                    {"name": "target.v2", "weights": "weights.sbs"},
                ],
            }
            config_path = base / "config.json"
            config_path.write_text(json.dumps(config))
            output = base / "results"
            calls = []

            def fake_run(command, env, stdout_path, stderr_path):
                calls.append(stdout_path.name)
                is_root = "--reference_out" in command
                ref_flag = "--reference_out" if is_root else "--reference_in"
                reference = Path(command[command.index(ref_flag) + 1])
                if is_root:
                    reference.write_text("reference")
                else:
                    self.assertEqual(reference.read_text(), "reference")
                self.assertEqual(
                    command[command.index("--weights") + 1], str(base / "weights.sbs")
                )
                result = {
                    "id": 1,
                    "expected": "A",
                    "predicted": "A" if is_root else "B",
                    "correct": is_root,
                }
                records = {
                    "MMLU_RESULT": result,
                    "MMLU_SUMMARY": {
                        "answers": 1,
                        "correct": int(is_root),
                        "accuracy": float(is_root),
                    },
                    "MMLU_TIMING": {
                        "generate_seconds": 2,
                        "sample_seconds": 1,
                        "inference_seconds": 1,
                    },
                }
                if not is_root:
                    result["full_vocab_kl"] = 0.1
                    records["MMLU_KL_SUMMARY"] = {
                        "samples": 1,
                        "mean": 0.1,
                        "median": 0.1,
                        "p95": 0.1,
                        "max": 0.1,
                        "direction": "root||target",
                        "unit": "nats",
                    }
                stdout_path.write_text(
                    "".join(k + " " + json.dumps(v) + "\n" for k, v in records.items())
                )
                stderr_path.touch()
                return RunMetrics(3.0, 1024)

            argv = [
                "compare_models.py",
                "--config",
                str(config_path),
                "--build_dir",
                str(base),
                "--output_dir",
                str(output),
            ]
            with patch.object(sys, "argv", argv), patch(
                "compare_models.run_command", side_effect=fake_run
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            self.assertEqual(
                calls, ["root.v1.mmlu.out", "target.v1.mmlu.out", "target.v2.mmlu.out"]
            )
            report = json.loads((output / "comparison.json").read_text())
            self.assertEqual(report["models"][1]["flips"]["correct_to_incorrect"], 1)
            self.assertEqual(report["models"][2]["kl"]["mean"], 0.1)
            table = (output / "comparison.md").read_text().splitlines()
            self.assertEqual(len(table[0].split("|")), len(table[1].split("|")))
            # Reruns cannot silently overwrite a report or mix stale artifacts.
            with patch.object(sys, "argv", argv), contextlib.redirect_stderr(
                io.StringIO()
            ):
                self.assertEqual(main(), 1)

    def test_rejects_incomplete_or_invalid_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            spec = parse_model_spec({"name": "test", "weights": "model"}, base)
            records = {
                "MMLU_RESULT": {
                    "id": 1,
                    "expected": "A",
                    "predicted": "A",
                    "correct": True,
                },
                "MMLU_SUMMARY": {"answers": 1, "correct": 1, "accuracy": 1.0},
                "MMLU_TIMING": {
                    "generate_seconds": 2,
                    "sample_seconds": 1,
                    "inference_seconds": 1,
                },
            }
            for change in (
                {"MMLU_SUMMARY": None},
                {"MMLU_SUMMARY": {"answers": 2, "correct": 1, "accuracy": 0.5}},
                {
                    "MMLU_TIMING": {
                        "generate_seconds": 2,
                        "sample_seconds": 1,
                        "inference_seconds": float("nan"),
                    }
                },
            ):
                changed = {**records, **change}

                def fake_run(command, env, out, err):
                    out.write_text(
                        "".join(
                            key + " " + json.dumps(value) + "\n"
                            for key, value in changed.items()
                            if value is not None
                        )
                    )
                    return RunMetrics(3, None)

                with self.subTest(change=change), patch(
                    "compare_models.run_command", side_effect=fake_run
                ), self.assertRaises(ValueError):
                    run_evaluation(
                        spec, base, base, base / "mmlu", 1, base / "ref", True, None
                    )

    def test_entropy_uses_final_cumulative_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "entropy.log"
            path.write_text(
                "Number of input tokens: 8\n"
                "Took 1 s [4 tokens / sec]\nTotal cross entropy: 3 [cumulative: 3]\n"
                "Took 2 s [4 tokens / sec]\nTotal cross entropy: 5 [cumulative: 8]\n"
            )
            self.assertEqual(parse_entropy(path)["bits_per_token"], 1)
            for text in (
                "",
                "Number of input tokens: 0\n[1 tokens / sec]\nTotal cross entropy: 0 [cumulative: 0]",
                "Number of input tokens: 8\n[0 tokens / sec]\nTotal cross entropy: 1 [cumulative: 1]",
            ):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    parse_entropy(path)


if __name__ == "__main__":
    unittest.main()
