#!/usr/bin/env python3
"""Run generic root-vs-target model comparisons and render a report.

The configuration contains arbitrary model weights, Gemma CLI arguments, and
environment variables. No optimization (W8A8 or otherwise) is special-cased.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compare_mmlu import compare_results, load_results


@dataclass(frozen=True)
class ModelSpec:
    name: str
    weights: Path
    args: tuple[str, ...]
    env: dict[str, str]


@dataclass(frozen=True)
class RunMetrics:
    wall_seconds: float
    peak_rss_kib: int | None


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def parse_model_spec(data: dict[str, Any], base: Path) -> ModelSpec:
    if not isinstance(data, dict):
        raise ValueError("model must be an object")
    try:
        name = str(data["name"])
        weights = _resolve_path(str(data["weights"]), base)
    except KeyError as error:
        raise ValueError(f"model is missing {error.args[0]!r}") from error
    if not name or re.search(r"[^A-Za-z0-9_.-]", name):
        raise ValueError(f"invalid model name {name!r}")
    raw_args = data.get("args", [])
    if not isinstance(raw_args, list):
        raise ValueError(f"{name}: args must be an array")
    args = tuple(str(arg) for arg in raw_args)
    raw_env = data.get("env", {})
    if not isinstance(raw_env, dict):
        raise ValueError(f"{name}: env must be an object")
    env = {str(key): str(value) for key, value in raw_env.items()}
    return ModelSpec(name=name, weights=weights, args=args, env=env)


def _read_rss_kib(pid: int) -> int | None:
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except OSError:
        return None
    values: dict[str, int] = {}
    for line in status.splitlines():
        if line.startswith(("VmHWM:", "VmRSS:")):
            key, value, *_ = line.split()
            values[key.rstrip(":")] = int(value)
    return values.get("VmHWM", values.get("VmRSS"))


def run_command(
    command: list[str], env_updates: dict[str, str], stdout_path: Path,
    stderr_path: Path
) -> RunMetrics:
    env = os.environ.copy()
    env.update(env_updates)
    start = time.perf_counter()
    peak_rss_kib: int | None = None
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env)
        while process.poll() is None:
            rss = _read_rss_kib(process.pid)
            if rss is not None:
                peak_rss_kib = max(peak_rss_kib or 0, rss)
            time.sleep(0.02)
        rss = _read_rss_kib(process.pid)
        if rss is not None:
            peak_rss_kib = max(peak_rss_kib or 0, rss)
        return_code = process.returncode
    wall_seconds = time.perf_counter() - start
    if return_code != 0:
        tail = "\n".join(
            stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()[
                -20:
            ]
        )
        raise RuntimeError(
            f"command failed ({return_code}): {' '.join(command)}\n{tail}"
        )
    return RunMetrics(wall_seconds=wall_seconds, peak_rss_kib=peak_rss_kib)


def parse_prefixed_json(path: Path, prefix: str) -> dict[str, Any]:
    found: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            found = json.loads(line[len(prefix) :])
    if found is None:
        raise ValueError(f"{path}: no {prefix.strip()} line")
    return found


def parse_entropy(path: Path) -> dict[str, float | int]:
    text = path.read_text(encoding="utf-8")
    token_matches = re.findall(r"Number of input tokens: (\d+)", text)
    speed_matches = re.findall(
        r"\[([0-9.eE+-]+) tokens / sec\]", text
    )
    entropy_matches = re.findall(
        r"Total cross entropy: [0-9.eE+-]+ \[cumulative: ([0-9.eE+-]+)\]",
        text,
    )
    if not token_matches or not speed_matches or not entropy_matches:
        raise ValueError(f"{path}: incomplete cross-entropy output")
    tokens = int(token_matches[-1])
    if tokens == 0:
        raise ValueError(f"{path}: cross-entropy input has no tokens")
    total_bits = float(entropy_matches[-1])
    return {
        "tokens": tokens,
        "total_bits": total_bits,
        "bits_per_token": total_bits / tokens,
        "tokens_per_second": float(speed_matches[-1]),
    }



def render_table(rows: list[dict[str, Any]]) -> str:
    root = rows[0]
    root_entropy = root.get("entropy")
    lines = [
        "| Model | Entropy bits/token | Δ entropy | tok/s | Speedup | "
        "MMLU accuracy | Flips | Mean KL | p95 KL | Peak RSS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        entropy = row.get("entropy")
        if entropy and root_entropy:
            entropy_delta = 100.0 * (
                entropy["total_bits"] / root_entropy["total_bits"] - 1.0
            )
            speedup = 100.0 * (
                entropy["tokens_per_second"]
                / root_entropy["tokens_per_second"]
                - 1.0
            )
            entropy_text = f"{entropy['bits_per_token']:.4f}"
            delta_text = f"{entropy_delta:+.3f}%"
            speed_text = f"{entropy['tokens_per_second']:.2f}"
            speedup_text = f"{speedup:+.1f}%"
        else:
            entropy_text = delta_text = speed_text = speedup_text = "—"
        flips = row.get("flips")
        kl = row.get("kl")
        rss = row.get("peak_rss_kib")
        lines.append(
            "| {name} | {entropy} | {delta} | {speed} | {speedup} | "
            "{accuracy:.1f}% | {flips} | {mean_kl} | {p95_kl} | {rss} |".format(
                name=row["name"],
                entropy=entropy_text,
                delta=delta_text,
                speed=speed_text,
                speedup=speedup_text,
                accuracy=100.0 * row["mmlu"]["accuracy"],
                flips="—" if flips is None else f"{flips['flips_percent']:.2f}%",
                mean_kl="—" if kl is None else f"{kl['mean']:.6g}",
                p95_kl="—" if kl is None else f"{kl['p95']:.6g}",
                rss="—" if rss is None else f"{rss / 1024.0:.1f} MiB",
            )
        )
    return "\n".join(lines) + "\n"


def run_evaluation(
    spec: ModelSpec, build_dir: Path, output_dir: Path, mmlu: Path,
    max_questions: int, reference: Path, is_root: bool,
    entropy_path: Path | None
) -> tuple[dict[str, Any], Path]:
    stem = output_dir / spec.name
    mmlu_out = stem.with_suffix(".mmlu.out")
    mmlu_err = stem.with_suffix(".mmlu.err")
    command = [
        str(build_dir / "gemma_mmlu"),
        "--weights",
        str(spec.weights),
        "--input",
        str(mmlu),
        "--verbosity",
        "0",
    ]
    if max_questions:
        command.extend(["--max_questions", str(max_questions)])
    command.extend(
        ["--reference_out" if is_root else "--reference_in", str(reference)]
    )
    command.extend(spec.args)
    mmlu_run = run_command(command, spec.env, mmlu_out, mmlu_err)
    mmlu_summary = parse_prefixed_json(mmlu_out, "MMLU_SUMMARY ")
    kl_summary = (
        None
        if is_root
        else parse_prefixed_json(mmlu_out, "MMLU_KL_SUMMARY ")
    )

    entropy: dict[str, float | int] | None = None
    entropy_run: RunMetrics | None = None
    if entropy_path is not None:
        entropy_out = stem.with_suffix(".entropy.out")
        entropy_err = stem.with_suffix(".entropy.err")
        entropy_command = [
            str(build_dir / "single_benchmark"),
            "--weights",
            str(spec.weights),
            "--cross_entropy",
            str(entropy_path),
            "--verbosity",
            "0",
            *spec.args,
        ]
        entropy_run = run_command(
            entropy_command, spec.env, entropy_out, entropy_err
        )
        entropy = parse_entropy(entropy_out)

    peak_values = [mmlu_run.peak_rss_kib]
    if entropy_run is not None:
        peak_values.append(entropy_run.peak_rss_kib)
    peak_rss = max((value for value in peak_values if value is not None), default=None)
    return (
        {
            "name": spec.name,
            "weights": str(spec.weights),
            "args": list(spec.args),
            "env": spec.env,
            "mmlu": mmlu_summary,
            "kl": kl_summary,
            "entropy": entropy,
            "mmlu_wall_seconds": mmlu_run.wall_seconds,
            "entropy_wall_seconds": None
            if entropy_run is None
            else entropy_run.wall_seconds,
            "peak_rss_kib": peak_rss,
        },
        mmlu_out,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare arbitrary target models against a root model."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--build_dir", type=Path, default=Path("build"))
    parser.add_argument("--output_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path = args.config.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        base = config_path.parent
        root = parse_model_spec(config["root"], base)
        targets = [parse_model_spec(item, base) for item in config["targets"]]
        if not targets:
            raise ValueError("targets must contain at least one model")
        names = [root.name, *(target.name for target in targets)]
        if len(names) != len(set(names)):
            raise ValueError("model names must be unique")
        mmlu = _resolve_path(str(config["mmlu"]), base)
        entropy_path = (
            None
            if not config.get("cross_entropy")
            else _resolve_path(str(config["cross_entropy"]), base)
        )
        max_questions = int(config.get("max_questions", 0))
        if max_questions < 0:
            raise ValueError("max_questions must be non-negative")
        build_dir = args.build_dir.resolve()
        required_paths = [mmlu, root.weights]
        required_paths.extend(target.weights for target in targets)
        if entropy_path is not None:
            required_paths.append(entropy_path)
        missing = [str(path) for path in required_paths if not path.is_file()]
        if missing:
            raise ValueError("file does not exist: " + ", ".join(missing))
        required_programs = [build_dir / "gemma_mmlu"]
        if entropy_path is not None:
            required_programs.append(build_dir / "single_benchmark")
        missing_programs = [
            str(path) for path in required_programs if not path.is_file()
        ]
        if missing_programs:
            raise ValueError(
                "build executable does not exist: " + ", ".join(missing_programs)
            )
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir
            else (base / f"{config_path.stem}-results").resolve()
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        reference = output_dir / f"{root.name}.root-kl.bin"

        root_row, root_output = run_evaluation(
            root, build_dir, output_dir, mmlu, max_questions,
            reference, True, entropy_path
        )
        root_row["flips"] = None
        root_results = load_results(root_output)
        rows = [root_row]
        for target in targets:
            row, target_output = run_evaluation(
                target, build_dir, output_dir, mmlu,
                max_questions, reference, False, entropy_path
            )
            row["flips"] = compare_results(
                root_results, load_results(target_output)
            )
            rows.append(row)

        report = {
            "schema_version": 1,
            "mmlu": str(mmlu),
            "cross_entropy": None if entropy_path is None else str(entropy_path),
            "reference": str(reference),
            "models": rows,
        }
        (output_dir / "comparison.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        table = render_table(rows)
        (output_dir / "comparison.md").write_text(table, encoding="utf-8")
        print(table, end="")
        return 0
    except (KeyError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
