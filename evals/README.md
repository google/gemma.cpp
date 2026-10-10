# Model comparison and MMLU

`gemma_mmlu` scores multiple-choice questions and emits machine-readable results.
`compare_models.py` runs a root model and one or more targets **serially**, then
writes `comparison.json` and `comparison.md`. It accepts arbitrary weight files,
CLI arguments, and environment overrides; no quantization method is required.
Models must use the same tokenizer and vocabulary for full-vocabulary KL.

## Build and test

From the repository root, with the usual CMake dependencies and GoogleTest:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGEMMA_ENABLE_TESTS=ON -DHWY_ENABLE_TESTS=OFF
cmake --build build --target gemma_mmlu model_comparison_test -j2
ctest --test-dir build -R 'ModelComparison|model_comparison_python' --output-on-failure
python3 -B -m unittest discover -s evals -p 'compare*_test.py'
```

Python 3.10+ is required; the scripts use only the standard library. CMake
registers Python tests when an interpreter is available. Optional cross-entropy
runs also require `cmake --build build --target single_benchmark -j2`.
Bazel targets `//:gemma_mmlu` and `//:model_comparison_test` are also provided.

## Compare models

Save a configuration such as `comparison.json`:

```json
{
  "mmlu": "gemma/evals/mmlu.json",
  "max_questions": 0,
  "root": {
    "name": "root",
    "weights": "root.sbs",
    "args": ["--num_threads", "6", "--pin", "1"],
    "env": {}
  },
  "targets": [
    {
      "name": "target",
      "weights": "target.sbs",
      "args": ["--num_threads", "6", "--pin", "1"],
      "env": {}
    }
  ]
}
```

Paths for weights, MMLU, and optional `cross_entropy` text are relative to the
configuration file. Extra CLI arguments retain their ordinary interpretation
relative to the runner's working directory. Names must be unique and contain
only letters, digits, underscores, dots, or hyphens, starting with a letter,
digit, or underscore. The runner owns input, output, weights, question-limit,
cross-entropy, and verbosity flags; these cannot be overridden in `args`.
Environment overrides extend the calling process's environment. Record or clear
any inherited settings relevant to the experiment, and use equivalent runtime
settings across models when comparing performance.

```sh
python3 -B evals/compare_models.py --config comparison.json \
  --build_dir build --output_dir eval-results/run-1
```

The output directory must be new or empty. Each model retains its stdout and
stderr. The root additionally writes `<name>.root-kl.jsonl`, reused by all
targets. A failed process or invalid result stops the series without producing a
successful comparison report; existing logs remain available for diagnosis.
`max_questions: 0` runs all samples; a positive value selects the first N.
To measure run-to-run variation, use the same weights and settings for root
and target. Existing timing-dependent MatMul autotuning can change floating-point
results between processes, so this self-comparison need not have zero KL or
flips. The later autotuner-control step is needed for reproducible schedules;
`--deterministic` controls sampling, not MatMul tuning.

## Scoring and metrics

Each question uses the dataset prompt, the runner's answer-only instruction,
and the model's chat template. The runner scores the **first generated token**.
It accepts single-token encodings of `A`–`D`, with or without a leading space,
and uses the highest logit among each label's available spellings. A label with
no single-token encoding causes an error. Equal label scores select A before B,
then C, then D. This replaces the previous runner's free-form answer decoding;
results from the two protocols should not be mixed.

- `MMLU_RESULT`: question ID, expected/predicted label, correctness, four label
  logits, their softmax probabilities, and the winning margin. These
  probabilities normalize the four label scores; they are not full-vocabulary
  token probabilities. Target runs also report `full_vocab_kl`.
- `MMLU_SUMMARY`: number answered, number correct, and accuracy as a fraction.
- `MMLU_KL_SUMMARY`: mean, median, p95, and maximum of
  `KL(root || target)` in nats over the **whole vocabulary at the first answer
  token**, after the model's logit soft cap. Percentiles use linear interpolation.
  This is not KL averaged over every token of all possible answer strings.
- **Flips**: correct-to-incorrect plus incorrect-to-correct transitions.
  Wrong-to-wrong changes are separate; `answer_changes` counts all changed
  labels. Aggregate accuracy can hide changes in individual answers.
- `MMLU_TIMING`: time in `Generate`, time inside the evaluation sampling callback,
  and their difference (`inference_seconds`). The latter includes MatMul
  autotuning; it excludes prompt construction, reference serialization/parsing,
  and result printing. This step does not add autotuner controls.
- **Wall time** includes process startup, model loading, evaluation, and reference
  I/O. It is not directly comparable between writing and reading references.
  Peak RSS is sampled from Linux `/proc` every 20 ms and can miss short-lived
  peaks; unavailable measurements are shown as null/—.
- Optional **cross entropy** comes from `single_benchmark`: total bits divided
  by input tokens, including its uniform first-token cost. The report uses the
  final cumulative entropy and throughput, and compares bits/token to root.

The bundled 83-question fixture is a small regression dataset, not the complete
MMLU benchmark. A single run is not a statistically reliable speed measurement.

## Reference files

References are versioned JSON Lines, with one metadata header followed by one
record per question. The header contains the exact dataset text and hex-encoded
serialized tokenizer bytes. Records store IDs, expected labels, tokenized
prompts, and float32 logits that round-trip without precision loss. Exact
comparisons avoid hash collisions and reject mismatched datasets, tokenizers,
question limits, or chat-template tokenization. Truncated, extra, malformed, or
non-finite records fail explicitly.

Text is inspectable and portable, but larger than binary; expect hundreds of
MiB for an 83-question, 262k-vocabulary run. Targets stream one question at a
time. Reference I/O is kept outside the reported `Generate` time.

For a standalone root/target pair:

```sh
build/gemma_mmlu --weights root.sbs --input gemma/evals/mmlu.json \
  --reference_out root.jsonl > root.out
build/gemma_mmlu --weights target.sbs --input gemma/evals/mmlu.json \
  --reference_in root.jsonl > target.out
python3 -B evals/compare_mmlu.py root.out target.out
```
