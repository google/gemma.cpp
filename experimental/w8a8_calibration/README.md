# W8A8 calibration

Offline calibration for the optional W8A8 model path. The recommended method is
**plain weight-only GPTQ**: keep `GEMMA_MM_I8_L2_SCALE=0` and omit
`--activation-correction`. Calibration changes packed weights and group scales;
it adds no work to the inference kernel. Activation correction and RMS
rescaling remain experimental options and are not part of this recommendation.

The scripts require Python 3.11+, NumPy, and CUDA-enabled PyTorch with a compatible
GPU. `--rows` controls weight-row batch size; the full input covariance still
requires memory proportional to K squared. Run these commands from the repository
root, using a gemma binary built with the W8A8 hooks and no other `GEMMA_MM_I8_*`
settings inherited from earlier experiments.

| Model | Quantization group | GPTQ damping | `MIN_K_SPLITS` |
|---|---:|---:|---:|
| Gemma 270M | 64 | 0.1 | 1 |
| Gemma 1B | 128 | 1.0 | 0 |

## Capture, export, calibrate

Choose paths and settings; use fresh output directories for each run:

```sh
W8_GEMMA=./build/gemma
W8_WEIGHTS=/path/to/270m-sfp-it.sbs
W8_WORK=/tmp/gemma-w8a8-270m
W8_GROUP=64
W8_DAMP=0.1
W8_MIN_K_SPLITS=1
mkdir -p "$W8_WORK"
```

Capture teacher inputs from diverse calibration prompts. In the interactive
session below, each prompt starts a new conversation; `%q` exits. Keep calibration
prompts separate from evaluation prompts. Capture works with W8A8 disabled and
records the BF16-rounded inputs used by the SFP reference. The row cap is per
tensor, with a global 1 GiB capture limit; check the JSON row counts before fitting.

```sh
env GEMMA_MM_I8=0 GEMMA_MM_I8_L2_SCALE=0 \
  GEMMA_MM_I8_BLOCK_SIZE=128 GEMMA_MM_I8_HASH_BITS=32 \
  GEMMA_MM_I8_CALIBRATION_CAPTURE="$W8_WORK/capture" \
  GEMMA_MM_I8_CALIBRATION_SAMPLES=512 \
  GEMMA_MM_I8_CALIBRATION_ROWS_PER_CALL=32 \
  "$W8_GEMMA" --weights "$W8_WEIGHTS" --top_k 1 \
  --deterministic 1 --multiturn 0 --max_generated_tokens 16
```

Export the matching rotated weights. One generated token exercises the dense
transformer and output head. Exported floats omit each tensor's `B.Scale()`;
the importer applies that factor once when packing.

```sh
env GEMMA_MM_I8=1 GEMMA_MM_I8_L2_SCALE=0 GEMMA_MM_I8_MICROSCALE=1 \
  GEMMA_MM_I8_BLOCK_SIZE=128 GEMMA_MM_I8_HASH_BITS=32 \
  GEMMA_MM_I8_QUANT_BLOCK_SIZE="$W8_GROUP" \
  GEMMA_MM_I8_EXPORT_DIR="$W8_WORK/raw_weights" \
  "$W8_GEMMA" --weights "$W8_WEIGHTS" --top_k 1 \
  --max_generated_tokens 1 --prompt 'Explain how a bicycle works.'

python3 experimental/w8a8_calibration/gptq_calibrate.py \
  --weights "$W8_WORK/raw_weights" --capture "$W8_WORK/capture" \
  --output "$W8_WORK/packed" --group "$W8_GROUP" --damp "$W8_DAMP" \
  --device cuda:0
```

The output manifest records tensor dimensions, reconstruction errors and source
hashes. Review its tensor coverage: missing capture files are skipped, and missing
import files fall back to ordinary packing. Reconstruction error is a calibration
metric, not a guarantee of lower end-to-end KL.

## Import

Use the same checkpoint, rotation and quantization group:

```sh
env GEMMA_MM_I8=1 GEMMA_MM_I8_L2_SCALE=0 GEMMA_MM_I8_MICROSCALE=1 \
  GEMMA_MM_I8_BLOCK_SIZE=128 GEMMA_MM_I8_HASH_BITS=32 \
  GEMMA_MM_I8_QUANT_BLOCK_SIZE="$W8_GROUP" \
  GEMMA_MM_I8_MIN_K_SPLITS="$W8_MIN_K_SPLITS" \
  GEMMA_MM_I8_IMPORT_DIR="$W8_WORK/packed" \
  "$W8_GEMMA" --weights "$W8_WEIGHTS" --top_k 1
```

For the separately validated dual-activation inference experiment, add
`GEMMA_MM_I8_PACKED_HEAD=1`, `GEMMA_MM_I8_PACKED_HEAD_FULL_K=1`,
`GEMMA_MM_I8_DUAL_A_BODY=1`, `GEMMA_MM_I8_DUAL_A_HEAD=1`, and
`GEMMA_MM_I8_MATCH_BF16_A=1`. These runtime settings affect performance and quality;
measure them separately from calibration. The writer's covariance remains fitted
to the original single-stream activation quantizer.
`MIN_K_SPLITS` controls the fixed MatMul schedule when autotuning is disabled.
Automatic dual-A routing currently requires the supported x86-64 GCC build
using the AVX2 target on an AVX-VNNI-capable CPU. The flags alone do not promise
the same behavior or results on other hardware or compiler targets.

## Compatibility and optional RMS experiments

Files are little-endian. `W8RAW001` stores rotated unscaled F32 weight rows;
`MMI8WQ01` stores signed int8 rows followed by group-major F32 scales. The runtime
checks dimensions, group size, rotation and hash settings. These checks do **not**
identify a checkpoint: retain the manifest and use the exact checkpoint that was
exported. Deterministic reproduction also requires matching calibration prompts,
token sequences, capture selection and runtime settings.

`prepare_rms_scaling.py CAPTURE OUTPUT` writes optional FFN input RMS files;
`--all-consumers` includes normalization consumers. These are used with
`GEMMA_MM_I8_L2_SCALE=1` and `GEMMA_MM_I8_SCALE_CALIBRATION_DIR=OUTPUT` when exporting
an experimental scaled basis. Export again after changing any scaling setting.
`W8RAW002` and `MMI8WQ02` include the exact F32 input-scale vector; import requires
a byte-identical vector. The writer preserves those bytes and transforms the
captured inputs into that basis. Scaled transforms currently support rotations
64/128 with hash32 and assume unit activation `A.Scale()`, as in these Gemma
buffers. Their inverse-rotation reconstruction is approximate and reported in
the manifest. Never pair unscaled packed weights with a scaled input basis.

Only reusable logic belongs here. Keep checkpoints, captured tensors, packed
weights, logs, evaluation reports and binaries outside this source directory.
