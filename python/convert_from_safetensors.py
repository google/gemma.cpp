# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert a PaliGemma[1/2] model from SafeTensors to gemma.cpp format."""
# Tested with:
# - PG1: huggingface.co/google/paligemma-3b-pt-224
# - PG1: huggingface.co/merve/paligemma_vqav2
# - PG2: huggingface.co/google/paligemma2-3b-pt-448
# - PG2: huggingface.co/merve/paligemma2-3b-vqav2
# The last one above is a Lora model, so the merged weights were saved using:
# model_name = "google/paligemma2-3b-pt-448"
# lora_weights_path = "merve/paligemma2-3b-vqav2"
# model = PaliGemmaForConditionalGeneration.from_pretrained(model_name)
# model = PeftModel.from_pretrained(model, lora_weights_path)
# model = model.merge_and_unload()
# model.save_pretrained("/tmp/lora-model")

from collections.abc import Sequence
import csv
import json
import os
import sys
from typing import Any, Dict

from absl import app
from absl import flags
from absl import logging
import numpy as np
import safetensors
import torch

from compression.python import compression
from python import configs


def flatten_f32(x: np.ndarray) -> np.ndarray:
  """Flattens an array.

  Args:
    x: input array

  Returns:
    Flattened array.
  """
  return x.ravel().astype(np.float32, copy=False)


def compute_scale(x: np.ndarray) -> float:
  """Rescales weight tensor to fit max magnitude within 1.875.

  Args:
    x: input array

  Returns:
    Scale value (1.0 means no rescaling).
  """
  magnitude = np.max(np.abs(x))
  return max(1.0, magnitude / 1.875)


def _is_float_param(param_name: str) -> bool:
  """Returns whether the tensor should be stored as float32."""
  for prefix in [
      "img_pos_emb",
      "attn_out_b",
      "linear_0_b",
      "linear_1_b",
      "qkv_ein_b",
      "img_emb_bias",
      "img_head_bias",
  ]:
    if param_name.startswith(prefix):
      return True
  return False


def _is_bf16_param(param_name: str) -> bool:
  """Returns whether the tensor should be stored as bf16."""
  for prefix in ["pre_", "post_", "c_", "img_head_kernel"]:
    if param_name.startswith(prefix):
      return True
  return False


# Layernorm names are slightly confusing in HF transformers between versions.
# Gemma layernorms:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma/modeling_gemma.py
# input_layernorm attn residual post_attention_layernorm mlp residual
# Gemma2 layernorms:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modeling_gemma2.py
# input_layernorm attn post_attention_layernorm residual
#   pre_feedforward_layernorm mlp post_feedforward_layernorm residual
# Note that post_attention_layernorm denotes something different.
# For comparison, the Big Vision Gemma2 keeps the same name for the same norm:
# pre_attention_norm attn [post_attention_norm] residual
#   pre_ffw_norm mlp [post_ffw_norm] residual


# Tuples correspond to (transformers-name, shape, sbs-name).
# The qkv-einsum weights are part of llm-layers but are handled separately and
# thus not included in the list.
def _get_layer_config(dims: Dict[str, Any]):
  """Returns a dictionary of layer configurations.

  Args:
    dims: A dictionary of (mostly) dimension values.

  Returns:
    A dictionary of layer configurations.
  """
  model_dim = dims["model_dim"]
  hidden_dim = dims["hidden_dim"]
  vit_seq_len = dims["vit_seq_len"]
  config = {
      "llm-non-layers": [
          (
              "language_model.model.embed_tokens.weight",
              (257152, model_dim),
              "c_embedding",
          ),
          ("language_model.model.norm.weight", (model_dim,), "c_final_norm"),
      ],
      "llm-layers": [
          (
              "language_model.model.layers.%d.mlp.down_proj.weight",
              (model_dim, hidden_dim),
              "linear_w",
          ),
      ],
      "img-non-layers": [
          (
              "vision_tower.vision_model.post_layernorm.bias",
              (1152,),
              "enc_norm_bias",
          ),
          (
              "vision_tower.vision_model.post_layernorm.weight",
              (1152,),
              "enc_norm_scale",
          ),
          (
              "vision_tower.vision_model.embeddings.patch_embedding.bias",
              (1152,),
              "img_emb_bias",
          ),
          (
              "vision_tower.vision_model.embeddings.patch_embedding.weight",
              (1152, 14, 14, 3),
              "img_emb_kernel",
          ),
          ("multi_modal_projector.linear.bias", (model_dim,), "img_head_bias"),
          (
              "multi_modal_projector.linear.weight",
              (model_dim, 1152),
              "img_head_kernel",
          ),
          (
              "vision_tower.vision_model.embeddings.position_embedding.weight",
              (vit_seq_len, 1152),
              "img_pos_emb",
          ),
      ],
      "img-layers": [
          (
              "vision_tower.vision_model.encoder.layers.%d.layer_norm1.bias",
              (1152,),
              "ln_0_bias",
          ),
          (
              "vision_tower.vision_model.encoder.layers.%d.layer_norm1.weight",
              (1152,),
              "ln_0_scale",
          ),
          (
              "vision_tower.vision_model.encoder.layers.%d.layer_norm2.bias",
              (1152,),
              "ln_1_bias",
          ),
          (
              "vision_tower.vision_model.encoder.layers.%d.layer_norm2.weight",
              (1152,),
              "ln_1_scale",
          ),
          (
              "vision_tower.vision_model.encoder.layers.%d.mlp.fc1.bias",
              (4304,),
              "linear_0_b",
          ),
          (
              "vision_tower.vision_model.encoder.layers.%d.mlp.fc1.weight",
              (4304, 1152),
              "linear_0_w",
          ),
          (
              "vision_tower.vision_model.encoder.layers.%d.mlp.fc2.bias",
              (1152,),
              "linear_1_b",
          ),
          (
              "vision_tower.vision_model.encoder.layers.%d.mlp.fc2.weight",
              (1152, 4304),
              "linear_1_w",
          ),
          (
              "vision_tower.vision_model.encoder.layers.%d.self_attn.out_proj.bias",
              (1152,),
              "attn_out_b",
          ),
          (
              "vision_tower.vision_model.encoder.layers.%d.self_attn.out_proj.weight",
              (1152, 16 * 72),
              "attn_out_w",
          ),
      ],
  }
  if dims["has_post_norm"]:  # See longer comment above.
    config["llm-layers"] += [
        (
            "language_model.model.layers.%d.input_layernorm.weight",
            (model_dim,),
            "pre_att_ns",
        ),
        (
            "language_model.model.layers.%d.pre_feedforward_layernorm.weight",
            (model_dim,),
            "pre_ff_ns",
        ),
        (
            "language_model.model.layers.%d.post_attention_layernorm.weight",
            (model_dim,),
            "post_att_ns",
        ),
        (
            "language_model.model.layers.%d.post_feedforward_layernorm.weight",
            (model_dim,),
            "post_ff_ns",
        ),
    ]
  else:
    config["llm-layers"] += [
        (
            "language_model.model.layers.%d.input_layernorm.weight",
            (model_dim,),
            "pre_att_ns",
        ),
        (
            "language_model.model.layers.%d.post_attention_layernorm.weight",
            (model_dim,),
            "pre_ff_ns",
        ),
    ]
  return config


def _get_dimensions(params):
  """Returns a dictionary of dimension values.

  Args:
    params: A dictionary with parameters.

  Returns:
    A dictionary of dimension values.
  """
  dims = {}
  # For PG1 and PG2-{3B,10B} head_dim is 256, would need update for PG2-28B.
  # Unfortunately not easily available in any of the input sizes.
  dims["head_dim"] = 256
  dims["model_dim"] = params["multi_modal_projector.linear.bias"].shape[0]
  dims["hidden_dim"] = params[
      "language_model.model.layers.0.mlp.gate_proj.weight"
  ].shape[0]
  dims["num_heads"] = (
      params["language_model.model.layers.0.self_attn.q_proj.weight"].shape[0]
      // dims["head_dim"]
  )
  dims["vit_seq_len"] = params[
      "vision_tower.vision_model.embeddings.position_embedding.weight"
  ].shape[0]
  dims["num_llm_layers"] = len(
      set([k for k in params.keys() if "input_layernorm.weight" in k])
  )
  dims["has_post_norm"] = (
      "language_model.model.layers.0.post_feedforward_layernorm.weight"
      in params
  )
  return dims


def export_paligemma_sbs(
    model_specifier: str,
    load_path: str,
    tokenizer_file: str,
    csv_file: str,
    sbs_file: str,
) -> None:
  """Exports sbs file from paligemma safetensors file(s)."""

  # If this is a multi-part checkpoint, get the list of files from the json.
  if load_path.endswith(".json"):
    with open(load_path, "r") as f:
      j_obj = json.load(f)
    files = list(set(j_obj["weight_map"].values()))
    files = [os.path.join(os.path.dirname(load_path), f) for f in files]
  else:
    files = [load_path]

  # Read the parameters from the files.
  params = {}
  for file in files:
    with safetensors.safe_open(file, framework="pt") as f:
      for k in f.keys():
        params[k] = f.get_tensor(k)
        print(k, params[k].shape, params[k].view(-1)[0].item())

  # See https://tinyurl.com/paligemmavocab - HF transformers extends the
  # embedding matrix by 64. Undo that here.
  params["language_model.model.embed_tokens.weight"] = params[
      "language_model.model.embed_tokens.weight"
  ][:-64]

  writer = compression.SbsWriter(sbs_file)
  metadata = []
  scales = {}
  dims = _get_dimensions(params)
  layer_config = _get_layer_config(dims)

  # Adds a parameter with expected shape to the writer.
  def add_data(param_name, data, expected_shape, sbs_name, layer_index=None):
    # Check shape.
    if not isinstance(expected_shape, tuple):
      expected_shape = (expected_shape,)
    print(f"Writing {param_name} with shape {data.shape} e:{expected_shape}")
    assert data.shape == expected_shape, param_name

    # Here we assume that the read data is a torch tensor and then convert it to
    # a numpy array.
    assert isinstance(data, torch.Tensor)
    data = data.to(torch.float32).numpy()
    data = np.array(data)

    # Add the layer index to the param name and sbs name if needed.
    if layer_index is not None:
      param_name = param_name % layer_index
      sbs_name = sbs_name + f"_{layer_index}"

    # Flatten the data and get scale.
    value = flatten_f32(data)
    scale = compute_scale(value)
    both_names = param_name + "::" + sbs_name
    print(f"Param {both_names} has scale {scale}")
    metadata.append((both_names, data.dtype, data.shape, scale))

    # Determine the type as which to insert.
    if _is_float_param(sbs_name):
      packed = configs.Type.kF32
      print(f"Inserting {both_names} as float (f32) (no scaling)")
    elif _is_bf16_param(sbs_name) or param_name.startswith("vision_tower"):
      packed = configs.Type.kBF16
      print(f"Inserting {both_names} as BF16 (no scaling)")
    else:
      packed = configs.Type.kSFP
      # Assumes that all scales are 1.0 for SFP. Consider adding scales.
      # They would still need to be written, but would be collected here.
      assert scale == 1.0, f"Scale for {both_names} is not 1.0"
      if scale != 1.0:
        value = value / scale
      scales[sbs_name] = scale  # Unused at the moment.
      print(f"Inserting {both_names} as SFP with scale {scale}")
    sys.stdout.flush()

    # Add the data to the writer.
    info = configs.TensorInfo()
    info.name = sbs_name
    info.shape = data.shape
    writer.insert(sbs_name, value, packed, info)

  def add_qkv_einsum(i):  # Handle qkv for layer i.
    name = "language_model.model.layers.%d.self_attn.q_proj.weight"  # (N*H, D)
    q_i = params.pop(name % i)
    (nh, d) = q_i.shape
    h = dims["head_dim"]
    n = dims["num_heads"]
    assert nh == n * h
    assert dims["model_dim"] == d
    q_i = q_i.reshape(n, h, d)
    name = "language_model.model.layers.%d.self_attn.k_proj.weight"  # (K*H, D)
    k_i = params.pop(name % i)
    kh = k_i.shape[0]
    k = kh // h
    assert k_i.shape[1] == d
    k_i = k_i.reshape(k, h, d)
    name = "language_model.model.layers.%d.self_attn.v_proj.weight"  # (K*H, D)
    v_i = params.pop(name % i)
    assert v_i.shape[0] == kh
    assert v_i.shape[1] == d
    v_i = v_i.reshape(k, h, d)
    # Stack and reshape KV to interleave (k,v), (k,v), ...
    stacked = torch.stack((k_i, v_i), dim=0)  # (2, K, H, D)
    transposed = stacked.transpose(0, 1)  # (K, 2, H, D)
    reshaped = transposed.reshape(2 * k, h, d)  # (2K, H, D)
    # Concatenate Q and KV to get the full qkv.
    qkv_i = torch.cat([q_i, reshaped], dim=0)
    name = "language_model.model.layers.%d.self_attn.qkv_proj.weight"
    expected_shape = (n + 2 * k, h, d)  # (N+2K, H, D)
    sbs_name = "qkv_ein"
    add_data(name, qkv_i, expected_shape, sbs_name, i)

  def add_att_einsum(i):  # Handle att_ein for layer i.
    name = "language_model.model.layers.%d.self_attn.o_proj.weight"  # (D, N*H)
    o_i = params.pop(name % i)
    (d, nh) = o_i.shape
    h = dims["head_dim"]
    n = dims["num_heads"]
    assert nh == n * h
    o_i = o_i.reshape(d, n, h).permute(1, 0, 2)  # (D, N, H) -> (N, D, H)
    expected_shape = (n, d, h)
    sbs_name = "att_ein"
    add_data(name, o_i, expected_shape, sbs_name, i)

  # Join gate and up projection weights to gating_einsum for layer i.
  def add_gating_einsum(i):
    name = "language_model.model.layers.%d.mlp.gate_proj.weight"
    gate_i = params.pop(name % i)
    f, d = gate_i.shape
    assert dims["hidden_dim"] == f
    assert dims["model_dim"] == d
    name = "language_model.model.layers.%d.mlp.up_proj.weight"
    up_i = params.pop(name % i)
    assert up_i.shape == gate_i.shape
    gating_einsum_i = torch.stack([gate_i, up_i], dim=0)
    name = "language_model.model.layers.%d.mlp.gating_einsum.weight"
    expected_shape = (2, f, d)
    sbs_name = "gating_ein"
    add_data(name, gating_einsum_i, expected_shape, sbs_name, i)

  # Handle the q and kv einsum parts for layer i in the ViT - merge into qkv.
  def add_vit_qkv_einsum(i):
    # Weights first.
    prefix = "vision_tower.vision_model.encoder.layers.%d.self_attn"
    name = prefix + ".q_proj.weight"  # (16 * 72, 1152)
    q_i = params.pop(name % i)
    q_i = q_i.reshape(16, 72, 1152)
    name = prefix + ".k_proj.weight"  # (16 * 72, 1152)
    k_i = params.pop(name % i)
    k_i = k_i.reshape(16, 72, 1152)
    name = prefix + ".v_proj.weight"  # (16 * 72, 1152)
    v_i = params.pop(name % i)
    v_i = v_i.reshape(16, 72, 1152)
    qkv_i, shape = torch.stack([q_i, k_i, v_i], dim=1), (16, 3, 72, 1152)
    name = prefix + ".qkv_proj.weight"
    sbs_name = "qkv_ein_w"
    add_data(name, qkv_i, shape, sbs_name, i)
    # Now the biases.
    name = prefix + ".q_proj.bias"  # (16 * 72)
    q_i = params.pop(name % i)
    q_i = q_i.reshape(16, 72)
    name = prefix + ".k_proj.bias"  # (16 * 72)
    k_i = params.pop(name % i)
    k_i = k_i.reshape(16, 72)
    name = prefix + ".v_proj.bias"  # (16 * 72)
    v_i = params.pop(name % i)
    v_i = v_i.reshape(16, 72)
    qkv_i, shape = torch.stack([q_i, k_i, v_i], dim=1), (16, 3, 72)
    name = prefix + ".qkv_proj.bias"
    sbs_name = "qkv_ein_b"
    add_data(name, qkv_i, shape, sbs_name, i)

  # Handle the image embedding kernel transpose.
  name = "vision_tower.vision_model.embeddings.patch_embedding.weight"
  assert params[name].shape == (
      1152,
      3,
      14,
      14,
  )
  params[name] = params[name].permute(0, 2, 3, 1)

  # Add the non-layer params.
  for name, shape, sbs_name in (
      layer_config["llm-non-layers"] + layer_config["img-non-layers"]
  ):
    add_data(name, params.pop(name), shape, sbs_name)

  # Go through the LLM layers and add the weights.
  for i in range(dims["num_llm_layers"]):
    add_att_einsum(i)
    add_gating_einsum(i)
    for name, shape, sbs_name in layer_config["llm-layers"]:
      add_data(name, params.pop(name % i), shape, sbs_name, i)
    add_qkv_einsum(i)

  # Go through the Vit layers and add the weights.
  for i in range(27):
    for name, shape, sbs_name in layer_config["img-layers"]:
      add_data(name, params.pop(name % i), shape, sbs_name, i)
    add_vit_qkv_einsum(i)

  assert not params, "Some params were not used: %s" % params.keys()

  # Write everything to the sbs file.
  assert model_specifier.startswith("paligemma")
  sbs_config = configs.ModelConfig(model_specifier)
  writer.write(sbs_config, tokenizer_file)

  # Write the metadata for manual inspection.
  with open(csv_file, "w") as csv_handle:
    csv.writer(csv_handle).writerows(metadata)


_MODEL_SPECIFIER = flags.DEFINE_string(
    "model_specifier",
    None,
    "String specifying model, size, weight, wrapping (ModelConfig.Specifier)",
    required=True,
)

_LOAD_PATH = flags.DEFINE_string(
    "load_path",
    None,
    "Path to the safetensors index.json file to read",
    required=True,
)
_TOKENIZER_FILE = flags.DEFINE_string(
    "tokenizer_file",
    "/tmp/tokenizer.spm",
    "Path to the tokenizer file to read and embed",
)
_METADATA_FILE = flags.DEFINE_string(
    "metadata_file",
    "/tmp/gemmacpp.csv",
    "Path to the metadata file to write",
)
_SBS_FILE = flags.DEFINE_string(
    "sbs_file",
    "/tmp/gemmacpp.sbs",
    "Path to the sbs file to write",
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  logging.use_python_logging()
  logging.set_verbosity(logging.INFO)
  model_specifier = _MODEL_SPECIFIER.value
  load_path = _LOAD_PATH.value
  tokenizer_file = _TOKENIZER_FILE.value
  metadata_file = _METADATA_FILE.value
  sbs_file = _SBS_FILE.value

  logging.info(
      "\n====\nReading %s from %s and %s, writing to %s\n====",
      model_specifier,
      load_path,
      tokenizer_file,
      sbs_file,
  )
  export_paligemma_sbs(
      model_specifier, load_path, tokenizer_file, metadata_file, sbs_file
  )


if __name__ == "__main__":
  app.run(main)
