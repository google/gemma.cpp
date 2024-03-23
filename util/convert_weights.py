# Copyright 2024 Google LLC
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


from collections import defaultdict
import torch
from gemma import config
from gemma import model as gemma_model
import numpy as np
import argparse
import os

# Requires torch 2.2 and gemma package from https://github.com/google/gemma_pytorch

def check_file_exists(value):
  if not os.path.exists(str(value)):
    raise argparse.ArgumentTypeError("The file %s does not appear to exist." % value)
  return value
    

def check_model_types(value):
  if str(value).lower() not in ["2b", "7b"]:
    raise argparse.ArgumentTypeError("Model type value %s is not in [2b, 7b]." % value)
  return value
    

parser = argparse.ArgumentParser()
parser.add_argument(
    "--tokenizer",
    dest="tokenizer",
    default="models/tokenizer.spm",
    help="Location of tokenizer file (.model or .spm)",
    type=check_file_exists,
)

parser.add_argument(
    "--weights",
    dest="weights",
    default="models/gemma-2b-it.ckpt",
    help="Location of input checkpoint file (.ckpt)",
    type=check_file_exists,
)

parser.add_argument(
    "--output_file",
    dest="output_file",
    default="2bit-f32.sbs",
    help="Location to write converted weights",
    type=str,
)

parser.add_argument(
    "--model_type",
    dest="model_type",
    default="2b",
    help="Model size / type (2b, 7b)",
    type=check_model_types,
)

args = parser.parse_args()


TRANSFORMATIONS = {
  "2b":defaultdict(
    lambda: lambda x: x,
    {
        "embedder.weight": lambda x: x,
        "self_attn.qkv_proj.weight": lambda x: x.reshape((10, 256, 2048)),
        "self_attn.o_proj.weight": lambda x: x.reshape((2048, 8, 256)).transpose([1,0,2]),
        "mlp.gate_proj.weight": lambda x: x[np.newaxis, :, :],
        "mlp.up_proj.weight": lambda x: x[np.newaxis, :, :],
        "mlp.down_proj.weight": lambda x: x,
    }
  ),
  "7b":defaultdict(
    lambda: lambda x: x,
    {
        "embedder.weight": lambda x: x,
        "self_attn.qkv_proj.weight": lambda x: x.reshape((3, 16, 256, 3072)).transpose([1,0,2,3]),
        "self_attn.o_proj.weight": lambda x: x.reshape((3072, 16, 256)).transpose([1,0,2]),
        "mlp.gate_proj.weight": lambda x: x[np.newaxis, :, :],
        "mlp.up_proj.weight": lambda x: x[np.newaxis, :, :],
        "mlp.down_proj.weight": lambda x: x,
    }
  ),
}

VALIDATIONS = {
  "2b": {
    "embedder.weight": lambda x: x.shape == (256000, 2048),
    "model.norm.weight": lambda x: x.shape == (2048,),
    "self_attn.qkv_proj.weight": lambda x: x.shape == (10, 256, 2048),
    "self_attn.o_proj.weight": lambda x: x.shape == (8, 2048, 256),
    "mlp.gate_proj.weight": lambda x: x.shape == (1, 16384, 2048),
    "mlp.up_proj.weight": lambda x: x.shape == (1, 16384, 2048),
    "mlp.down_proj.weight": lambda x: x.shape == (2048, 16384),
    "input_layernorm.weight": lambda x: x.shape == (2048,),
    "post_attention_layernorm.weight": lambda x: x.shape == (2048,),
  },
  "7b": {
    "embedder.weight": lambda x: x.shape == (256000, 3072),
    "model.norm.weight": lambda x: x.shape == (3072,),
    "self_attn.qkv_proj.weight": lambda x: x.shape == (16, 3, 256, 3072),
    "self_attn.o_proj.weight": lambda x: x.shape == (16, 3072, 256),
    "mlp.gate_proj.weight": lambda x: x.shape == (1, 24576, 3072),
    "mlp.up_proj.weight": lambda x: x.shape == (1, 24576, 3072),
    "mlp.down_proj.weight": lambda x: x.shape == (3072, 24576),
    "input_layernorm.weight": lambda x: x.shape == (3072,),
    "post_attention_layernorm.weight": lambda x: x.shape == (3072,),
  },
}


def param_names(num_hidden_layers: int):
    """Return parameter names in the order they are expected for deserialization."""

    # note *weight_scaler params are ignored in the forward computation unless
    # quantization is being used.
    #
    # since we are working with the full precision weights as input, don't
    # include these in the parameters being iterated over.

    # fmt: off
    names = [
        ("embedder.weight", ) * 2,          # embedder_input_embedding
        ("model.norm.weight", ) * 2         # final_norm_scale
    ]
    layer_params = [
        "self_attn.o_proj.weight",          # attn_vec_einsum_w
        "self_attn.qkv_proj.weight",        # qkv_einsum_w
        "mlp.gate_proj.weight",             # gating_einsum_w
        "mlp.up_proj.weight",  
        "mlp.down_proj.weight",             # linear_w
        "input_layernorm.weight",           # pre_attention_norm_scale
        "post_attention_layernorm.weight",  # pre_ffw_norm_scale
    ]
    # fmt: on
    for layer in range(num_hidden_layers):
        for layer_param in layer_params:
            names = names + [(f"model.layers.{layer}.{layer_param}", layer_param)]
    return names


def convert_weights():
    model_type = args.model_type
    output_file = args.output_file
  
    model_config = config.get_model_config(model_type)
    model_config.dtype = "float32"
    model_config.tokenizer = args.tokenizer
    device = torch.device("cpu")
    torch.set_default_dtype(torch.float)
    model = gemma_model.GemmaForCausalLM(model_config)
  
    model.load_weights(args.weights)
    model.to(device).eval()
  
    model_dict = dict(model.named_parameters())  
    param_order = param_names(model_config.num_hidden_layers)

    all_ok = True
    print("Checking transformations ...")
    for name, layer_name in param_order:
        arr = model_dict[name].detach().numpy()
        arr = TRANSFORMATIONS[model_type][layer_name](arr)
        check = "OK" if VALIDATIONS[model_type][layer_name](arr) else "FAILED"

        if check == "FAILED":
          all_ok = False
          print(f"  {name : <60}{str(arr.shape) : <20}{check}")

    if all_ok:
      print("Writing parameters ...")
      gate = None
      with open(output_file, "wb") as bin_handle:
          for name, layer_name in param_order:
              arr = model_dict[name].detach().numpy()
              arr = TRANSFORMATIONS[model_type][layer_name](arr)
              check = "OK" if VALIDATIONS[model_type][layer_name](arr) else "FAILED"
              print(f"  {name : <60}{str(arr.shape) : <20}{check}")
              arr.flatten().astype(np.float32).tofile(bin_handle)


if __name__ == "__main__":
    convert_weights()
    print("Done")
