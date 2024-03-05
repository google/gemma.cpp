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

# WIP - DO NOT MERGE
# Requires torch 2.2 and gemma package from https://github.com/google/gemma_pytorch

from collections import defaultdict
import torch
from gemma import config
from gemma import model as gemma_model
import numpy as np

def expand_qkv(qkv_proj: np.array) -> np.array:
    """This won't be needed anymore when MQA is implemented"""
    ## this will only be true for 2b
    assert qkv_proj.shape == (2560, 2048)
    qkv = qkv_proj.reshape((10, 256, 2048))

    ## based on line 230 of 
    ## https://github.com/google/gemma_pytorch/blob/main/gemma/model.py
    q_proj = qkv[:8].reshape((1,8,256,2048))
    kv_proj = qkv[8:]
    kv_proj = kv_proj[:, np.newaxis, :, :]
    kv_proj = np.repeat(kv_proj, 8, axis=1)

    qkv = np.concatenate([q_proj, kv_proj])
    qkv = np.transpose(qkv, axes=[1,0,2,3])
    return qkv

TRANSFORMATIONS = defaultdict(
    lambda: lambda x: x,
    {
        ## padding goes at end per discussion
        "embedder.weight": lambda x: np.concatenate([x, np.zeros([128, 2048])], 0),
        "self_attn.qkv_proj.weight": expand_qkv,
      
        ## based on line 234 of 
        ## https://github.com/google/gemma_pytorch/blob/main/gemma/model.py
        "self_attn.o_proj.weight": lambda x: x.reshape(2048, 8, 256).transpose([1,0,2]), # TODO: which of the 2048 is unpacked to 8 x 256, and which is model_dim?
        "mlp.gate_proj.weight": lambda x: x[np.newaxis, :, :],
        "mlp.up_proj.weight": lambda x: x[np.newaxis, :, :],
        "mlp.down_proj.weight": lambda x: x,
    },
)

VALIDATIONS = {
    "embedder.weight": lambda x: x.shape == (256128, 2048),
    "model.norm.weight": lambda x: x.shape == (2048,),
    "self_attn.qkv_proj.weight": lambda x: x.shape == (8, 3, 256, 2048),
    "self_attn.o_proj.weight": lambda x: x.shape == (8, 2048, 256),
    "mlp.gate_proj.weight": lambda x: x.shape == (1, 16384, 2048),
    "mlp.up_proj.weight": lambda x: x.shape == (1, 16384, 2048),
    "mlp.down_proj.weight": lambda x: x.shape == (2048, 16384),
    "input_layernorm.weight": lambda x: x.shape == (2048,),
    "post_attention_layernorm.weight": lambda x: x.shape == (2048,),
}


def param_names():
    """Return parameter names in the order they are expected for deserialization."""

    # note *weight_scaler params are ignored in the forward computation unless
    # quantization is being used.
    #
    # since we are working with the full precision weights as input, don't
    # include these in the parameters being iterated over.

    # fmt: off
    names = [
        ("embedder.weight", ) * 2,                  # embedder_input_embedding (vocab=256000, model_dim=2048) -> (vocab=256128, model_dim=2048)
        ("model.norm.weight", ) * 2                 # final_norm_scale         (model_dim=2048)
    ]
    layer_params = [
                                            # TODO(austinvhuang): transpositions here ...
        "self_attn.o_proj.weight",          # attn_vec_einsum_w        (2048, 2048) -> (heads=8, model_dim=2048, qkv_dim=256)
                                            # # ( q_heads = 8 + kv = 2 ) x qkv_dim =  2560
        "self_attn.qkv_proj.weight",        # qkv_einsum_w             (2560, 2048) -> (heads=8, qkv=3, qkv_dim=256, model_dim=2048)
        # these are the same without any change
        "mlp.gate_proj.weight",             # gating_einsum_w          (16384, 2048) => (gate/up=2, hidden=16384, model_dim=2048)
        "mlp.up_proj.weight",  
        "mlp.down_proj.weight",             # linear_w                 (2048, 16384) => (model_dim=2048, hidden=16384)
        "input_layernorm.weight",           # pre_attention_norm_scale (model_dim=2048)
        "post_attention_layernorm.weight",  # pre_ffw_norm_scale       (model_dim=2048)
    ]
    # fmt: on
    for layer in range(18):
        for layer_param in layer_params:
            names = names + [(f"model.layers.{layer}.{layer_param}", layer_param)]
    print("names:", names)
    return names


def convert_weights():
    # TODO: parameterize paths as CLI args instead of hard coding them
    output_file = "2bit-f32.sbs"
    model_config = config.get_model_config("2b")
    model_config.dtype = "float32"

    ## this turns on int8 quantization
    # model_config.quant = "store_true"
    model_config.tokenizer = "models/tokenizer.spm"
    device = torch.device("cpu")
    torch.set_default_dtype(torch.float)
    model = gemma_model.GemmaForCausalLM(model_config)
    model.load_weights("models/gemma-2b-it.ckpt")
    model_dict = dict(model.named_parameters())

    for layer_name in model_dict:
      ## Make sure we're not silently having int8 quantization turned on.
      print(layer_name, model_dict[layer_name].max())
      assert(model_dict[layer_name].max() > 0.0)
  
    param_order = param_names()

    all_ok = True
    print("Checking transformations ...")
    for name, layer_name in param_order:
        arr = model_dict[name].detach().numpy()
        arr = TRANSFORMATIONS[layer_name](arr)
        check = "OK" if VALIDATIONS[layer_name](arr) else "FAILED"

        if check == "FAILED":
          all_ok = False
      
        print(f"  {name : <60}{str(arr.shape) : <20}{check}")

    if all_ok:
      print("Writing parameters ...")
      gate = None
      with open(output_file, "wb") as bin_handle:
          for name, layer_name in param_order:
              arr = model_dict[name].detach().numpy()
              arr = TRANSFORMATIONS[layer_name](arr)
              check = "OK" if VALIDATIONS[layer_name](arr) else "FAILED"
              print(f"  {name : <60}{str(arr.shape) : <20}{check}")

              if "gate_proj" in name:
                gate = arr
              elif "up_proj" in name:
                up = arr
                f = np.concatenate([gate, up])
                print (f.shape)
                f.flatten().astype(np.float32).tofile(bin_handle)
              else:
                arr.flatten().astype(np.float32).tofile(bin_handle)


if __name__ == "__main__":
    convert_weights()
    print("Done")
