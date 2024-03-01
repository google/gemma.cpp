# WIP - DO NOT MERGE

import torch
from gemma import config
from gemma import model as gemma_model
import numpy as np


def param_names():
    """Return parameter names in the order they are expected for deserialization."""
    names = ["embedder.weight", "model.norm.weight"]
    # note *weight_scaler params are ignored in the forward computation unless quantization is being used.
    # since we are working with the full precision weights as input, don't include these in the parameters being iterated over
    layer_params = [
        "self_attn.qkv_proj.weight",  # attn_vec_einsum_w
        "self_attn.o_proj.weight",  # qkv_einsum_w
        "mlp.gate_proj.weight",  # qkv_einsum_w
        "mlp.up_proj.weight",  # gating_einsum_w
        "mlp.down_proj.weight",  # linear_w
        "input_layernorm.weight",  # pre_attention_norm_scale
        "post_attention_layernorm.weight",  # pre_ffw_norm_scale
    ]
    for layer in range(18):
        for layer_param in layer_params:
            names = names + ["model.layers." + str(layer) + "." + layer_param]
    return names


def convert_weights():
    # TODO(austinvhuang): move code in here
    pass


if __name__ == "__main__":
    # TODO(austinvhuang): parameterize paths
    output_file = "2bit-f32.sbs"
    model_config = config.get_model_config("2b")
    model_config.dtype = "float32"
    model_config.quant = "store_true"
    model_config.tokenizer = "models/tokenizer.spm"
    device = torch.device("cpu")
    torch.set_default_dtype(torch.float)
    model = gemma_model.GemmaForCausalLM(model_config)
    model.load_weights("models/gemma-2b-it.ckpt")
    model_dict = dict(model.named_parameters())
    param_order = param_names()
    print("Writing parameters ...")
    with open(output_file, "wb") as bin_handle:
        for name in param_order:
            arr = model_dict[name].detach().numpy()
            # TODO(austinvhuang): reshapes
            print(f"  {name : <60}{str(arr.shape)}")
            arr.flatten().astype(np.float32).tofile(bin_handle)

    print("Done")
