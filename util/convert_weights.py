# WIP - DO NOT MERGE

from collections import defaultdict
import torch
from gemma import config
from gemma import model as gemma_model
import numpy as np

TRANSFORMATIONS = defaultdict(lambda: lambda x: x, {
    "embedder.weight": lambda x: np.concatenate([np.zeros([128, 2048]), x], 0), 
    "self_attn.qkv_proj.weight": lambda x: x,
    "mlp.up_proj.weight" : lambda x: x,
    "mlp.down_proj.weight" : lambda x: x,
})


def param_names():
    """Return parameter names in the order they are expected for deserialization."""

    # note *weight_scaler params are ignored in the forward computation unless
    # quantization is being used.
    #
    # since we are working with the full precision weights as input, don't
    # include these in the parameters being iterated over.

    # fmt: off
    names = [
        "embedder.weight",                  # embedder_input_embedding (vocab=256000, model_dim=2048) -> (vocab=256128, model_dim=2048)
        "model.norm.weight"                 # final_norm_scale         (model_dim=2048)
    ]
    layer_params = [
                                            # TODO(austinvhuang): transpositions here ...
        "self_attn.qkv_proj.weight",        # attn_vec_einsum_w        (2560, 2048) -> (heads=8, model_dim=2048, qkv_dim=256)
        "self_attn.o_proj.weight",          # qkv_einsum_w             (2048, 2048) -> (heads=8, qkv=3, qkv_dim=256, model_dim=2048)

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
            names = names + [f"model.layers.{layer}.{layer_param}"]
    return names


def convert_weights():
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
            arr = TRANSFORMATIONS[name](arr)
            # TODO(austinvhuang): reshapes
            print(f"  {name : <60}{str(arr.shape)}")
            arr.flatten().astype(np.float32).tofile(bin_handle)


if __name__ == "__main__":
    convert_weights()
    print("Done")
