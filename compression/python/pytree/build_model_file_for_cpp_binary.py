"""Ad-hoc glue code for building the griffin model-file for the C++ binary.

Usage:

python3 -m venv $HOME/clients/griffin-venv

. $HOME/clients/griffin-venv/bin/activate

python3 -m pip install -r requirements.txt

time python3 build_model_file_for_cpp_binary.py \
  $HOME/GRIFFIN/model_data \
  cpp_load_log.txt /tmp/G2B.data

real    3m5.821s
user    2m9.205s
sys     2m46.720s

./compress_weights --weights /tmp/G2B.data --model gr2b-it \
  --compressed_weights /tmp/G2B.compressed
./gemma --tokenizer tokenizer.spm --weights /tmp/G2B.compressed \
  --model gr2b-it

Weights for the recurrent-gemma model that can be converted with this script
can be found at:

  https://www.kaggle.com/models/google/recurrentgemma/flax/2b-it
"""

import pprint
import re
import sys

from typing import Any, Mapping

import numpy

import orbax.checkpoint

import ml_model_transforms
import pytree_transforms


def _fn_identity(x): return x


def _fn_transpose(x): return x.T


def _fn_transpose_all_heads(x): return x.transpose(0, 2, 1)


def _fn_scaled_softplus(a):
  return -8 * numpy.logaddexp(a, 0)


def _fn_attention_moveaxis(a):
  return a.reshape(10, 256, 2560).transpose(0, 2, 1)


def _aspec(pieces=(), transforms=()):
  """Short-hand array-save-specification.

  Args:
    pieces: Sequence of key-sequences identifying an array.
    transforms: Sequence of transformations, indexed in
      parallel to `pieces`, to apply to data arrays prior to saving.
      Will be padded with identity-transformations to the length of `pieces`.

  Returns:
    Specification as for use in _LAYETR_NAME_MAPPING.
  """
  # `zip` trims to shortest sequence, so this amounts to using
  # default-transforms.
  # tuple() since we need a Sequence here, not a stateful-iterator zip_object.
  return tuple(zip(pieces, list(transforms) + [_fn_identity] * len(pieces)))


_LAYER_NAME_MAPPING = pytree_transforms.deep_freeze({
    # Recurrent Layer
    'griffin_linear_x_w': _aspec(
        [('recurrent_block', 'linear_x', 'kernel')],
        [_fn_transpose]),
    'griffin_linear_x_biases': _aspec(
        [('recurrent_block', 'linear_x', 'bias')]),
    'griffin_linear_y_w': _aspec(
        [('recurrent_block', 'linear_y', 'kernel')],
        [_fn_transpose]),
    'griffin_linear_y_biases': _aspec(
        [('recurrent_block', 'linear_y', 'bias')]),
    'griffin_linear_out_w': _aspec(
        [('recurrent_block', 'linear_out', 'kernel')],
        [_fn_transpose]),
    'griffin_linear_out_biases': _aspec(
        [('recurrent_block' ,'linear_out', 'bias')]),
    'griffin_conv_w': _aspec(
        [('recurrent_block', 'conv_1d', 'w')]),
    'griffin_conv_biases': _aspec(
        [('recurrent_block', 'conv_1d', 'b')]),
    'griffin_gate_w': _aspec(
        [('recurrent_block', 'rg_lru', 'input_gate', 'w'),
         ('recurrent_block', 'rg_lru', 'a_gate', 'w')],
        [_fn_transpose_all_heads, _fn_transpose_all_heads]),
    'griffin_gate_biases': _aspec(
        [('recurrent_block', 'rg_lru', 'input_gate', 'b'),
         ('recurrent_block', 'rg_lru', 'a_gate', 'b')]),
    'griffin_a': _aspec(
        [('recurrent_block', 'rg_lru', 'a_param')],
        [_fn_scaled_softplus]),
    # Attention Layer
    'qkv_einsum_w': _aspec(
        [('attention_block', 'proj_q', 'kernel'),
         ('attention_block', 'proj_k', 'kernel'),
         ('attention_block', 'proj_v', 'kernel'),
         ],
        [_fn_transpose, _fn_transpose, _fn_transpose]),
    'attn_vec_einsum_w': _aspec(
        [('attention_block', 'proj_final', 'kernel')],
        [_fn_attention_moveaxis]),
    'attention_output_biases': _aspec(
        [('attention_block', 'proj_final', 'bias')]),
    # Common
    'pre_attention_norm_scale': _aspec(
        [('temporal_pre_norm', 'scale')]),
    'pre_ffw_norm_scale': _aspec(
        [('channel_pre_norm', 'scale')]),
    'gating_einsum_w': _aspec(
        [('mlp_block', 'ffw_up', 'w')],
        [_fn_transpose_all_heads]),
    'ffw_gating_biases': _aspec(
        [('mlp_block', 'ffw_up', 'b')]),
    'linear_w': _aspec(
        [('mlp_block', 'ffw_down', 'kernel')],
        [_fn_transpose]),
    'ffw_output_biases': _aspec(
        [('mlp_block', 'ffw_down', 'bias')]),
    # Other
    'embedder_input_embedding': _aspec(
        [('embedder', 'input_embedding')]),
    'final_norm_scale': _aspec(
        [('final_norm', 'scale')]),
})


def process_param_line(line : str) -> tuple[None | str, int, str]:
  """Processes a "loading parameters" log-line from the griffin binary."""
  # This is slightly more permissive than strictly needed, to also handle
  # some earlier form of the output.
  matched = re.match(
      r'(?a)Loading Parameters:? \('
      r'(?:layer=(?P<layer>\d+), )?'
      r'size (?P<size>\d+)\):? '
      r'(?P<tag>\S+)',
      line)
  if not matched:
    return None
  layer = matched['layer']
  wanted_size = int(matched['size'])
  cpp_tag = matched['tag']
  return matched['layer'], int(matched['size']), matched['tag']


def collect_pytree_keys(param_lines):
  """Collects all the pytree keys and transforms for model-serialization."""
  pytree_keys = []
  array_transforms = []
  unsatisfied = []
  for maybe_spec in map(process_param_line, param_lines):
    if not maybe_spec: continue  # Skip non-parameter lines.
    layer, wanted_size, cpp_tag = maybe_spec
    pytree_key_tails_and_transforms = _LAYER_NAME_MAPPING.get(cpp_tag, ())
    if not pytree_key_tails_and_transforms:
      unsatisfied.append((layer, cpp_tag))
    else:
      for key_tail, array_transform in pytree_key_tails_and_transforms:
        pytree_keys.append(
            key_tail if layer is None
            else (f'blocks.{layer}',) + key_tail)
        array_transforms.append(array_transform)
  return pytree_keys, array_transforms, unsatisfied


class UnsatisfiedArrayLoadsError(ValueError):
  """Some array-loads could not be satisfied."""


def flatten_model_for_cpp_binary(tree,
                                 cpp_expectations_logfile_path : str,
                                 out_path : str,
                                 unsatisfied_ok : bool = False
                                 ):
  """Produces a model-parameters file readable by the C++ binary.

  Args:
    tree: The pytree with model-parameters.
    cpp_expectations_logfile_path:
      Path to a logfile produced by the C++ binary that shows
      the expected array-order.
    out_path: Path to the model-weights file to be written.
    unsatisfied_ok: If true, we ignore the presence of unsatisfied
      array-loads and write a model-parameters file that skips these pieces.
      This will lead to an unusable model-parameters file which however
      still might be useful for other analysis.

  Returns:
    Tuple `(unknown_keys, missing_keys)`, where `unknown_keys`
    is a sequence of `(layer_or_None, name)` descriptions of the keys
    in the C++ log that could not be satisfied, and `missing_keys`
    is a sequence of linearized pytree key-sequences for keys
    not found in the checkpoint.

  Raises:
    UnsatisfiedArrayLoadsError: If some of the expected arrays
      could not be included in the output and `unsatisfied_ok`
      is false.
  """
  with open(cpp_expectations_logfile_path, 'rt') as h_log:
    pytree_keys, array_transforms, unknown_keys = collect_pytree_keys(
        list(h_log))
  rank_by_pytree_key = {k: n for n, k in enumerate(pytree_keys)}
  array_transform_by_pytree_key = dict(zip(pytree_keys, array_transforms))
  #
  model_contents = ml_model_transforms.model_contents(tree)
  missing_keys = set(pytree_keys) - model_contents.keys()
  if (unknown_keys or missing_keys) and not unsatisfied_ok:
    raise ValueError(
      f'Unsatisfied loads: unknown_keys: {unknown_keys!r}, '
      f'missing keys: {sorted(missing_keys)!r}')
  ml_model_transforms.model_save(
    tree,
    filepath_stem=out_path,
    data_suffix='',
    manifest_suffix=None,
    array_transform_by_pytree_key=array_transform_by_pytree_key,
    key=rank_by_pytree_key.get,
    report=lambda line: print(line, file=sys.stderr),
    byte_align=1)
  return tuple(unknown_keys), tuple(sorted(missing_keys))


def main(args):
  """Creates the model-file.

  Args:
    sys.argv[] parameters from command line sans the leading one.

  Returns:
    The pytree with all the de-serialized variables, such as for convenient
    `python3 -i` inspection.
  """
  try:
    model_dir, cpp_load_log, out_path = args
  except Exception:
    sys.exit(f'Usage: {__file__} [model_dir] [cpp_load_log] [output_filename]')
  pattern = ("recurrent", "recurrent", "attention")
  orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  variables = orbax_checkpointer.restore(model_dir)
  if sorted(variables) == ['params']:
    print('Warning: Using `variables["params"]` as tree-root.', file=sys.stderr)
    variables_to_use = variables['params']
  else:
    variables_to_use = variables
  unknown, missing = flatten_model_for_cpp_binary(variables_to_use,
                                                  cpp_load_log,
                                                  out_path,
                                                  unsatisfied_ok=True)
  print('Model file saved.\n'
        f'# unknown:\n{pprint.pformat(unknown)}\n'
        f'# missing:\n{pprint.pformat(missing)}')
  return variables


if __name__ == '__main__':
  # Return value assignment is for `python3 -i ...` inspection.
  pytree = main(sys.argv[1:])
