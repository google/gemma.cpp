"""Transformations for python-trees representing the parameters of a ML model.

Important: This module assumes that byte-order is the same on the
machine that serializes data and the machine that deserializes
data. If, for example, numpy-data gets dumped, respectively loaded,
with a dtype-specification of numpy.float32, on-file byte-order
will be host byte order.

"""

import ast
import hashlib
import itertools
import pprint
import sys
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, TypeVar

import numpy
import pytree_transforms


NT = TypeVar('NT')


def ml_model_leaf_summary(path, x, sep=', '):
  """Produces a textual summary of a leaf-node and its path.

  Args:
    path: The path-to-root, as a reverse-order recursive
      pair of path-components, with `()` as root.
    x: The leaf-object.
    sep: the separator between description-elements.
      Default ', ' allows for convenient line-by-line processing
      (such as via grep, perl -ne, etc.), but using e.g. sep=',\n  '
      might be more useful for human consumption.

  Returns:
    A human-readable string providing information about the node.
  """
  # Using `repr` for path-components to get a faithful presentation.
  # (...which still however would be somewat painful to correctly
  # split into components.)
  path_str = ','.join(map(repr,
                          pytree_transforms.linearize_revtuple_path(path)))
  tx = type(x)
  mod = tx.__module__  # Either a module or a string like 'builtins'.
  modname = mod if isinstance(mod, str) else mod.__name__
  type_str = f'{modname}.{tx.__qualname__}'
  try:
    # `numpy.ndarray` instances have a `.data` property that gives access
    # to a buffer via which we can hashlib-fingerprint the numerical
    # contents. We here simply try to produce a fingerprint and also look
    # up the .dtype of the object. Technically, there is a somewhat-unsound
    # assumption here that if these operations succeed, we are indeed looking
    # at a ndarray or sufficiently similar object for these operations to
    # make sense. As the output is declared "for human consumption", this
    # fishiness is not a problem.
    fp = hashlib.sha256(x.data).hexdigest()
    start = list(itertools.islice(x.flat, 5))
    stats_str = (
        f'min={numpy.min(x):.6g}, max={numpy.max(x):.6g}, '
        f'mean={numpy.mean(x):.6g}, std={numpy.std(x):.6g}')
    return (f'{path_str:60s}: <{type_str}{sep}'
            f'fp=0x{fp}{sep}{stats_str}{sep}shape={x.shape}, '
            f'dtype={x.dtype}{sep}start={start}>')
  except (AttributeError, ValueError, TypeError):
    # Fallback - trying to include information about the data-content
    # of a likely-numerical-array failed.
    return f'{path_str:60s}: {type_str}({repr(x)})'


# A specialized node-handler.
# Interface follows node-handler expectations defined in pytree_transforms.
def _ml_model_tree_node_handler(path: tuple, node : NT) -> (
    None | tuple[Callable[[Iterable[tuple[Any, NT]]], NT],
                 Iterator[tuple[Any, NT]]]):
  """Processes a tree-node as required by pytree-iteration and -mapping.

  Args:
    path: revtuple path to the current node.
    node: a tree-node in a ML-model tree that is recursively
      built out of `numpy.ndarray` leaf-values and dicts mapping
      node-name string-keys to other such nodes representing subtrees -
      and nothing else.

  Returns:
    `None` if the tree-node is to be regarded as a leaf, otherwise
    a pair `(rebuilder, iterator)`, where `iterator` iterates
    over the data-content of the node, each item represented as a pair
    of `(lookup_path_item, value_item)`, and `rebuilder` is a function
    which, when applied to `iterator` or any iterable with the same
    elements, returns a node that is equivalent to the original.

  Raises:
    NotAMLModelTreeNodeError: If the tree contains a node that is neither
      a `dict` nor a `numpy.ndarray` instance.
  """
  # The astute reader will notice that we are doing something fishy
  # here - this code could not be translated to Haskell as-is, since
  # `NT` cannot actually be a proper type-variable in the sense
  # of parametric polymorphism.
  del path  # Unused.
  if isinstance(node, dict):
    return dict, iter(node.items())
  if isinstance(node, numpy.ndarray):
    return None
  raise pytree_transforms.NotAMLModelTreeNodeError(
      f'Type of bad node: {type(node)}')


def _ml_model_extract_leaf_transform(
        path: pytree_transforms.RevTuplePath,
        leaf: Any):
  """Maps an array-leaf to a pair `(full_path, lambda: array)`.

  The computation that produces the leaf-value is lazified underneath
  a `lambda`, since if we e.g. performed a memory-expensive
  transformation (such as some dtype-changes) directly at this point,
  then going from an iterator over tree-items for one-by-one
  consumption to a list of these items would have all the
  dtype-transformed values around simultaneously. We want to avoid
  situations where we can do nothing about having multiple variants
  of the data simultaneously in memory.
  """
  # Hack: If we are encountering a `bfloat16` numpy-array,
  # we pretend to have the data as a numpy.float32 array,
  # since that's about all that contemporary CPUs can process
  # efficiently here.
  linearized_path = pytree_transforms.linearize_revtuple_path(path)
  try:
    # We have to use some trickery to detect `bfloat16`.
    if leaf.dtype.descr[-1] == ('', '<V2'):
      return linearized_path, lambda: leaf.astype(numpy.float32)
    else:
      return linearized_path, lambda: leaf
  except Exception:
    return linearized_path, lambda: leaf


# Here, we cannot properly specify the return-type, since this can
# either be a leaf-type or something recursively-defined.
def revtuple_autovifify_from_linear(
        keys_and_vals: Iterable[tuple[Any, Any]]) -> Any:
  """Performs perl-style autovivification on a nested-dict tree.

  Args:
    keys_and_vals: An iterable of pairs `(key_path, value)`, where
      `key_path` is a sequence of keys to be used to navigate to
      the result via iterative dict-lookup, left-to-right.
      Must not have duplicate keys, and must not more than one key if
      an empty-sequence key is present. If this iterable is an
      iterator, it will be fully exhausted on successful execution.

  Returns:
    An object representing a nested-dict structure such that
    for every `key_path` from `keys_and_vals`, recursive-dict-lookup
    on the elements of that path starting from this object will
    produce the corresponding value. An empty `keys_and_vals`
    set will return `{}`. Every dict in the nested return-value
    that has been populated by autovivification is newly allocated.
  """
  # Code structure is a bit gnarly here due to f(keys_and_vals=[((), x)])
  # having to evaluate to x and not a dict.
  # There may be ways to prettify/simplify this.
  result = None
  empty = {}
  for linear_path, val in keys_and_vals:
    if linear_path == ():
      if result is not None:
        raise ValueError('Root-value seen alongside other values.')
      result = val
    else:
      if result is None:
        result = {}
      elif type(result) is not dict:
        # We already did encounter a root-value.
        raise ValueError('Root-value seen alongside other values.')
      cursor = result
      for n in range(len(linear_path) - 1):
        cursor = cursor.setdefault(linear_path[n], empty)
        if cursor is empty:
          # Regenerate `empty` if we just used it up.
          empty = {}
      cursor[linear_path[-1]] = val
  return {} if result is None else result


def model_overview(tree, out=None) -> None:
  """Prints a human-readable overview to `(out or sys.stdout)`."""
  actual_out = out or sys.stdout
  for line in pytree_transforms.pytree_leaf_iter(
      tree, ml_model_leaf_summary,
      _ml_model_tree_node_handler):
    print(line, file=actual_out)


def model_contents(tree) -> Mapping[tuple[str, ...], Any]:
  """Maps a model to a {pytree_keys: data_array} mapping.

  Args:
    tree: The ML-model parameter-tree, built recursively out of
      dict-instances with numpy.ndarray instances as leaves.

  Returns:
    A mapping from linearized pytree-key-sequence tuple to the corresponding
    leaf-value.
  """
  def leaf_transform(revtuple_path, leaf):
    return pytree_transforms.linearize_revtuple_path(revtuple_path), leaf
  return dict(
      pytree_transforms.pytree_leaf_iter(
          tree, leaf_transform, _ml_model_tree_node_handler))


def _fn_identity(x): return x


def model_save(tree,
               filepath_stem: str,
               data_suffix: str = '.data',
               manifest_suffix: str | None = '.manifest',
               key: Callable[[tuple[str, ...]], Any] | None = None,
               array_transform_by_pytree_key: (
                   Mapping[tuple[str, ...],
                           Callable[[numpy.ndarray], numpy.ndarray]] |
                   None) = None,
               report: Callable[[str], None] | None = None,
               byte_align: int = 8) -> tuple[int, float]:
  """Saves the content of a ML-model parameter-tree to filesystem.

  After successful execution, the file f"{filepath_stem}.data"
  will hold the combined numerical model-parameters, and
  f"{filepath_stem}.manifest" will contain the key for interpreting
  (and rebuilding) the data.

  Args:
    tree: The ML-model parameter-tree, built recursively out of
      dict-instances with numpy.ndarray instances as leaves.
    filepath_stem: Filesystem location for data.
    data_suffix: Suffix to use for the data file.
    manifest_suffix: Either `None`, in which case no manifest-file
      will get written, or the suffix for the manifest-file.
    key: `None` or a key-function that will be applied to the linear model-path
      and used for sorting the data arrays by increasing value of the
      key-function. If the key-function returns `None` on an item,
      then this item is not included.
    array_transform_by_pytree_key: Optional mapping from pytree-key
      to an array-to-array transformation function to apply to the array
      prior to serialization.
    report: Optional callable for logging progress-reports.
    byte_align: byte-alignment to use for numerical array data.
      Numerical arrays whose size in bytes is not a multiple of this
      will get padded to the next full multiple.

  Returns:
    A pair of `(size, time_sec)`, where `size` is the total byte-size
    of the `.data` file and `time_sec` is the elapsed time
    for saving the model, in seconds.
  """
  time0 = time.monotonic()
  if array_transform_by_pytree_key is None:
    array_transform_by_pytree_key = {}
  model_lazy_items = (
      pytree_transforms.pytree_leaf_iter(
          tree, _ml_model_extract_leaf_transform,
          _ml_model_tree_node_handler))
  if key is not None:
    to_write = [
        nkv[1:] for nkv in sorted(
            (nkv for nkv in ((key(path), path, v)
                             for path, v in model_lazy_items)
             if nkv[0] is not None), key=lambda nkv: nkv[0])]
  else:
    to_write = list(model_lazy_items)
  #
  def lazy_arr_path_shape_dtype_size(path_and_lazy_arr):
    path, lazy_arr = path_and_lazy_arr
    arr = array_transform_by_pytree_key.get(path, _fn_identity)(lazy_arr())
    return path, arr.shape, arr.dtype, arr.data.nbytes
  arrs_path_shape_dtype_nbytes = list(
      map(lazy_arr_path_shape_dtype_size, to_write))
  # We need to know the total size of all the data.
  bytesizes = [nbytes for *_, nbytes in arrs_path_shape_dtype_nbytes]
  padded_bytesizes = [-(-bytesize // byte_align * byte_align)
                      for bytesize in bytesizes]
  offsets = numpy.cumsum([0] + padded_bytesizes)
  membuf = numpy.memmap(filepath_stem + data_suffix,
                        mode='w+', shape=offsets[-1])
  try:
    for (path, shape, dtype, nbytes), offset, (_, lazy_arr) in zip(
        arrs_path_shape_dtype_nbytes, offsets, to_write):
      # Note that if getting the array from the lazy lambda involved some
      # computation, such as a copying dtype-change, that computation would
      # end up being done multiple times here - including once above, to compute
      # byte-sizes, and once more here.
      transformed_arr = array_transform_by_pytree_key.get(
          path,
          _fn_identity)(lazy_arr())
      membuf[offset : offset + nbytes] = numpy.frombuffer(
          transformed_arr.ravel().data, 'u1')
      if report is not None:
        samples = ', '.join(map(str, transformed_arr.ravel()[:5]))
        report(f'# Adding: {path!r}\n  bytes: {nbytes:10d}, '
               f'shape: {shape!r:30},\n  start: [{samples}, ...]')
      transformed_arr = None  # Drop memory references to numerical arrays ASAP.
  finally:
    if membuf is not None:
      membuf.flush()
      # NumPy wart: the memory-buffer is a resource that conceptually
      # should be .close()able - since mmap()ing holds on to a
      # file descriptor. However, it looks as if that clean-up were done
      # in the "finalizer", despite that having meanwhile been widely
      # understood as dubious practice. So, the best we can do here is
      # to explicitly and clearly remove our reference to the instance.
      del membuf
  if manifest_suffix is not None:
    # We still have to serialize the data that allows us to reconstruct
    # a tree that is equivalent to the original.
    manifest_data = [
        dict(path=path,
             dtype=dtype.descr[-1][-1],
             shape=shape,
             nbytes=nbytes,
             offset=offset)
        for (path, shape, dtype, nbytes), offset in zip(
            arrs_path_shape_dtype_nbytes, offsets)]
    with open(filepath_stem + '.manifest', 'wt') as h_manifest:
      pprint.pprint(manifest_data, stream=h_manifest)
  time_taken = time.monotonic() - time0
  return offsets[-1], time_taken


def model_load(filepath_stem, mmapped=True):
  """Loads a model saved by `model_save`.

  Tries to load the model from f"{filepath_stem}.data"
  and f"{filepath_stem}.manifest".

  Args:
    filepath_stem: The model location on the filesystem.
    mmapped: Whether data-arrays will be slices of a
      `numpy.memmap` mapped buffer, to be paged in
      on demand only, or in-memory copies of the data.
  Returns:
    A dict/numpy.ndarray tree representation of the model,
    equivalent to the original model.
  """
  with open(filepath_stem + '.manifest', 'rt') as h_manifest:
    manifest = ast.literal_eval(h_manifest.read())
  membuf = numpy.memmap(filepath_stem + '.data', mode='r+')
  paths_and_arrays = []
  for item in manifest:
    path = item['path']
    dtype = numpy.dtype(item['dtype'])
    shape = item['shape']
    nbytes = item['nbytes']
    offset = item['offset']
    data_array = numpy.frombuffer(membuf[offset : offset + nbytes].data,
                                  dtype=dtype).reshape(shape)
    paths_and_arrays.append(
        (path,
         data_array if mmapped else data_array.copy()))
  # At this point, the memory-buffer is no longer needed. Still, if
  # data-arrays retain references to the underlying data
  # (i.e. when mmapped=False), this should keep the mapping
  # - and hence file descriptor - open. We then are in a somewhat
  # undesirable situation of clean-up of a resource that happens in a
  # hard-to-predict way releasing a file descriptor.
  del membuf
  return revtuple_autovifify_from_linear(paths_and_arrays)
