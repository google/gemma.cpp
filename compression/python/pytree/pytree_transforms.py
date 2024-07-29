"""Tools for transforming "nested python object" tree data structures.

# Context

The motivation for this module came from ML applications that ought to
be based on a principled handling of nested Python data structures.
Having such principled pytree-transforming code available solves
some other problems, such as doing away with a need to abuse
tree-mapping for-side-effect-only and having to use a hope-and-pray
approach to processing very deeply nested values which with a recursive
approach might trigger a RecursionError.

We specifically want to cover the use case of having ML model
parameters that are available in a nested Python data structure for
which there "almost" is a unique-up-to-unique-isomorphism mapping from
and to this Algebraic Data Type:

`data ModelParams a = Array a | Node [(String, ModelParams a)]`

In this correspondence, `a` is some array-type (perhaps
`numpy.ndarray`, `jax.numpy.ndarray`, `tf.tensor`, etc.), but the
data-processing code is effectively entirely agnostic to this, and a
`Node` is "almost" an associative-list of (key, value) pairs
representing a Python dict. (Note: The "almost" here is mostly about
the conceptual wart that assoc-lists can in principle have key
duplicates, but Python dicts can not. This is however not a problem
since all we need is the transformation in one direction,
i.e. whatever data-processing `f` we want to express on the
model-parameters-pytree, we can express by specifying a "faithful"
mapping `m` into the above algebraic data type through which every
such pytree data transform factorizes, i.e. for every `f` we can find
a `g` such that `f(p) = g(m(p))`.)

## Components

The main workhorse in this module is the `pytree_iter` function that
maps a "PyTree (such as representing `ModelParams`)" to an iterator
over values obtained by applying a mapping-function to the "key-path"
and leaf-value for every leaf, where the "key-path" contains a
linked-list representation of the reversed sequence of keys from the
tree-root, with list-nodes being represented by pairs
`(latest_dict_key, rest_path)`, and the empty path being represented
by `()`.

For the sake of genericity, `pytree_iter` is built in such a way that
it actually can handle any kind of traversal of PyTree-trees that do
represent algebraic data types (note however that some some do not) -
but for this to make sense, the user must have a way to define how to
interpret tree-nodes, in particular identify leaves. This requires
providing a function `node_handler` with the same signature and
behavior as described below for "node handlers".

Additionally, this module provides mapping-over-pytrees via
`pytree_map`, which is also built in such a way that it makes the
correspondence between an algebraic data type and its Python
nested-tree representation explicit. Despite being powerful and
flexible, this, however, may in general require a bit more effort to
wire up, since node-rebuilding can be fairly nontrivial.

Furthermore, as a prominent application, this module provides a simple
deep-freezing function that translates a nested Python data structure
to deeply-immutable form.

## Concepts and Conventions

"revtuple representation":

  As we iterate over a tree, we will have to keep track of the
  path-to-tree-root. Naturally, two sibling nodes `n1` and `n2`
  will share the same parent-path (being siblings), so it makes
  sense to use a linked-list-with-shared-tail representation.
  Python does not have a natural notion for that, so we use
  recursively-constructed tuples `(node_tag, parent_path)`
  that represent the path-from-root in-reverse-order, i.e.
  for a non-empty path `p`, `p[0]` is the node-tag at the
  deepest nesting level. We call this a "revtuple representation"
  of the path.

"node handler":

  A node-handler classifies a tree-node as "leaf or other node", and
  for non-leaf nodes provides information about both its children and
  how to rebuild it. The behavior of a node-handler function must be
  in alignment with this docstring:

  '''Processes a tree-node as required by pytree-iteration and -mapping.

  Args:
    revtuple_path: Revtuple-representation of the path-from-root
      to the current node.
    node: a tree-node in a ML-model tree that is recursively
      built out of leaf-values and other nodes.

  Returns:
    `None` if the tree-node is to be regarded as a leaf, otherwise
    a pair `(rebuilder, iterator)`, where `iterator` iterates
    over the data-content of the node, each item represented as a pair
    of `(lookup_path_item, value_item)`, and `rebuilder` is a function
    which, when applied to an iterable of the aforementioned value-items
    (or some transformation thereof) returns a node that is equivalent
    to the original (or up to a transformation of the contents).

  Raises:
    InvalidTreeNodeError: If the tree contains a node of a kind
      that is not expected to show up.
  '''

  Examples:

    (The behavior of a node-handler is somewhat nontrivial, so covering
    two very common cases via examples is in order.)

    This node-handler would allow descending into (nested)
    instances of `list` (but not subclass instances thereof):

    ```def list_node_handler(revtuple_path, obj):
         ''' ... '''
         if type(obj) is list:
           return list, enumerate(obj)
         else:
           return None
    ```

    This node-handler would allow descending into (nested) mappings,
    which upon rebuilding would get turned into `dict` instances:

    ```def mapping_node_handler(revtuple_path, obj):
         ''' ... '''
         if isinstance(obj, collections.abc.Mapping):
           # For generic mappings, we cannot rely on key- and item-iteration
           # being guaranteed to use identical iteration-order.
           items = list(obj.items())
           keys = [kv[0] for kv in items]
           return (lambda values: dict(zip(keys, values))), items
         else:
           return None
    ```

    A dict/mapping node-handler can of course rename keys, add or remove
    entries, make decisions based on the item-path, or map a dict to
    an associative list, etc.

## Further Design Notes

The `pytree_map` function requests the leaf-transform and node-handler
to be side-effect-free functions. This is both required to leave
implementation-side flexibility, and also follows the general LISP
recommendation to not abuse mapping (which should be a pure
data-transformation) for imperative data processing. Overall, if
a need for more general "nested datastructures" processing becomes
pressing, it is for the better if this leads to a proper articulation
of the specific needs, to be addressed with appropriate design, rather
than abuse of functional data-transforms becoming "a bad idiom
that turned into established practice".

"""

import collections.abc
import immutabledict

import numpy

from typing import Any, Callable, Iterable, Iterator, TypeVar


T = TypeVar('T')
U = TypeVar('U')

KT = TypeVar('KT')
NT = TypeVar('NT')


## Type of the reverse-order-keys-to-root path.
# (This code actually illustrates why https://xkcd.com/2483/ is very misguided.)
RevTuplePath = tuple

## Type of the `leaf_transform` function-argument used for tree-iteration.
#
# This would be the correct type we would have to specify here but cannot,
# since the design of Python's static typing at the time of this writing
# is too broken for that:
#
#   type LeafTransformFunc[L, R] = Callable[[RevTuplePath, L], R]
#
# Instead, we have to settle for...:
LeafTransformFunc = Callable[[RevTuplePath, Any], Any]


## Type of the `tree_node_handler` function-argument used for
## tree-iteration and tree-mapping.
#
# Again, this is the correct type we would have to put here but cannot:
#
# type NodeHandlerFunc[KT] = (
#   Callable[[NT],
#            None | tuple[Callable[[Iterable[tuple[KT, NT]]], NT],
#                         Iterator[tuple[KT, NT]]]])
#
# ...so, we have to instead settle for:
NodeHandlerFunc = (
  Callable[[RevTuplePath, NT],
           None | tuple[Callable[[Iterable[tuple[Any, NT]]], NT],
                        Iterator[tuple[Any, NT]]]])


Predicate = Callable[[object], bool]


class InvalidTreeNodeError(ValueError):
  """Encountered a tree-node of invalid type."""


def linearize_revtuple_path(
    revtuple_path: RevTuplePath,
    present_as: Callable[[Iterator[T]], U] = tuple) -> U:
  """Translates a revtuple path to (typically) linear form.

  With default `present_as`, this will map a path of the form
  `(key_{N}, (key_{N-1}, ..., (root, ())))` into a tuple
  (root, ..., key_{N-1}, key_{N}).

  Args:
    revtuple_path: A linked-list-as-recursive-pairs
      reverse-order tuple-representation of the path.
      Path-root is `()`, and node-key `x` relative to
      earlier path `p` is represented as `(x, p)`.
    present_as: Callable that consumes an iterator over
      path-pieces - with the deepest-nesting level coming last -
      turning it into a linearized path. Defaults to `tuple`.

  Returns:
    Linearized presentation of all the node-keys in the
    recursive-path in order, deepest-down path component coming last.
  """
  pieces = []
  todo = revtuple_path
  while todo:
    node, todo = todo
    pieces.append(node)
  return present_as(reversed(pieces))


# This function itself has type `NodeHandlerFunc`, but Python does not
# allow us to here simply type-annotate it like this. We cannot even
# introduce an abbreviation for the complicated output-type,
# since that would have to be parametric in node-type `NT` (and `KT`).
def everything_is_a_leaf_node_handler(
    revtuple_path: tuple,
    node : NT) -> (
    None | tuple[Callable[[Iterable[tuple[Any, NT]]], NT],
                 Iterator[tuple[Any, NT]]]):
  """Processes a tree-node as required by pytree-iteration and -mapping.

  Interface and signature are in alignment with the requirements for a
  "node handler" function explained in the module-docstring.

  Args:
    revtuple_path: the path-to-root for this node.
    node: a tree-node.

  Returns:
    `None`, i.e. classifying any kind of node as a leaf-node.
  """
  del revtuple_path, node  # Unused.
  return None


def leaf_summary(path: RevTuplePath, x: object):
  """Produces a human-readable summary-string for a leaf-node.

  Args:
    path: revtuple representation of the path-to-root.
    x: The leaf-value.
  """
  del path  # Ignored here.
  tx = type(x)
  mod = tx.__module__
  modname = mod if isinstance(mod, str) else mod.__name__
  type_str = f'{modname}.{tx.__qualname__}'
  repr_str = repr(x)
  repr_abbrev = repr_str if len(repr_str) < 40 else repr_str[:40] + ' ...'
  # On str, int, float, etc. `{type_str}(repr(x))` would actually still be
  # a (non-literal) Python-expression that would evaluate to the original value.
  # However, we make no promises beyond "human-readable".
  return f'{type_str}({repr_abbrev})'


# With respect to static type annotations, the limitations of Python's
# approach to static typing really become prominently visible here.
#
# Different arguments have type-parameters, but since there is no way
# to have parametric abbreviations such as `LeafTransformFunc[L, R]`,
# the only way we would have available to express relations between
# type-parameters would be to substitute in the not-abbreviated form of
# `NodeHandlerFunc` and `LeafTransformFunc`, giving us something monstrous.
# We instead here settle for "we cannot express that `tree` must
# have the same type as the input-type to `tree_node_handler` and use `Any`,
# and likewise for leaf_transform and the output.
def pytree_leaf_iter(
    tree: Any,
    leaf_transform: LeafTransformFunc,
    node_handler: NodeHandlerFunc = everything_is_a_leaf_node_handler,
  ) -> Iterator[Any]:
  # ...actual return type would be `Iterator[{what leaf_transform returns}]`.
  """Iterates over the leaves of a tree.

  Args:
    tree: The tree to iterate over.
    leaf_transform: A callable `f` that will get applied
      as `f(revtuple_path, leaf)`, where `revtuple_path`
      is the revtuple representation of the path to the
      leaf from the root.
    node_handler: A "node handler" (see module docstring)
      that processes nodes encountered during iterative traversal.

  Yields:
    Value of `leaf_transform(p, x)`, where `x` is the current leaf
    and `p` is its revtuple-path to the root.
  """
  # Note: Exit points for the code below are in non-obvious places
  # and hence marked with " # ***EXIT***".
  #
  # Doing iteration properly is slightly nontrivial.
  # One may be tempted to go for a very simple recursive implementation
  # (with an extra pre-final `path` argument to `pytree_iter`):
  #
  # maybe_substructure = node_handler(path, tree)
  # if maybe_substructure is None:
  #   # We are looking at a leaf-node.
  #   yield leaf_transform(path, tree)
  # else:
  #   _, contents_iter = maybe_substructure
  #   for k, v in contents_iter:
  #     yield from pytree_iter(v, leaf_transform, (k, path), node_handler)
  #
  # That, however, would be flawed, since there is no a priori reason
  # why a pytree may not be a very deeply nested structure - such as a
  # long linked list. That would then risk raising `RecursionError`,
  # and since Python by design(!) does not perform tail call elimination
  # or any other kind of advanced CPS transforms, there is no recursive
  # solution here. So, to do this properly, we have to do this iteratively.
  #
  # We are facing an annoying situation here: If `tree` itself is a leaf,
  # we have two options: (a) wrapping it up in a one-node tree
  # and processing that, or (b) special-casing "root is a leaf".
  # Option (b) leads to some mild node-processing code-duplication
  # for a single node (the root).
  # Option (a) requires having special cases for node-processing that
  # get looked at for every tree node. We go with option (b) here.
  maybe_substructure = node_handler((), tree)
  if maybe_substructure is None:
    # The tree itself is a leaf.
    yield leaf_transform((), tree)
    return  # ***EXIT***
  # Otherwise, we are looking at a tree.
  _, contents_iter = maybe_substructure
  current_revtuple_path = ()
  work_to_do = [contents_iter]
  # Otherwise-unreachable sentinel for reliably identifying
  # iterator-exhaustion without using exceptions:
  sentinel = object()
  while True:
    current_iter = work_to_do[-1]
    maybe_next_item = next(current_iter, sentinel)
    if maybe_next_item is sentinel:
      # We are done at this level.
      work_to_do.pop()
      if not work_to_do: return  # ***EXIT***
      current_revtuple_path = current_revtuple_path[1]
    else:
      path_piece, subtree = maybe_next_item
      extended_revtuple_path = (path_piece, current_revtuple_path)
      maybe_subtree_substructure = node_handler(extended_revtuple_path, subtree)
      if maybe_subtree_substructure is None:  # Case: subtree is a leaf.
        yield leaf_transform(extended_revtuple_path, subtree)
      else:  # Case: subtree is a tree.
        current_revtuple_path = (path_piece, current_revtuple_path)
        _, items_iter = maybe_subtree_substructure
        work_to_do.append(items_iter)


# The current design approach here would be appropriate for
# applying leaf-transforms while retaining the structure of the tree -
# which closely corresponds to e.g. a (a -> b) -> (Tree a -> Tree b) functor.
#
# It is not entirely clear whether this is the abstraction that we should
# consider as being appropriately generic to flesh out explicitly - rather
# than starting from a more general approach of which this then is a special
# case. Some background: https://ncatlab.org/nlab/show/recursion+scheme
#
# On the other hand, there is a lot of flexibility via whatever
# node-rebuilder a node-handler produces - this can do quite some reshaping
# of a tree, including dropping or duplicating nodes.
def pytree_map(
    tree: Any,
    leaf_transform,
    node_handler: NodeHandlerFunc = everything_is_a_leaf_node_handler,
  ):
  """Maps a (potentially nested) Python value to another such value.

  Args:
    tree: The Python-object to be mapped.
    leaf_transform: A callable `f` that will get applied
      as `f(revtuple_path, leaf)`, where `revtuple_path`
      is the revtuple representation of the path to the
      leaf from the root. Must be side effect free.
    node_handler: A "node handler" (see module docstring)
      that processes nodes encountered during iterative traversal.
      Must be side effect free.

  Returns:
    The outcome of translating `tree`.
  """
  # Note: Exit points for the code below are in non-obvious places
  # and hence marked with " # ***EXIT***".
  #
  # Otherwise-inaccessible sentinel object, for reliably identifying
  # missing-values via identity-check against sentinel lookup-default.
  sentinel = object()
  # Code structure mostly follows pytree_leaf_iter.
  maybe_substructure = node_handler((), tree)
  if maybe_substructure is None:
    return leaf_transform((), tree)  # ***EXIT***
  rebuilder, items_iter = maybe_substructure
  current_revtuple_path = ()
  # Per-level, we have a triplet of:
  # (rebuilder, remaining_items_to_iterate_over, processed).
  parts_for_assembly = [(rebuilder, items_iter, [])]
  while True:
    this_rebuilder, this_items_iter, this_done_pieces = parts_for_assembly[-1]
    maybe_next_item = next(this_items_iter, sentinel)
    if maybe_next_item is sentinel:
      # We are done with all the items for this level.
      parts_for_assembly.pop()
      built_iter = this_rebuilder(this_done_pieces)
      if not parts_for_assembly:  # No outer structure, so at-top-level.
        return built_iter  # ***EXIT***
      else:  # We have outer structure.
        parts_for_assembly[-1][-1].append(built_iter)
        current_revtuple_path = current_revtuple_path[1]
        continue  # ...with next is-the-final-item-complete-check.
    else:
      # More constituents of the current item.
      path_piece, subtree_item = maybe_next_item
      extended_revtuple_path = (path_piece, current_revtuple_path)
      maybe_subtree_substructure = node_handler(
          extended_revtuple_path,
          subtree_item)
      if maybe_subtree_substructure is None:
        this_done_pieces.append(
            leaf_transform(extended_revtuple_path, subtree_item))
      else:
        # We have a subtree.
        subtree_rebuilder, subtree_items_iter = maybe_subtree_substructure
        current_revtuple_path = (path_piece,
                                 current_revtuple_path)
        parts_for_assembly.append(
          (subtree_rebuilder, subtree_items_iter, []))


def deep_freeze(
    tree,
    *,
    is_mapping : Predicate = lambda x: isinstance(x, collections.abc.Mapping),
    is_set : Predicate = lambda x: isinstance(x, collections.abc.Set),
    is_sequence : Predicate = lambda x: isinstance(x, (list, tuple)),
    leaf_fn: Callable[[Any], Any] = lambda x: x,
    ):
  """Recursively freezes Set/Mapping/List/Tuple structures.

  Args:
    tree: The potentially deeply-nested object to deep-freeze.
    is_mapping: Callable that decides whether a sub-object is a mapping.
      Defaults to an `isinstance()` check for `collections.abc.Mapping`.
    is_set: Callable that decides whether a sub-object is a set.
      Defaults to an `isinstance()` check for `collections.abc.Set`.
    is_sequence: Callable that decides whether a sub-object is a sequence.
      Defaults to a check for being a `tuple` or `list` instance.
    leaf_fn: Function to use for translating non-mapping/set/sequence
      instances.

  Returns:
    Translated-to-deeply-immutable form of `tree`.
  """
  idict = immutabledict.immutabledict
  def freeze_node_handler(path, x):
    if is_set(x):
      return frozenset, ((None, y) for y in x)
    if is_mapping(x):
      # Mappings already have hashable, so
      # (should-be-)deeply-immutable keys.
      # Hence, we only need to deep-freeze the values.
      #
      # Note that non-`dict` mappings might not guarantee
      # to respect iteration-order, so we have to be careful here:
      items = list(x.items())
      keys = [kv[0] for kv in items]
      values = [kv[1] for kv in items]
      return ((lambda ys: idict(zip(keys, ys))),
              iter(items))
    if is_sequence(x):
      return tuple, enumerate(iter(x))
    # Otherwise, this should not be traversed.
    return None
  def leaf_transform(revtuple_path, value):
    del revtuple_path  # Unused.
    return leaf_fn(value)
  return pytree_map(tree, leaf_transform, freeze_node_handler)
