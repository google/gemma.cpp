"""Basic tests for 'algebraic data type based pytree' transformations."""


import collections.abc
import sys
import unittest

import pytree_transforms


def _get_deep_pytree(packaging_fn, bottom, depth):
  current = bottom
  for n in reversed(range(depth)):
    current = packaging_fn(n, current)
  return current


def _dict_node_handler(p, d):
  del p  # Unused.
  if isinstance(d, dict):
    keys = d.keys()
    newdict = lambda vals: dict(zip(keys, vals))
    return (newdict, iter(d.items()))
  else:
    return None


class PyTreeTest(unittest.TestCase):
  """Basic correctness validation tests for PyTree transformations."""

  def test_linearize_revtuple_path(self):
    """Tests guarantees given by `linearize_revtuple_path`."""
    linearize_revtuple_path = pytree_transforms.linearize_revtuple_path
    with self.subTest(guarantee='empty'):
      self.assertEqual(linearize_revtuple_path(()), ())
    with self.subTest(guarantee='typical'):
      self.assertEqual(linearize_revtuple_path((30, (20, (10, ())))),
                       (10, 20, 30))
    with self.subTest(guarantee='present_as'):
      self.assertEqual(
          linearize_revtuple_path(
              (30, (20, (10, ()))), present_as=list),
          [10, 20, 30])

  def test_everything_is_a_leaf_node_handler(self):
    """Tests guarantees given by `everything_is_a_leaf_node_handler`."""
    everything_is_a_leaf_node_handler = (
        pytree_transforms.everything_is_a_leaf_node_handler)
    self.assertEqual(everything_is_a_leaf_node_handler((), 'abc'),
                     None)
    self.assertEqual(everything_is_a_leaf_node_handler(('b', ()),
                                                       dict(a=3)),
                     None)

  def test_leaf_summary(self):
    """Tests guarantees given by `leaf_summary`."""
    # Since the docstring only guarantees "a human-readable presentation",
    # we can and should only do loose checks.
    thing = (5678, 9531)
    summary = pytree_transforms.leaf_summary(('key', ()), thing)
    self.assertIsInstance(summary, str)
    self.assertIn(str(thing[0]), summary)
    self.assertIn(str(thing[1]), summary)

  def test_pytree_leaf_iter(self):
    """Tests guarantees given by `pytree_leaf_iter`."""
    pytree_leaf_iter = pytree_transforms.pytree_leaf_iter
    def leaf_transform(path, leaf):
      return repr(leaf) if path and path[0].startswith('R') else leaf
    with self.subTest(guarantee='returns_iterator'):
      result = pytree_leaf_iter(7, leaf_transform, _dict_node_handler)
      self.assertIsInstance(result, collections.abc.Iterator)
    with self.subTest(guarantee='totally_empty'):
      result = list(pytree_leaf_iter({}, leaf_transform, _dict_node_handler))
      self.assertEqual(result, [])
    with self.subTest(guarantee='no_leaves'):
      result = list(pytree_leaf_iter(dict(a={}),
                                     leaf_transform, _dict_node_handler))
      self.assertEqual(result, [])
    with self.subTest(guarantee='is_leaf'):
      result = list(pytree_leaf_iter(777, leaf_transform, _dict_node_handler))
      self.assertEqual(result, [777])
    with self.subTest(guarantee='generic'):
      result = list(pytree_leaf_iter(
          dict(n0=dict(n01=dict(n012=1002,
                                n013=1003,
                                Rn014=1004,
                                ),
                       n02=1005),
               n5=1006),
          leaf_transform, _dict_node_handler))
      self.assertEqual(result, [1002, 1003, '1004', 1005, 1006])
    with self.subTest(guarantee='with_keys'):
      result = list(pytree_leaf_iter(
          dict(n0=dict(n01=dict(n012=1002,
                                n013=1003)),
               n1=1004),
          lambda p, s: (pytree_transforms.linearize_revtuple_path(p), s),
          _dict_node_handler))
      self.assertEqual(result,
                       [(('n0', 'n01', 'n012'), 1002),
                        (('n0', 'n01', 'n013'), 1003),
                        (('n1',), 1004)])

  def test_pytree_map(self):
    """Tests guarantees given by `pytree_map`."""
    pytree_map = pytree_transforms.pytree_map
    leaf_transform = lambda p, s: repr(s)
    tree1 = dict(t0=dict(t10=1001,
                     t11=dict(t110=1002,
                              t111=1003),
                     t12=dict(t120=1004,
                              t121=1005,
                              t122=1006)),
             t1=1007)
    with self.subTest(guarantee='no_leaves'):
      result = pytree_map(dict(a={}),
                          leaf_transform,
                          _dict_node_handler)
      self.assertEqual(result, dict(a={}))
    with self.subTest(guarantee='is_leaf'):
      result = pytree_map(777, leaf_transform, _dict_node_handler)
      self.assertEqual(result, '777')
    with self.subTest(guarantee='generic'):
      result = pytree_map(tree1, leaf_transform, _dict_node_handler)
      self.assertEqual(result['t0']['t10'], '1001')

  def test_deeply_nested(self):
    """Tests correct behavior on deeply-nested data structures."""
    pytree_leaf_iter = pytree_transforms.pytree_leaf_iter
    pytree_map = pytree_transforms.pytree_map
    #
    depth = max(10**5, sys.getrecursionlimit() + 100)
    deep_tree = _get_deep_pytree(lambda n, t: {n: t},
                                 'leaf', depth)
    with self.subTest(function='pytree_leaf_iter'):
      leaves = list(pytree_leaf_iter(deep_tree,
                                     lambda p, s: s.upper(),
                                     _dict_node_handler))
      self.assertEqual(leaves, ['LEAF'])
    with self.subTest(function='pytree_map'):
      mapped_deep_tree = pytree_map(deep_tree,
                                    lambda p, s: s,
                                    _dict_node_handler)
      self.assertIsInstance(mapped_deep_tree, dict)
    with self.subTest(function='combined'):
      leaves = list(
          pytree_leaf_iter(
              pytree_map(deep_tree,
                         lambda p, s: s.capitalize(),
                         _dict_node_handler),
              lambda p, s: s + s,
              _dict_node_handler))
      self.assertEqual(leaves, ['LeafLeaf'])

  def test_deep_freeze(self):
    """Tests guarantees given by `deep_freeze`."""
    frozen = pytree_transforms.deep_freeze(
        dict(a=[1001, 1002, dict(b=(1003, [1004, {1005, 1006}]))]))
    self.assertIsInstance(frozen, collections.abc.Mapping)
    self.assertNotIsInstance(frozen, collections.abc.MutableMapping)
    self.assertIsInstance(frozen['a'], tuple)
    # `frozen` is hashable, and hashes to an integer.
    self.assertIsInstance(hash(frozen), int)


if __name__ == '__main__':
  unittest.main()
