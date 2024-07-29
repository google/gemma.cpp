"""Basic tests for 'algebraic data type based pytree' transformations."""


import io
import os
import tempfile
import unittest

import numpy

import ml_model_transforms


def _get_model(prefix):
  return {
    prefix + 'a1': numpy.arange(1000, 1024).reshape(6, 4).astype(numpy.float32),
    prefix + 'a2': numpy.arange(2000, 2048).reshape(6, 8).astype(numpy.float32),
    prefix + 'b1': {
      prefix + 'c1': numpy.arange(100, 127).reshape(3, 3, 3).astype(numpy.int8),
      prefix + 'c2': numpy.arange(100, 128).reshape(7, 4).astype(numpy.float64)
    }}


class MLModeltransformsTest(unittest.TestCase):
  """Basic correctness validation tests for ML-model transformations."""

  def test_ml_model_leaf_summary(self):
    """Tests guarantees given by `ml_model_leaf_summary`."""
    summary = ml_model_transforms.ml_model_leaf_summary(
      ('a', ()),
      numpy.arange(1000, 1024).reshape(6, 4).astype(numpy.int16),
      sep='##')
    self.assertIn('##', summary)  # Separator is respected.
    self.assertIn('(6, 4)', summary)  # Shape is mentioned somewhere.
    self.assertIn('int16', summary)  # dtype is mentioned somewhere.

  def test_revtuple_autovivify_from_linear(self):
    """Tests guarantees given by `revtuple_autovifify_from_linear`."""
    with self.subTest(guarantee='empty'):
      self.assertEqual(
        ml_model_transforms.revtuple_autovifify_from_linear([]),
        {})
    with self.subTest(guarantee='generic'):
      keys_vals = [(('a', 'b1', 'c1'), 1001),
                   (('a', 'b2'), 1002),
                   (('a2',), 1003),
                   ]
      self.assertEqual(
        ml_model_transforms.revtuple_autovifify_from_linear(keys_vals),
        {'a': {'b1': {'c1': 1001}, 'b2': 1002}, 'a2': 1003})

  def test_model_overview(self):
    """Tests guarantees given by `model_overview`."""
    model = _get_model('xyz')
    out_io = io.StringIO()
    ml_model_transforms.model_overview(model, out=out_io)
    overview = out_io.getvalue()
    self.assertIn('xyz', overview)

  def test_model_contents(self):
    """Tests guarantees given by `model_contents`."""
    model = _get_model('pq_')
    contents = ml_model_transforms.model_contents(model)
    fingerprints = {k: (a.shape, a.ravel()[:3].tolist())
                    for k, a in contents.items()}
    self.assertEqual(fingerprints,
                     {('pq_a1',): ((6, 4), [1000.0, 1001.0, 1002.0]),
                      ('pq_a2',): ((6, 8), [2000.0, 2001.0, 2002.0]),
                      ('pq_b1', 'pq_c1'): ((3, 3, 3), [100, 101, 102]),
                      ('pq_b1', 'pq_c2'): ((7, 4), [100.0, 101.0, 102.0])})

  def test_model_save_load_basic(self):
    """Tests basic guarantees given by `model_save` and `model_load`."""
    # What we care about here is that the round trip works - so
    # it makes more sense to test saving and loading as one unit.
    model_orig = _get_model('model_')
    with tempfile.TemporaryDirectory() as tempdir:
      filepath_stem = os.path.join(tempdir, 'the_model')
      total_size, total_time = ml_model_transforms.model_save(model_orig,
                                                              filepath_stem)
      self.assertGreater(total_size, 0)
      self.assertGreater(total_time, 0)
      model_reloaded = ml_model_transforms.model_load(filepath_stem)
      contents_orig = ml_model_transforms.model_contents(model_orig)
      contents_reloaded = ml_model_transforms.model_contents(model_reloaded)
      self.assertEqual(
        {k: v.tolist() for k, v in contents_orig.items()},
        {k: v.tolist() for k, v in contents_reloaded.items()})


if __name__ == '__main__':
  unittest.main()
