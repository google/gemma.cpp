"""Tests for CLIF wrapped .sbs writer."""

import numpy as np

import unittest
from compression.python import compression
from gemma.python import configs


class CompressionTest(unittest.TestCase):

  def test_sbs_writer(self):
    temp_file = self.create_tempfile("test.sbs")

    writer = compression.SbsWriter()
    writer.insert(
        "foo",
        np.array([0.0012] * 128 + [0.001] * 64, dtype=np.float32),
        configs.Type.kSFP,
    )
    writer.insert_sfp(
        "bar", np.array([0.000375] * 128 + [0.00009] * 128, dtype=np.float32)
    )
    writer.insert_nuq(
        "baz", np.array([0.000125] * 128 + [0.00008] * 128, dtype=np.float32)
    )
    writer.insert_bf16(
        "qux", np.array([0.000375] * 128 + [0.00007] * 128, dtype=np.float32)
    )
    writer.insert_float(
        "quux", np.array([0.000375] * 128 + [0.00006] * 128, dtype=np.float32)
    )
    self.assertEqual(writer.write(temp_file.full_path), 0)


if __name__ == "__main__":
  unittest.main()
