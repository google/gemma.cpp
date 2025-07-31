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

"""Tests for CLIF wrapped .sbs writer."""

import numpy as np

from absl.testing import absltest
from compression.python import compression
from python import configs


class CompressionTest(absltest.TestCase):

  def test_sbs_writer(self):
    info_192 = configs.TensorInfo()
    info_192.name = "ignored_192"
    info_192.axes = [0]
    info_192.shape = [192]

    temp_file = self.create_tempfile("test.sbs")
    writer = compression.SbsWriter(temp_file.full_path)
    writer.insert(
        "tensor0",
        # Large enough to require scaling.
        np.array([3.0012] * 128 + [4.001] * 64, dtype=np.float32),
        configs.Type.kSFP,
        info_192,
    )

    # 2D tensor.
    info_2d = configs.TensorInfo()
    info_2d.name = "ignored_2d"
    info_2d.axes = [0, 1]
    info_2d.shape = [96, 192]
    writer.insert(
        "tensor_2d",
        np.array([i / 1e3 for i in range(96 * 192)], dtype=np.float32),
        configs.Type.kBF16,
        info_2d,
    )

    # 3D collapsed into rows.
    info_3d = configs.TensorInfo()
    info_3d.name = "ignored_3d"
    info_3d.axes = [0, 1, 2]
    info_3d.shape = [10, 12, 192]
    info_3d.cols_take_extra_dims = False
    writer.insert(
        "tensor_3d",
        # Verification of scale below depends on the shape and multiplier here.
        np.array([i / 1e3 for i in range(10 * 12 * 192)], dtype=np.float32),
        configs.Type.kSFP,
        info_3d,
    )

    # Exercise all types supported by Compress.
    info_256 = configs.TensorInfo()
    info_256.name = "ignored_256"
    info_256.axes = [0]
    info_256.shape = [256]
    writer.insert(
        "tensor_sfp",
        np.array([0.000375] * 128 + [0.00009] * 128, dtype=np.float32),
        configs.Type.kSFP,
        info_256,
    )
    writer.insert(
        "tensor_bf",
        np.array([0.000375] * 128 + [0.00007] * 128, dtype=np.float32),
        configs.Type.kBF16,
        info_256,
    )
    writer.insert(
        "tensor_f32",
        np.array([0.000375] * 128 + [0.00006] * 128, dtype=np.float32),
        configs.Type.kF32,
        info_256,
    )

    config = configs.ModelConfig(
        configs.Model.GEMMA_TINY,
        configs.Type.kSFP,
        configs.PromptWrapping.GEMMA_IT,
    )
    tokenizer_path = ""  # no tokenizer required for testing
    writer.write(config, tokenizer_path)

    print("Ignore next two warnings; test does not enable model deduction.")
    reader = compression.SbsReader(temp_file.full_path)

    self.assertEqual(reader.config.model, configs.Model.GEMMA_TINY)
    self.assertEqual(reader.config.weight, configs.Type.kSFP)

    mat = reader.find_mat("tensor0")
    self.assertEqual(mat.cols, 192)
    self.assertEqual(mat.rows, 1)
    self.assertEqual(mat.type, configs.Type.kSFP)
    self.assertAlmostEqual(mat.scale, 4.001 / 1.875, places=5)

    mat = reader.find_mat("tensor_2d")
    self.assertEqual(mat.cols, 192)
    self.assertEqual(mat.rows, 96)
    self.assertEqual(mat.type, configs.Type.kBF16)
    self.assertAlmostEqual(mat.scale, 1.0)

    mat = reader.find_mat("tensor_3d")
    self.assertEqual(mat.cols, 192)
    self.assertEqual(mat.rows, 10 * 12)
    self.assertEqual(mat.type, configs.Type.kSFP)
    self.assertAlmostEqual(mat.scale, 192 * 120 / 1e3 / 1.875, places=2)

    mat = reader.find_mat("tensor_sfp")
    self.assertEqual(mat.cols, 256)
    self.assertEqual(mat.rows, 1)
    self.assertEqual(mat.type, configs.Type.kSFP)
    self.assertAlmostEqual(mat.scale, 1.0)

    mat = reader.find_mat("tensor_bf")
    self.assertEqual(mat.cols, 256)
    self.assertEqual(mat.rows, 1)
    self.assertEqual(mat.type, configs.Type.kBF16)
    self.assertAlmostEqual(mat.scale, 1.0)

    mat = reader.find_mat("tensor_f32")
    self.assertEqual(mat.cols, 256)
    self.assertEqual(mat.rows, 1)
    self.assertEqual(mat.type, configs.Type.kF32)
    self.assertAlmostEqual(mat.scale, 1.0)


if __name__ == "__main__":
  absltest.main()
