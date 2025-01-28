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
    temp_file = self.create_tempfile("test.sbs")
    tensor_info = configs.TensorInfo()
    tensor_info.name = "foo"
    tensor_info.axes = [0]
    tensor_info.shape = [192]

    writer = compression.SbsWriter(compression.CompressorMode.NO_TOC)
    writer.insert(
        "foo",
        np.array([0.0012] * 128 + [0.001] * 64, dtype=np.float32),
        configs.Type.kSFP,
        tensor_info,
        1.0,
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
    self.assertEqual(writer.debug_num_blobs_added(), 5)
    self.assertEqual(writer.write(temp_file.full_path), 0)


if __name__ == "__main__":
  absltest.main()
