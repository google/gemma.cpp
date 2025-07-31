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

"""A simple example of using the gemma.cpp Python wrapper."""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
import numpy as np

from python import gemma


_MODEL_DIR = flags.DEFINE_string(
    "model_dir",
    "",
    "Path to the Gemma model directory.",
)

_PROMPT = flags.DEFINE_string(
    "prompt",
    "Write an email to the moon.",
    "Prompt to generate text with.",
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  tokenizer_path = os.path.join(_MODEL_DIR.value, "tokenizer.spm")
  weights_path = os.path.join(_MODEL_DIR.value, "gemma2-2b-it-sfp.sbs")
  print(f"Loading model from {tokenizer_path} and {weights_path}")
  model = gemma.GemmaModel(
      tokenizer_path=tokenizer_path,
      weights_path=weights_path,
      max_threads=24,
  )

  prompt = _PROMPT.value
  print(f"Running example with prompt='{prompt}'")
  output = model.generate(prompt)
  print(f"Generated output:\n{output}")

  def callback(tok, _):
    s = model.detokenize([tok])
    print(s, end="", flush=True)
    return True

  print(f"\n\nRunning example with streaming callback, prompt='{prompt}'")
  print("Generating output:\n")
  model.generate_ex(prompt, callback, skip_prompt=True)

  prompts = [
      prompt,
      "Tell me a joke.",
      "Please recite the first paragraph of the Declaration of Independence.",
      prompt,
  ]
  print("\n\n\nRunning example with batch generation")
  outputs = model.generate_batch(
      prompts, max_generated_tokens=16, temperature=2.0, top_k=30, seed=123456,
  )
  print("Generated outputs:")
  for prompt, output in zip(prompts, outputs):
    print(f"Prompt: '{prompt}' --->\nOutput: {output}\n")

  # PaliGemma example.
  tokenizer_path = os.path.join(_MODEL_DIR.value, "paligemma_tokenizer.model")
  weights_path = os.path.join(_MODEL_DIR.value, "paligemma-3b-mix-224-sfp.sbs")
  print(f"Loading model from {tokenizer_path} and {weights_path}")
  model = gemma.GemmaModel(
      tokenizer_path=tokenizer_path,
      weights_path=weights_path,
      max_threads=24,
  )
  image = np.array(
      [
          [[255, 0, 0], [0, 255, 0]],  # Red, Green
          [[0, 0, 255], [255, 255, 255]],  # Blue, White
      ],
      dtype=np.float32,
  )
  model.set_image(image)
  prompt = "Describe this image."
  print(f"Running example with a tiny image and prompt='{prompt}'.")
  output, tokens = model.generate_with_image(prompt)
  print(f"Generated {len(tokens)} tokens, output:\n{output}")


if __name__ == "__main__":
  app.run(main)
