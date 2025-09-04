// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "util/basics.h"

#include <stddef.h>
#include <stdint.h>

#include "hwy/contrib/sort/vqsort.h"
#include "hwy/highway.h"
#include "hwy/timer.h"

namespace gcpp {

RNG::RNG(bool deterministic) {
  // Pi-based nothing up my sleeve numbers from Randen.
  key_[0] = 0x243F6A8885A308D3ull;
  key_[1] = 0x13198A2E03707344ull;

  if (!deterministic) {  // want random seed
    if (!hwy::Fill16BytesSecure(key_)) {
      HWY_WARN("Failed to fill RNG key with secure random bits");
      // Entropy not available. The test requires that we inject some
      // differences relative to the deterministic seeds.
      key_[0] ^= reinterpret_cast<uint64_t>(this);
      key_[1] ^= hwy::timer::Start();
    }
  }

  // Simple key schedule: swap and add constant (also from Randen).
  for (size_t i = 0; i < kRounds; ++i) {
    key_[2 + 2 * i + 0] = key_[2 * i + 1] + 0xA4093822299F31D0ull;
    key_[2 + 2 * i + 1] = key_[2 * i + 0] + 0x082EFA98EC4E6C89ull;
  }
}

namespace hn = hwy::HWY_NAMESPACE;
using D = hn::Full128<uint8_t>;  // 128 bits for AES
using V = hn::Vec<D>;

static V Load(const uint64_t* ptr) {
  return hn::Load(D(), reinterpret_cast<const uint8_t*>(ptr));
}

RNG::result_type RNG::operator()() {
  V state = Load(counter_);
  counter_[0]++;
  state = hn::Xor(state, Load(key_));  // initial whitening

  static_assert(kRounds == 5 && sizeof(key_) == 12 * sizeof(uint64_t));
  state = hn::AESRound(state, Load(key_ + 2));
  state = hn::AESRound(state, Load(key_ + 4));
  state = hn::AESRound(state, Load(key_ + 6));
  state = hn::AESRound(state, Load(key_ + 8));
  // Final round: fine to use another AESRound, including MixColumns.
  state = hn::AESRound(state, Load(key_ + 10));

  // Return lower 64 bits of the u8 vector.
  const hn::Repartition<uint64_t, D> d64;
  return hn::GetLane(hn::BitCast(d64, state));
}

}  // namespace gcpp
