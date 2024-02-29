/*
 * Copyright (c) 2021 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

kernel void Collatz(
    global int * const result
) {
    const size_t gli = get_global_id(0);
    const size_t ind = gli - get_global_offset(0);

    int steps = 0;

    ulong n = gli + 1;

    while (n != 1) {
        if (steps > INT_MAX - 3) {
            steps = 0;
            break;
        }
        if (n & 1) {
            ulong m = 3 * n + 1;
            if (m < n) {
                steps = -steps - 1;
                break;
            }
            n = m >> 1;
            steps += 2;
        }
        else {
            n >>= 1;
            ++steps;
        }
    }

    result[ind] = steps;
}
