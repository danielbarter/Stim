/*
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _STIM_MEM_SIMD_BIT_TABLE_V2_H
#define _STIM_MEM_SIMD_BIT_TABLE_V2_H

#include "stim/mem/simd_bits.h"

void debug_stub() {}

namespace stim {

/// A 2d array of bit-packed booleans, padded and aligned to make simd operations more efficient.
/// The table contents are indexed by a major axis (not contiguous in memory) then a minor axis (contiguous in memory).

template <size_t W>

struct simd_bit_table_v2 {

    /// num_minor_bits_padded is always a multiple of W
    size_t num_major_bits_padded;
    size_t num_minor_bits_padded;
    size_t num_major_bits;
    size_t num_minor_bits;

    simd_bits<W> data;

    simd_bit_table_v2(size_t num_major_bits, size_t num_minor_bits);

};

}  // namespace stim

template <size_t W>
stim::simd_bit_table_v2<W>::simd_bit_table_v2(size_t num_major_bits, size_t num_minor_bits)
    : num_major_bits_padded(num_major_bits),
      num_minor_bits_padded(min_bits_to_num_bits_padded<W>(num_minor_bits)),
      num_major_bits(num_major_bits),
      num_minor_bits(num_minor_bits),
      data(min_bits_to_num_bits_padded<W>(num_minor_bits) * num_major_bits)
{
    debug_stub();
}

#endif
