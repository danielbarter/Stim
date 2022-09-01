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

#include <sstream>

#include "stim/mem/simd_bits.h"

void debug_stub() {}

namespace stim {

/// A 2d array of bit-packed booleans, padded and aligned to make simd operations more efficient.
/// The table contents are indexed by a major axis (not contiguous in memory) then a minor axis (contiguous in memory).

template <size_t W>

struct simd_bit_table_v2 {

    /// num_minor_bits_padded is always a multiple of W
    /// TODO: num_minor_bits_padded should only be settable through a setter method
    /// which asserts that num_minor_bits_padded % W == 0
    size_t num_major_bits_padded;
    size_t num_minor_bits_padded;
    size_t num_major_bits;
    size_t num_minor_bits;

    simd_bits<W> data;

    simd_bit_table_v2(size_t num_major_bits, size_t num_minor_bits);

    inline size_t num_simd_words_minor() const {
        return num_minor_bits_padded / W;
    }

    /// Returns a reference to a row (column) of the table, when using row (column) major indexing.
    inline simd_bits_range_ref<W> operator[](size_t major_index) {
        size_t num_simd_words_minor = this->num_simd_words_minor();
        return data.word_range_ref(major_index * num_simd_words_minor, num_simd_words_minor);
    }
    /// Returns a const reference to a row (column) of the table, when using row (column) major indexing.
    inline const simd_bits_range_ref<W> operator[](size_t major_index) const {
        size_t num_simd_words_minor = this->num_simd_words_minor();
        return data.word_range_ref(major_index * num_simd_words_minor, num_simd_words_minor);
    }

    /// Creates a square table with 1s down the diagonal.
    static simd_bit_table_v2 identity(size_t n);

    /// Returns a string representation of the tables contents
    std::string str() const;

    /// Parses a bit table from some text.
    ///
    /// Args:
    ///     text: A paragraph of characters specifying the contents of a bit table.
    ///         Each line is a row (a major index) of the table.
    ///         Each position within a line is a column (a minor index) of the table.
    ///         A '1' at character C of line L (counting from 0) indicates out[L][C] will be set.
    ///         A '0', '.', or '_' indicates out[L][C] will not be set.
    ///         Leading newlines and spaces at the beginning of the text are ignored.
    ///         Leading spaces at the beginning of a line are ignored.
    ///         Other characters result in errors.
    ///
    /// Returns:
    ///     A simd_bit_table with cell contents corresponding to the text.
    static simd_bit_table_v2 from_text(const char *text);


};

}  // namespace stim

// TODO: move following into an inl file
namespace stim {

template <size_t W>
simd_bit_table_v2<W>::simd_bit_table_v2(size_t num_major_bits, size_t num_minor_bits)
    : num_major_bits_padded(num_major_bits),
      num_minor_bits_padded(min_bits_to_num_bits_padded<W>(num_minor_bits)),
      num_major_bits(num_major_bits),
      num_minor_bits(num_minor_bits),
      data(min_bits_to_num_bits_padded<W>(num_minor_bits) * num_major_bits) {
}

template <size_t W>
simd_bit_table_v2<W> simd_bit_table_v2<W>::identity(size_t n) {
    simd_bit_table_v2<W> result(n, n);
    for (size_t k = 0; k < n; k++) {
        result[k][k] = true;
    }
    return result;
}

template <size_t W>
std::string simd_bit_table_v2<W>::str() const {
    std::stringstream out;
    for (size_t row = 0; row < num_major_bits; row++) {
        if (row) {
            out << "\n";
        }
        for (size_t col = 0; col < num_minor_bits; col++) {
            out << ".1"[(*this)[row][col]];
        }
    }
    return out.str();
}


template <size_t W>
simd_bit_table_v2<W> simd_bit_table_v2<W>::from_text(const char *text) {
    std::vector<std::vector<bool>> lines;
    lines.push_back({});

    // Skip indentation.
    while (*text == '\n' || *text == ' ') {
        text++;
    }

    for (const char *c = text; *c;) {
        if (*c == '\n') {
            lines.push_back({});
            c++;
            // Skip indentation.
            while (*c == ' ') {
                c++;
            }
        } else if (*c == '0' || *c == '.' || *c == '_') {
            lines.back().push_back(false);
            c++;
        } else if (*c == '1') {
            lines.back().push_back(true);
            c++;
        } else {
            throw std::invalid_argument(
                "Expected indented characters from \"10._\\n\". Got '" + std::string(1, *c) + "'.");
        }
    }

    // Remove trailing newline.
    if (!lines.empty() && lines.back().empty()) {
        lines.pop_back();
    }

    size_t num_cols = 0;
    for (const auto &v : lines) {
        num_cols = std::max(v.size(), num_cols);
    }

    size_t num_rows = 0;
    num_rows = std::max(num_rows, lines.size());
    simd_bit_table_v2<W> out(num_rows, num_cols);
    for (size_t row = 0; row < lines.size(); row++) {
        for (size_t col = 0; col < lines[row].size(); col++) {
            out[row][col] = lines[row][col];
        }
    }
    return out;
}


}  // namespace stim

#endif
