// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cassert>
#include <cstring>
#include <sstream>

// #include "stim/mem/simd_bit_table.h"

namespace stim {

template <size_t W>
simd_bit_table<W>::simd_bit_table(size_t num_bits_major, size_t num_bits_minor)
    : num_bits_major(num_bits_major),
      num_bits_minor(num_bits_minor),
      num_bits_major_padded(min_bits_to_num_bits_padded<W>(num_bits_major)),
      num_bits_minor_padded(min_bits_to_num_bits_padded<W>(num_bits_minor)),
      data(num_bits_major_padded * num_bits_minor_padded) {}



template <size_t W>
simd_bit_table<W> simd_bit_table<W>::identity(size_t n) {
    simd_bit_table<W> result(n, n);
    for (size_t k = 0; k < n; k++) {
        result[k][k] = true;
    }
    return result;
}

template <size_t W>
void simd_bit_table<W>::clear() {
    data.clear();
}

template <size_t W>
bool simd_bit_table<W>::operator==(const simd_bit_table<W> &other) const {
    return num_bits_major == other.num_bits_major
        && num_bits_minor == other.num_bits_minor
        && data == other.data;
}

template <size_t W>
bool simd_bit_table<W>::operator!=(const simd_bit_table<W> &other) const {
    return !(*this == other);
}

template <size_t W>
simd_bit_table<W> simd_bit_table<W>::square_mat_mul(const simd_bit_table<W> &rhs, size_t n) const {
    assert(num_major_bits_padded() >= n && num_minor_bits_padded() >= n);
    assert(rhs.num_major_bits_padded() >= n && rhs.num_minor_bits_padded() >= n);

    auto tmp = rhs.transposed();

    simd_bit_table<W> result(n, n);
    for (size_t row = 0; row < n; row++) {
        for (size_t col = 0; col < n; col++) {
            bitword<W> acc{};
            (*this)[row].for_each_word(tmp[col], [&](bitword<W> &w1, bitword<W> &w2) {
                acc ^= w1 & w2;
            });
            result[row][col] = acc.popcount() & 1;
        }
    }

    return result;
}

template <size_t W>
simd_bit_table<W> simd_bit_table<W>::inverse_assuming_lower_triangular(size_t n) const {
    assert(num_major_bits_padded() >= n && num_minor_bits_padded() >= n);

    simd_bit_table<W> result = simd_bit_table<W>::identity(n);
    simd_bits<W> copy_row(num_minor_bits_padded());
    for (size_t target = 0; target < n; target++) {
        copy_row = (*this)[target];
        for (size_t pivot = 0; pivot < target; pivot++) {
            if (copy_row[pivot]) {
                copy_row ^= (*this)[pivot];
                result[target] ^= result[pivot];
            }
        }
    }
    return result;
}

/// Lets say that a word has W bits and we have a matrix
///
/// A11  A12  ...  A1n
/// A21  A22  ...  A2n
///           ...
/// An1  An2  ...  Ann
///
/// Aij consists of W words (W * W bits), and it is stored in
/// memory with stride n * W bits. In order to do a bitwise
/// transpose, we need to first bitwise transpose each Aij and then
/// swap Aij with Aji. We have inplace_transpose_square(bitword<W>
/// *data, size_t stride) which handles the first part and std::swap
/// which handles the second part.
///
/// Now, suppose that we are in a situation without major axis
/// padding. We have a matrix which looks like this
///
/// A11  A12  ...  A1n
/// A21  A22  ...  A2n
///           ...
/// Bn1  Bn2  ...  Bnn
///
/// where Aij still consists of W words, but Bnj might have fewer. Now
/// we proceed as follows: for the Aij blocks, transpose them inplace
/// using inplace_transpose_square. Allocate an array C of W words on
/// the stack (and zero it out). Swaps that don't involve the last row
/// can happen in place as before. For swapping Bnj and Ajn, copy Bnj
/// into C. Call inplace_transpose_square on C (with stride 0). Swap C
/// and Ajn, and then copy the required rows from C back into Bnj. For
/// Bnn we just copy it to C, transpose it and then copy the required
/// rows back into Bnn.


/// return the pointer to the first row of an simd block.
template <size_t W>
inline bitword<W> *simd_bit_table<W>::block_start(size_t maj, size_t min) const {
    return data.ptr_simd + maj * W * num_minor_simd_padded() + min;
}


/// zero out a W x W bit array
template <size_t W>
void clear_bitword_array(bitword<W> *c) {
    for (size_t row = 0; row < W; row++) {
        (*(c + row))^(*(c + row));
    }
}

template <size_t W>
void simd_bit_table<W>::do_square_transpose() {
    assert(num_bits_minor == num_bits_major);

    // transpose the blocks Aij
    // if we are padded in both axes, this transposes all blocks.
    // if we are not padded in the major axis, we miss the last row.
    for (size_t maj = 0; maj < num_major_simd_padded().number_of_words; maj++) {
        for (size_t min = 0; min < num_minor_simd_padded(); min++) {
            bitword<W> *block = block_start(maj, min);
            bitword<W>::inplace_transpose_square(block, num_minor_simd_padded());
        }
    }

    // swap Aij with Aji.
    // if we are padded in both axes, this swaps all required blocks.
    // if we are note padded in the major axis, we need to handle the last row and column
    for (size_t maj = 0; maj < num_major_simd_padded().number_of_words; maj++) {
        for (size_t min = maj + 1; min < num_minor_simd_padded(); min++) {
            for (size_t row = 0; row < W; row++) {
                std::swap(
                    *(block_start(maj,min) + row * num_minor_simd_padded()),
                    *(block_start(min,maj) + row * num_minor_simd_padded())
                );
            }
        }
    }

    if (! is_major_padded()) {
        size_t maj = num_major_simd_padded().number_of_words;

        // handle Bnn
        {
            bitword<W> c[W];
            clear_bitword_array(c);
            bitword<W> *block = block_start(maj, maj);
            for (size_t row = 0; row < num_major_simd_padded().remaining_bits; row++) {
                c[row] = block[row * num_minor_simd_padded()];
            }
            bitword<W>::inplace_transpose_square(c,0);
            for (size_t row = 0; row < num_major_simd_padded().remaining_bits; row++) {
                block[row * num_minor_simd_padded()] = c[row];
            }
        }

        // handle Bni
        for (size_t min = 0; min < num_major_simd_padded().number_of_words; min++) {
            bitword<W> c[W];
            clear_bitword_array(c);
            bitword<W> *block_bottom = block_start(maj, min);
            bitword<W> *block_right = block_start(min, maj);

            // copy Bni into C
            for (size_t row = 0; row < num_major_simd_padded().remaining_bits; row++) {
                c[row] = block_bottom[row * num_minor_simd_padded()];
            }

            // transpose C
            bitword<W>::inplace_transpose_square(c,0);

            // swap C and Ain
            for (size_t row = 0; row < W; row++) {
                std::swap(
                    *(block_right + row * num_minor_simd_padded()),
                    *(c + row)
                );
            }

            // copy top rows of C into Bni
            for (size_t row = 0; row < num_major_simd_padded().remaining_bits; row++) {
                block_bottom[row * num_minor_simd_padded()] = c[row];
            }
        }
    }
}

template <size_t W>
simd_bit_table<W> simd_bit_table<W>::transposed() const {
    simd_bit_table<W> result(num_minor_bits_padded(), num_major_bits_padded());
    transpose_into(result);
    return result;
}

template <size_t W>
simd_bit_table<W> simd_bit_table<W>::slice_maj(size_t maj_start_bit, size_t maj_stop_bit) const {
    simd_bit_table<W> result(maj_stop_bit - maj_start_bit, num_minor_bits_padded());
    for (size_t k = maj_start_bit; k < maj_stop_bit; k++) {
        result[k - maj_start_bit] = (*this)[k];
    }
    return result;
}

template <size_t W>
void simd_bit_table<W>::transpose_into(simd_bit_table<W> &out) const {
    assert(is_major_padded() && out.is_major_padded());
    assert(out.num_minor_simd_padded() == num_major_simd_padded().number_of_words);
    assert(out.num_major_simd_padded().number_of_words == num_minor_simd_padded());

    size_t num_simd_words_major = num_major_simd_padded().number_of_words;
    size_t num_simd_words_minor = num_minor_simd_padded();
    for (size_t maj_high = 0; maj_high < num_simd_words_major; maj_high++) {
        for (size_t min_high = 0; min_high < num_simd_words_minor; min_high++) {
            for (size_t maj_low = 0; maj_low < W; maj_low++) {
                bitword<W> *src = block_start(maj_high, min_high) + maj_low;
                bitword<W> *dst = block_start(min_high, maj_high) + maj_low;
                *dst = *src;
            }
        }
    }

    for (size_t maj = 0; maj < out.num_major_simd().number_of_words; maj++) {
        for (size_t min = 0; min < out.num_minor_simd_padded(); min++) {
            bitword<W> *block = out.block_start(maj, min);
            bitword<W>::inplace_transpose_square(block, num_minor_simd_padded());
        }
    }
}

template <size_t W>
simd_bit_table<W> simd_bit_table<W>::from_quadrants(
    size_t n,
    const simd_bit_table<W> &upper_left,
    const simd_bit_table<W> &upper_right,
    const simd_bit_table<W> &lower_left,
    const simd_bit_table<W> &lower_right) {
    assert(upper_left.num_minor_bits_padded() >= n && upper_left.num_major_bits_padded() >= n);
    assert(upper_right.num_minor_bits_padded() >= n && upper_right.num_major_bits_padded() >= n);
    assert(lower_left.num_minor_bits_padded() >= n && lower_left.num_major_bits_padded() >= n);
    assert(lower_right.num_minor_bits_padded() >= n && lower_right.num_major_bits_padded() >= n);

    simd_bit_table<W> result(n << 1, n << 1);
    for (size_t row = 0; row < n; row++) {
        for (size_t col = 0; col < n; col++) {
            result[row][col] = upper_left[row][col];
            result[row][col + n] = upper_right[row][col];
            result[row + n][col] = lower_left[row][col];
            result[row + n][col + n] = lower_right[row][col];
        }
    }
    return result;
}

template <size_t W>
std::string simd_bit_table<W>::str(size_t rows, size_t cols) const {
    std::stringstream out;
    for (size_t row = 0; row < rows; row++) {
        if (row) {
            out << "\n";
        }
        for (size_t col = 0; col < cols; col++) {
            out << ".1"[(*this)[row][col]];
        }
    }
    return out.str();
}

template <size_t W>
std::string simd_bit_table<W>::str(size_t n) const {
    return str(n, n);
}

template <size_t W>
std::string simd_bit_table<W>::str() const {
    return str(num_major_bits_padded(), num_minor_bits_padded());
}

template <size_t W>
simd_bit_table<W> simd_bit_table<W>::from_text(const char *text, size_t min_rows, size_t min_cols) {
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

    size_t num_cols = min_cols;
    for (const auto &v : lines) {
        num_cols = std::max(v.size(), num_cols);
    }
    size_t num_rows = std::max(min_rows, lines.size());
    simd_bit_table<W> out(num_rows, num_cols);
    for (size_t row = 0; row < lines.size(); row++) {
        for (size_t col = 0; col < lines[row].size(); col++) {
            out[row][col] = lines[row][col];
        }
    }

    return out;
}

template <size_t W>
simd_bit_table<W> simd_bit_table<W>::random(
    size_t num_randomized_major_bits, size_t num_randomized_minor_bits, std::mt19937_64 &rng) {
    simd_bit_table<W> result(num_randomized_major_bits, num_randomized_minor_bits);
    for (size_t maj = 0; maj < num_randomized_major_bits; maj++) {
        result[maj].randomize(num_randomized_minor_bits, rng);
    }
    return result;
}

template <size_t W>
std::ostream &operator<<(std::ostream &out, const stim::simd_bit_table<W> &v) {
    for (size_t k = 0; k < v.num_major_bits_padded(); k++) {
        if (k) {
            out << '\n';
        }
        out << v[k];
    }
    return out;
}

}
