/*
 * Copyright 2021 Google LLC
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

#ifndef _STIM_MEM_SIMD_BIT_TABLE_H
#define _STIM_MEM_SIMD_BIT_TABLE_H

#include "stim/mem/simd_bits.h"
#include <cassert>

namespace stim {

/// struct for storing row size relative to some word size
struct relative_size {
    size_t number_of_words;
    size_t remaining_bits;
};

/// A 2d array of bit-packed booleans, padded and aligned to make simd operations more efficient.
///
/// The table contents are indexed by a major axis (not contiguous in memory) then a minor axis (contiguous in memory).
///
/// Note that, due to the padding, the smallest table you can have is 256x256 bits (8 KiB).
/// Technically the padding of the major axis is not necessary, but it's included so that transposing preserves size.
///
/// Similar to simd_bits, simd_bit_table has no notion of the "intended" size of data, only the padded size. The
/// intended size has to be stored separately.
template <size_t W>
struct simd_bit_table {
private:
    size_t num_bits_major_padded;
    size_t num_bits_minor_padded; // always divisble by W
    size_t num_bits_major;
    size_t num_bits_minor;

public:
    simd_bits<W> data;

    /// return pointer to first row of an simd square
    inline bitword<W> *block_start(size_t maj, size_t min) const;

    /// Creates zero initialized table.
    simd_bit_table(size_t num_bits_major, size_t num_bits_minor);
    /// Creates a randomly initialized table.
    static simd_bit_table random(
        size_t num_randomized_major_bits, size_t num_randomized_minor_bits, std::mt19937_64 &rng);
    /// Creates a square table with 1s down the diagonal.
    static simd_bit_table identity(size_t n);
    /// Concatenates tables together to form a larger table.
    static simd_bit_table from_quadrants(
        size_t n,
        const simd_bit_table &upper_left,
        const simd_bit_table &upper_right,
        const simd_bit_table &lower_left,
        const simd_bit_table &lower_right);
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
    static simd_bit_table from_text(const char *text, size_t min_rows = 0, size_t min_cols = 0);

    /// Equality.
    bool operator==(const simd_bit_table &other) const;
    /// Inequality.
    bool operator!=(const simd_bit_table &other) const;

    /// Returns a reference to a row (column) of the table, when using row (column) major indexing.
    inline simd_bits_range_ref<W> operator[](size_t major_index) {
        return data.word_range_ref(major_index * num_minor_simd_padded(), num_minor_simd_padded());
    }
    /// Returns a const reference to a row (column) of the table, when using row (column) major indexing.
    inline const simd_bits_range_ref<W> operator[](size_t major_index) const {
        return data.word_range_ref(major_index * num_minor_simd_padded(), num_minor_simd_padded());
    }

    /// Square matrix multiplication (assumes row major indexing). n is the diameter of the matrix.
    simd_bit_table square_mat_mul(const simd_bit_table &rhs, size_t n) const;
    /// Square matrix inverse, assuming input is lower triangular. n is the diameter of the matrix.
    simd_bit_table inverse_assuming_lower_triangular(size_t n) const;
    /// Transposes the table inplace.
    void do_square_transpose();
    /// Transposes the table out of place into a target location.
    void transpose_into(simd_bit_table &out) const;
    /// Transposes the table out of place.
    simd_bit_table transposed() const;
    /// Returns a subset of the table.
    simd_bit_table slice_maj(size_t maj_start_bit, size_t maj_stop_bit) const;

    /// Sets all bits in the table to zero.
    void clear();

    /// Number of simd words in a column (row) + remaining bits assuming row (column) major indexing.
    inline relative_size num_major_simd() const {
        size_t d = W;
        relative_size result;
        result.number_of_words = num_bits_major / d;
        result.remaining_bits = num_bits_major % d;
        return result;
    }
    /// Number of 64 bit words in a column (row) + remaining bits assuming row (column) major indexing.
    inline relative_size num_major_u64() const {
        size_t d = 8 * sizeof(uint64_t);
        relative_size result;
        result.number_of_words = num_bits_major / d;
        result.remaining_bits = num_bits_major % d;
        return result;
    }
    /// Number of 32 bit words in a column (row) + remaining bits assuming row (column) major indexing.
    inline relative_size num_major_u32() const {
        size_t d = 8 * sizeof(uint32_t);
        relative_size result;
        result.number_of_words = num_bits_major / d;
        result.remaining_bits = num_bits_major % d;
        return result;
    }
    /// Number of 16 bit words in a column (row) + remaining bits assuming row (column) major indexing.
    inline relative_size num_major_u16() const {
        size_t d = 8 * sizeof(uint16_t);
        relative_size result;
        result.number_of_words = num_bits_major / d;
        result.remaining_bits = num_bits_major % d;
        return result;
    }
    /// Number of 8 bit words in a column (row) + remaining bits assuming row (column) major indexing.
    inline relative_size num_major_u8() const {
        size_t d = 8 * sizeof(uint8_t);
        relative_size result;
        result.number_of_words = num_bits_major / d;
        result.remaining_bits = num_bits_major % d;
        return result;
    }
    /// Number of bits in a column (row) assuming row (column) major indexing.
    inline size_t num_major_bits() const {
        return num_bits_major;
    }

    /// Number of simd words in a row (column) + remaining bits assuming row (column) major indexing.
    inline relative_size num_minor_simd() const {
        size_t d = W;
        relative_size result;
        result.number_of_words = num_bits_minor / d;
        result.remaining_bits = num_bits_minor % d;
        return result;
    }
    /// Number of 64 bit words in a row (column) + remaining bits assuming row (column) major indexing.
    inline relative_size num_minor_u64() const {
        size_t d = 8 * sizeof(uint64_t);
        relative_size result;
        result.number_of_words = num_bits_minor / d;
        result.remaining_bits = num_bits_minor % d;
        return result;
    }
    /// Number of 32 bit words in a row (column) + remaining bits assuming row (column) major indexing.
    inline relative_size num_minor_u32() const {
        size_t d = 8 * sizeof(uint32_t);
        relative_size result;
        result.number_of_words = num_bits_minor / d;
        result.remaining_bits = num_bits_minor % d;
        return result;

    }
    /// Number of 16 bit words in a row (column) + remaining bits assuming row (column) major indexing.
    inline relative_size num_minor_u16() const {
        size_t d = 8 * sizeof(uint16_t);
        relative_size result;
        result.number_of_words = num_bits_minor / d;
        result.remaining_bits = num_bits_minor % d;
        return result;

    }
    /// Number of 8 bit words in a row (column) + remaining bits assuming row (column) major indexing.
    inline relative_size num_minor_u8() const {
        size_t d = 8 * sizeof(uint8_t);
        relative_size result;
        result.number_of_words = num_bits_minor / d;
        result.remaining_bits = num_bits_minor % d;
        return result;

    }
    /// Number of bits in a row (column) assuming row (column) major indexing.
    inline size_t num_minor_bits() const {
        return num_bits_minor;
    }


    /// Number of simd words in a padded column (row) + remaining bits assuming row (column) major indexing.
    inline relative_size num_major_simd_padded() const {
        size_t d = W;
        relative_size result;
        result.number_of_words = num_bits_major_padded / d;
        result.remaining_bits = num_bits_major_padded % d;
        return result;
    }
    /// Number of 64 bit words in a padded column (row) + remaining bits assuming row (column) major indexing.
    inline relative_size num_major_u64_padded() const {
        size_t d = 8 * sizeof(uint64_t);
        relative_size result;
        result.number_of_words = num_bits_major_padded / d;
        result.remaining_bits = num_bits_major_padded % d;
        return result;
    }
    /// Number of 32 bit words in a padded column (row) + remaining bits assuming row (column) major indexing.
    inline relative_size num_major_u32_padded() const {
        size_t d = 8 * sizeof(uint32_t);
        relative_size result;
        result.number_of_words = num_bits_major_padded / d;
        result.remaining_bits = num_bits_major_padded % d;
        return result;
    }
    /// Number of 16 bit words in a padded column (row) + remaining bits assuming row (column) major indexing.
    inline relative_size num_major_u16_padded() const {
        size_t d = 8 * sizeof(uint16_t);
        relative_size result;
        result.number_of_words = num_bits_major_padded / d;
        result.remaining_bits = num_bits_major_padded % d;
        return result;
    }
    /// Number of 8 bit words in a padded column (row) + remaining bits assuming row (column) major indexing.
    inline relative_size num_major_u8_padded() const {
        size_t d = 8 * sizeof(uint8_t);
        relative_size result;
        result.number_of_words = num_bits_major_padded / d;
        result.remaining_bits = num_bits_major_padded % d;
        return result;
    }
    /// Number of bits in a padded column (row) assuming row (column) major indexing.
    inline size_t num_major_bits_padded() const {
        return num_bits_major_padded;
    }

    /// Number of simd words in a padded row (column) assuming row (column) major indexing.
    inline size_t num_minor_simd_padded() const {
        return num_bits_minor_padded / W;
    }
    /// Number of 64 bit words in a padded row (column) assuming row (column) major indexing.
    inline size_t num_minor_u64_padded() const {
        return num_bits_minor_padded / (8 * sizeof(uint64_t));
    }
    /// Number of 32 bit words in a padded row (column) assuming row (column) major indexing.
    inline size_t num_minor_u32_padded() const {
        return num_bits_minor_padded / (8 * sizeof(uint32_t));
    }
    /// Number of 16 bit words in a padded row (column) assuming row (column) major indexing.
    inline size_t num_minor_u16_padded() const {
        return num_bits_minor_padded / (8 * sizeof(uint16_t));
    }
    /// Number of 8 bit words in a padded row (column) assuming row (column) major indexing.
    inline size_t num_minor_u8_padded() const {
        return num_bits_minor_padded / (8 * sizeof(uint8_t));
    }
    /// Number of bits in a padded row (column) assuming row (column) major indexing.
    inline size_t num_minor_bits_padded() const {
        return num_bits_minor_padded;
    }

    /// Returns a padded description of the table's contents.
    std::string str() const;
    /// Returns a truncated square description of the table's contents.
    std::string str(size_t n) const;
    /// Returns a truncated rectangle description of the table's contents.
    std::string str(size_t rows, size_t cols) const;
};

template <size_t W>
std::ostream &operator<<(std::ostream &out, const simd_bit_table<W> &v);

constexpr uint8_t lg(size_t k) {
    return k <= 1 ? 0 : lg(k >> 1) + 1;
}

}  // namespace stim

// #include "stim/mem/simd_bit_table.inl"

#endif
