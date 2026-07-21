// bw24-native CPU implementation of one routed MoE token for the Hy3 CPU/GPU expert split.
// This translation unit is self-contained: it owns the packed-format decoders, activation
// quantizer, SIMD dot products, storage pipeline, and stable C ABI. No external inference runtime
// is compiled, linked, or loaded.

#include <omp.h>
#include <immintrin.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <exception>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

extern "C" {

struct bw24_cpu_projection_v2 {
    const std::uint8_t * weights;
    std::int32_t qtype;
    std::int32_t in_features;
    std::int32_t out_features;
    std::size_t row_bytes;
    std::size_t byte_len;
    std::int32_t file_fd;
    std::uint64_t file_offset;
    float scale;
};

struct bw24_cpu_expert_v2 {
    bw24_cpu_projection_v2 gate;
    bw24_cpu_projection_v2 up;
    bw24_cpu_projection_v2 down;
    float route_weight;
};

std::uint32_t bw24_cpu_experts_abi_version() {
    return 2;
}

} // extern "C"

namespace {

// Keep these values identical to crates/bw24-engine/src/lib.rs. They are bw24's kernel ABI,
// not identifiers borrowed from another runtime.
enum QuantType : std::int32_t {
    QT_Q8_0 = 0,
    QT_Q4_K = 1,
    QT_Q6_K = 2,
    QT_Q5_K = 3,
    QT_Q3_K = 4,
    QT_IQ4_XS = 5,
    QT_IQ3_S = 6,
    QT_NVFP4 = 7,
    QT_F32 = 8,
    QT_BF16 = 11,
    QT_Q4_0 = 12,
    QT_Q2_K = 13,
};

struct QuantSpec {
    int block;
    int bytes;
    const char * name;
};

QuantSpec quant_spec(std::int32_t qtype) {
    switch (qtype) {
        case QT_Q8_0: return {32, 34, "Q8_0"};
        case QT_Q4_K: return {256, 144, "Q4_K"};
        case QT_Q6_K: return {256, 210, "Q6_K"};
        case QT_Q5_K: return {256, 176, "Q5_K"};
        case QT_Q3_K: return {256, 110, "Q3_K"};
        case QT_IQ4_XS: return {256, 136, "IQ4_XS"};
        case QT_IQ3_S: return {256, 110, "IQ3_S"};
        case QT_NVFP4: return {64, 36, "NVFP4"};
        case QT_F32: return {1, 4, "F32"};
        case QT_BF16: return {1, 2, "BF16"};
        case QT_Q4_0: return {32, 18, "Q4_0"};
        case QT_Q2_K: return {256, 84, "Q2_K"};
        default: throw std::runtime_error("unsupported bw24 CPU qtype " + std::to_string(qtype));
    }
}

struct AlignedBytes {
    struct Free {
        void operator()(void * pointer) const { std::free(pointer); }
    };

    std::unique_ptr<void, Free> storage;
    std::size_t capacity = 0;
    std::size_t alignment = 0;
    void * data = nullptr;

    void resize(std::size_t bytes, std::size_t alignment = 64) {
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
            throw std::runtime_error("alignment must be a power of two");
        }
        if (capacity >= bytes && this->alignment >= alignment) return;
        void * allocation = nullptr;
        const int status = posix_memalign(&allocation, alignment, bytes);
        if (status != 0) {
            throw std::runtime_error(
                "aligned CPU expert allocation failed: " + std::string(std::strerror(status)));
        }
        storage.reset(allocation);
        capacity = bytes;
        this->alignment = alignment;
        data = allocation;
    }

    std::size_t size() const { return capacity; }
};

#include "bw24_iq3s_grid.inc"

constexpr std::array<std::array<std::int8_t, 8>, 256> make_iq3_signs() {
    std::array<std::array<std::int8_t, 8>, 256> signs {};
    for (int mask = 0; mask < 256; ++mask) {
        for (int lane = 0; lane < 8; ++lane) {
            signs[mask][lane] = (mask & (1 << lane)) != 0 ? -1 : 1;
        }
    }
    return signs;
}

alignas(16) constexpr auto BW24_IQ3S_SIGNS = make_iq3_signs();

float fp16_to_f32(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000) << 16;
    const std::uint32_t exp = (h >> 10) & 0x1f;
    const std::uint32_t mantissa = h & 0x03ff;
    std::uint32_t bits = 0;
    if (exp == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            std::uint32_t m = mantissa;
            std::uint32_t shift = 0;
            while ((m & 0x0400) == 0) {
                m <<= 1;
                ++shift;
            }
            m &= 0x03ff;
            bits = sign | ((127 - 14 - shift) << 23) | (m << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000 | (mantissa << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float bf16_to_f32(const std::uint8_t * bytes) {
    std::uint16_t value;
    std::memcpy(&value, bytes, sizeof(value));
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::uint16_t read_u16(const std::uint8_t * bytes) {
    std::uint16_t value;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

std::uint32_t read_u32(const std::uint8_t * bytes) {
    std::uint32_t value;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

float ue4m3_to_f32(std::uint8_t value) {
    if (value == 0 || value == 0x7f) return 0.0f;
    const int exponent = (value >> 3) & 0x0f;
    const float mantissa = static_cast<float>(value & 7);
    const float decoded = exponent == 0
        ? std::ldexp(mantissa, -9)
        : std::ldexp(1.0f + mantissa / 8.0f, exponent - 7);
    return 0.5f * decoded;
}

struct alignas(32) Q8Block16 {
    float scale = 0.0f;
    std::int32_t sum = 0;
    alignas(16) std::int8_t values[16] {};
};

struct QuantizedActivation {
    std::vector<Q8Block16> blocks;

    void prepare(int count) {
        if (count <= 0 || count % 16 != 0) {
            throw std::runtime_error("bw24 CPU activation width must be a positive multiple of 16");
        }
        blocks.resize(static_cast<std::size_t>(count / 16));
    }

    bool quantize(const float * input, int count) noexcept {
        if (input == nullptr || count <= 0 || count % 16 != 0
            || blocks.size() != static_cast<std::size_t>(count / 16)) {
            return false;
        }
        bool finite = true;
        for (std::size_t block_index = 0; block_index < blocks.size(); ++block_index) {
            auto & block = blocks[block_index];
            const float * values = input + block_index * 16;
            float absolute_max = 0.0f;
            for (int index = 0; index < 16; ++index) {
                if (std::isfinite(values[index])) {
                    absolute_max = std::max(absolute_max, std::abs(values[index]));
                } else {
                    finite = false;
                }
            }
            block.scale = absolute_max == 0.0f ? 0.0f : absolute_max / 127.0f;
            block.sum = 0;
            for (int index = 0; index < 16; ++index) {
                float rounded = 0.0f;
                if (block.scale != 0.0f && std::isfinite(values[index])) {
                    rounded = std::nearbyint(values[index] / block.scale);
                }
                const int quantized = static_cast<int>(
                    std::clamp(rounded, -127.0f, 127.0f));
                block.values[index] = static_cast<std::int8_t>(quantized);
                block.sum += quantized;
            }
        }
        return finite;
    }
};

std::int32_t dot_i8_16(
        const std::int8_t * left, const std::int8_t * right, std::int32_t right_sum) {
#if defined(__AVXVNNI__)
    const __m128i weights = _mm_loadu_si128(reinterpret_cast<const __m128i *>(left));
    const __m128i activations = _mm_loadu_si128(reinterpret_cast<const __m128i *>(right));
    const __m128i biased = _mm_xor_si128(weights, _mm_set1_epi8(static_cast<char>(0x80)));
    __m128i sums = _mm_dpbusd_epi32(_mm_setzero_si128(), biased, activations);
    sums = _mm_hadd_epi32(sums, sums);
    sums = _mm_hadd_epi32(sums, sums);
    return _mm_cvtsi128_si32(sums) - 128 * right_sum;
#elif defined(__AVX2__)
    const __m128i left8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(left));
    const __m128i right8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(right));
    const __m256i left16 = _mm256_cvtepi8_epi16(left8);
    const __m256i right16 = _mm256_cvtepi8_epi16(right8);
    const __m256i products = _mm256_mullo_epi16(left16, right16);
    const __m256i pairs = _mm256_madd_epi16(products, _mm256_set1_epi16(1));
    const __m128i low = _mm256_castsi256_si128(pairs);
    const __m128i high = _mm256_extracti128_si256(pairs, 1);
    __m128i sum = _mm_add_epi32(low, high);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
#else
    std::int32_t sum = 0;
    for (int index = 0; index < 16; ++index) sum += left[index] * right[index];
    return sum;
#endif
}

#if defined(__SSSE3__)
#if !defined(__AVXVNNI__) || !defined(__AVX2__)
__m128i byte_shift_right(__m128i values, int shift) {
    switch (shift) {
        case 0: return values;
        case 2: return _mm_srli_epi16(values, 2);
        case 4: return _mm_srli_epi16(values, 4);
        case 6: return _mm_srli_epi16(values, 6);
        default: throw std::runtime_error("invalid packed-byte shift");
    }
}
#endif

__m128i unpack_nibbles(const std::uint8_t * values, bool high) {
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(values));
    return _mm_and_si128(
        high ? _mm_srli_epi16(packed, 4) : packed,
        _mm_set1_epi8(15));
}

void store_i8(std::int8_t * destination, __m128i values) {
    _mm_store_si128(reinterpret_cast<__m128i *>(destination), values);
}

#if !defined(__AVXVNNI__) || !defined(__AVX2__)
std::int32_t dot_i8_16(__m128i weights, const Q8Block16 & input) {
#if defined(__AVXVNNI__)
    const __m128i activations = _mm_load_si128(
        reinterpret_cast<const __m128i *>(input.values));
    const __m128i biased = _mm_xor_si128(weights, _mm_set1_epi8(static_cast<char>(0x80)));
    __m128i sums = _mm_dpbusd_epi32(_mm_setzero_si128(), biased, activations);
    sums = _mm_hadd_epi32(sums, sums);
    sums = _mm_hadd_epi32(sums, sums);
    return _mm_cvtsi128_si32(sums) - 128 * input.sum;
#else
    alignas(16) std::int8_t unpacked[16];
    store_i8(unpacked, weights);
    return dot_i8_16(unpacked, input.values, input.sum);
#endif
}
#endif

#if defined(__AVXVNNI__) && defined(__AVX2__)
std::array<std::int32_t, 2> dot_i8_16_pair(
        __m256i weights, const Q8Block16 & low, const Q8Block16 & high) {
    __m256i activations = _mm256_castsi128_si256(
        _mm_load_si128(reinterpret_cast<const __m128i *>(low.values)));
    activations = _mm256_inserti128_si256(
        activations,
        _mm_load_si128(reinterpret_cast<const __m128i *>(high.values)),
        1);
    const __m256i biased = _mm256_xor_si256(
        weights, _mm256_set1_epi8(static_cast<char>(0x80)));
    const __m256i products = _mm256_dpbusd_epi32(
        _mm256_setzero_si256(), biased, activations);
    auto reduce = [](__m128i lanes) {
        lanes = _mm_hadd_epi32(lanes, lanes);
        lanes = _mm_hadd_epi32(lanes, lanes);
        return _mm_cvtsi128_si32(lanes);
    };
    return {
        reduce(_mm256_castsi256_si128(products)) - 128 * low.sum,
        reduce(_mm256_extracti128_si256(products, 1)) - 128 * high.sum,
    };
}
#endif
#endif

float dot_q2_k_row(
        const std::uint8_t * weights,
        const QuantizedActivation & activation,
        int count) {
    float result = 0.0f;
    const int superblocks = count / 256;
    for (int superblock = 0; superblock < superblocks; ++superblock) {
        const std::uint8_t * block = weights + static_cast<std::size_t>(superblock) * 84;
        const std::uint8_t * scales = block;
        const std::uint8_t * quants = block + 16;
        const float d = fp16_to_f32(read_u16(block + 80));
        const float dmin = fp16_to_f32(read_u16(block + 82));
#if defined(__AVXVNNI__) && defined(__AVX2__)
        for (int half = 0; half < 2; ++half) {
            const __m256i packed = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(quants + half * 32));
            for (int pair = 0; pair < 4; ++pair) {
                const int group = half * 8 + pair * 2;
                const __m256i decoded = _mm256_and_si256(
                    _mm256_srli_epi16(packed, pair * 2), _mm256_set1_epi8(3));
                const auto integer_dots = dot_i8_16_pair(
                    decoded,
                    activation.blocks[static_cast<std::size_t>(superblock * 16 + group)],
                    activation.blocks[static_cast<std::size_t>(superblock * 16 + group + 1)]);
                for (int lane = 0; lane < 2; ++lane) {
                    const int current = group + lane;
                    const auto & input = activation.blocks[
                        static_cast<std::size_t>(superblock * 16 + current)];
                    result += input.scale * (
                        d * static_cast<float>(scales[current] & 15) * integer_dots[lane]
                        - dmin * static_cast<float>(scales[current] >> 4) * input.sum);
                }
            }
        }
#else
        for (int group = 0; group < 16; ++group) {
            const int start = group * 16;
            const int half = start / 128;
            const int within = start % 128;
            const auto & input = activation.blocks[
                static_cast<std::size_t>(superblock * 16 + group)];
#if defined(__SSSE3__)
            const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(
                quants + half * 32 + within % 32));
            const __m128i decoded = _mm_and_si128(
                byte_shift_right(packed, 2 * (within / 32)), _mm_set1_epi8(3));
            const int integer_dot = dot_i8_16(decoded, input);
#else
            alignas(16) std::int8_t decoded[16];
            for (int index = 0; index < 16; ++index) {
                decoded[index] = static_cast<std::int8_t>(
                    (quants[half * 32 + (within + index) % 32]
                        >> (2 * (within / 32))) & 3);
            }
            const int integer_dot = dot_i8_16(decoded, input.values, input.sum);
#endif
            result += input.scale * (
                d * static_cast<float>(scales[group] & 15) * integer_dot
                - dmin * static_cast<float>(scales[group] >> 4) * input.sum);
        }
#endif
    }
    return result;
}

float scaled_dot16(
        const std::int8_t * weights,
        const QuantizedActivation & activation,
        std::size_t block,
        float weight_scale) {
    const auto & input = activation.blocks[block];
    return weight_scale * input.scale
        * static_cast<float>(dot_i8_16(weights, input.values, input.sum));
}

float scaled_sum16(
        const QuantizedActivation & activation,
        std::size_t block,
        float weight_scale) {
    const auto & input = activation.blocks[block];
    return weight_scale * input.scale * static_cast<float>(input.sum);
}

void scale_min_k4(int group, const std::uint8_t * scales, int & scale, int & minimum) {
    if (group < 4) {
        scale = scales[group] & 63;
        minimum = scales[group + 4] & 63;
    } else {
        scale = (scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4);
        minimum = (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
    }
}

float dot_quantized_row(
        std::int32_t qtype,
        const std::uint8_t * weights,
        const QuantizedActivation & activation,
        int count) {
    if (qtype == QT_Q2_K) return dot_q2_k_row(weights, activation, count);
    alignas(16) std::int8_t decoded[16];
    float result = 0.0f;
    const auto spec = quant_spec(qtype);
    const int superblocks = count / spec.block;
    const auto block16 = [spec](int superblock, int local) {
        return static_cast<std::size_t>(superblock * (spec.block / 16) + local);
    };

    for (int superblock = 0; superblock < superblocks; ++superblock) {
        const std::uint8_t * block = weights + static_cast<std::size_t>(superblock) * spec.bytes;
        if (qtype == QT_Q8_0 || qtype == QT_Q4_0) {
            const float d = fp16_to_f32(read_u16(block));
            for (int half = 0; half < 2; ++half) {
#if defined(__SSSE3__)
                if (qtype == QT_Q8_0) {
                    store_i8(decoded, _mm_loadu_si128(
                        reinterpret_cast<const __m128i *>(block + 2 + half * 16)));
                } else {
                    store_i8(decoded, _mm_sub_epi8(
                        unpack_nibbles(block + 2, half != 0), _mm_set1_epi8(8)));
                }
#else
                for (int index = 0; index < 16; ++index) {
                    if (qtype == QT_Q8_0) {
                        decoded[index] = static_cast<std::int8_t>(block[2 + half * 16 + index]);
                    } else {
                        const std::uint8_t packed = block[2 + index];
                        decoded[index] = static_cast<std::int8_t>(
                            (half == 0 ? packed & 15 : packed >> 4) - 8);
                    }
                }
#endif
                const std::size_t input_block = static_cast<std::size_t>(
                    superblock * (spec.block / 16) + half);
                result += scaled_dot16(decoded, activation, input_block, d);
            }
            continue;
        }
        if (qtype == QT_NVFP4) {
            static constexpr std::int8_t values[16] =
                {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
            const std::uint8_t * quants = block + 4;
            for (int group = 0; group < 4; ++group) {
#if defined(__SSSE3__)
                const __m128i table = _mm_loadu_si128(reinterpret_cast<const __m128i *>(values));
                const __m128i packed = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i *>(quants + group * 8));
                const __m128i low = _mm_shuffle_epi8(table, _mm_and_si128(packed, _mm_set1_epi8(15)));
                const __m128i high = _mm_shuffle_epi8(
                    table, _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(15)));
                store_i8(decoded, _mm_unpacklo_epi64(low, high));
#else
                for (int index = 0; index < 8; ++index) {
                    const std::uint8_t packed = quants[group * 8 + index];
                    decoded[index] = values[packed & 15];
                    decoded[index + 8] = values[packed >> 4];
                }
#endif
                result += scaled_dot16(
                    decoded, activation, block16(superblock, group), ue4m3_to_f32(block[group]));
            }
            continue;
        }
        if (qtype == QT_Q4_K || qtype == QT_Q5_K) {
            const float d = fp16_to_f32(read_u16(block));
            const float dmin = fp16_to_f32(read_u16(block + 2));
            const std::uint8_t * scales = block + 4;
            const std::uint8_t * high = qtype == QT_Q5_K ? block + 16 : nullptr;
            const std::uint8_t * quants = qtype == QT_Q5_K ? block + 48 : block + 16;
            for (int group = 0; group < 8; ++group) {
                const int pair = group / 2;
                const bool upper_nibble = group % 2 != 0;
                const int high_mask = upper_nibble ? (2 << (2 * pair)) : (1 << (2 * pair));
                for (int half = 0; half < 2; ++half) {
#if defined(__SSSE3__)
                    __m128i values = unpack_nibbles(quants + pair * 32 + half * 16, upper_nibble);
                    if (high != nullptr) {
                        const __m128i high_bytes = _mm_loadu_si128(
                            reinterpret_cast<const __m128i *>(high + half * 16));
                        const __m128i present = _mm_cmpeq_epi8(
                            _mm_and_si128(high_bytes, _mm_set1_epi8(high_mask)),
                            _mm_set1_epi8(high_mask));
                        values = _mm_add_epi8(values, _mm_and_si128(present, _mm_set1_epi8(16)));
                    }
                    store_i8(decoded, values);
#else
                    for (int index = 0; index < 16; ++index) {
                        const int qindex = half * 16 + index;
                        const std::uint8_t packed = quants[pair * 32 + qindex];
                        int value = upper_nibble ? packed >> 4 : packed & 15;
                        if (high != nullptr && (high[qindex] & high_mask) != 0) value += 16;
                        decoded[index] = static_cast<std::int8_t>(value);
                    }
#endif
                    int scale = 0;
                    int minimum = 0;
                    scale_min_k4(group, scales, scale, minimum);
                    const std::size_t input_block = block16(superblock, group * 2 + half);
                    result += scaled_dot16(decoded, activation, input_block, d * scale);
                    result -= scaled_sum16(activation, input_block, dmin * minimum);
                }
            }
            continue;
        }
        if (qtype == QT_Q6_K) {
            const std::uint8_t * low = block;
            const std::uint8_t * high = block + 128;
            const auto * scales = reinterpret_cast<const std::int8_t *>(block + 192);
            const float d = fp16_to_f32(read_u16(block + 208));
            for (int group = 0; group < 16; ++group) {
                const int half128 = group / 8;
                const int within128 = group % 8;
                const int segment = within128 / 2;
                const int lane_half = within128 % 2;
                for (int index = 0; index < 16; ++index) {
                    const int lane = lane_half * 16 + index;
                    const int low_offset = half128 * 64;
                    const int high_offset = half128 * 32;
                    int value = 0;
                    if (segment == 0) value = (low[low_offset + lane] & 15) | ((high[high_offset + lane] & 3) << 4);
                    if (segment == 1) value = (low[low_offset + lane + 32] & 15) | (((high[high_offset + lane] >> 2) & 3) << 4);
                    if (segment == 2) value = (low[low_offset + lane] >> 4) | (((high[high_offset + lane] >> 4) & 3) << 4);
                    if (segment == 3) value = (low[low_offset + lane + 32] >> 4) | (((high[high_offset + lane] >> 6) & 3) << 4);
                    decoded[index] = static_cast<std::int8_t>(value - 32);
                }
                const int scale_index = half128 * 8 + lane_half + segment * 2;
                result += scaled_dot16(
                    decoded, activation, block16(superblock, group), d * scales[scale_index]);
            }
            continue;
        }
        if (qtype == QT_Q3_K) {
            const std::uint8_t * high = block;
            const std::uint8_t * quants = block + 32;
            const std::uint8_t * packed_scales = block + 96;
            const float d = fp16_to_f32(read_u16(block + 108));
            const std::uint32_t aux0 = read_u32(packed_scales);
            const std::uint32_t aux1 = read_u32(packed_scales + 4);
            const std::uint32_t aux2 = read_u32(packed_scales + 8);
            const std::uint32_t words[4] = {
                (aux0 & 0x0f0f0f0fU) | (((aux2 >> 0) & 0x03030303U) << 4),
                (aux1 & 0x0f0f0f0fU) | (((aux2 >> 2) & 0x03030303U) << 4),
                ((aux0 >> 4) & 0x0f0f0f0fU) | (((aux2 >> 4) & 0x03030303U) << 4),
                ((aux1 >> 4) & 0x0f0f0f0fU) | (((aux2 >> 6) & 0x03030303U) << 4),
            };
            std::uint8_t scales[16];
            std::memcpy(scales, words, sizeof(scales));
            for (int group = 0; group < 16; ++group) {
                const int half128 = group / 8;
                const int local = group % 8;
                const int shift = 2 * (local / 2);
                const int qoffset = half128 * 32 + (local % 2) * 16;
                const int high_bit = 1 << (half128 * 4 + local / 2);
                for (int index = 0; index < 16; ++index) {
                    const int high_value = (high[(local % 2) * 16 + index] & high_bit) ? 0 : 4;
                    decoded[index] = static_cast<std::int8_t>(((quants[qoffset + index] >> shift) & 3) - high_value);
                }
                result += scaled_dot16(
                    decoded, activation, block16(superblock, group),
                    d * (static_cast<int>(scales[group]) - 32));
            }
            continue;
        }
        if (qtype == QT_IQ4_XS) {
            static constexpr std::int8_t values[16] =
                {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
            const float d = fp16_to_f32(read_u16(block));
            const std::uint16_t high_scales = read_u16(block + 2);
            const std::uint8_t * low_scales = block + 4;
            const std::uint8_t * quants = block + 8;
            for (int group = 0; group < 8; ++group) {
                const int packed_scale = (low_scales[group / 2] >> (4 * (group % 2))) & 15;
                const int scale = packed_scale | (((high_scales >> (2 * group)) & 3) << 4);
                const std::uint8_t * q = quants + group * 16;
                for (int half = 0; half < 2; ++half) {
#if defined(__SSSE3__)
                    const __m128i table = _mm_loadu_si128(
                        reinterpret_cast<const __m128i *>(values));
                    store_i8(decoded, _mm_shuffle_epi8(table, unpack_nibbles(q, half != 0)));
#else
                    for (int index = 0; index < 16; ++index) {
                        decoded[index] = values[half == 0 ? q[index] & 15 : q[index] >> 4];
                    }
#endif
                    result += scaled_dot16(
                        decoded, activation, block16(superblock, group * 2 + half),
                        d * (scale - 32));
                }
            }
            continue;
        }
        if (qtype == QT_IQ3_S) {
            const float d = fp16_to_f32(read_u16(block));
            const std::uint8_t * quants = block + 2;
            const std::uint8_t * high = block + 66;
            const std::uint8_t * signs = block + 74;
            const std::uint8_t * scales = block + 106;
#if defined(__AVXVNNI__) && defined(__AVX2__)
            // Paired-group path: decode all 32 weights of a group (8 grid lookups, 4 sign
            // bytes) into one 256-bit vector and evaluate both 16-wide activation blocks
            // with a single vpdpbusd. Scale terms apply sequentially in original half
            // order, preserving floating-point accumulation order exactly.
            for (int group = 0; group < 8; ++group) {
                const int scale_nibble = group % 2 == 0
                    ? scales[group / 2] & 15 : scales[group / 2] >> 4;
                const __m256i index_bytes = _mm256_cvtepu8_epi32(
                    _mm_loadl_epi64(reinterpret_cast<const __m128i *>(quants + group * 8)));
                const __m256i high_bits = _mm256_and_si256(
                    _mm256_srlv_epi32(
                        _mm256_set1_epi32(high[group]),
                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)),
                    _mm256_set1_epi32(1));
                const __m256i indices = _mm256_or_si256(
                    index_bytes, _mm256_slli_epi32(high_bits, 8));
                const __m256i grid = _mm256_i32gather_epi32(
                    reinterpret_cast<const int *>(BW24_IQ3S_GRID), indices, 4);
                const __m128i signs_low = _mm_unpacklo_epi64(
                    _mm_loadl_epi64(reinterpret_cast<const __m128i *>(
                        BW24_IQ3S_SIGNS[signs[group * 4]].data())),
                    _mm_loadl_epi64(reinterpret_cast<const __m128i *>(
                        BW24_IQ3S_SIGNS[signs[group * 4 + 1]].data())));
                const __m128i signs_high = _mm_unpacklo_epi64(
                    _mm_loadl_epi64(reinterpret_cast<const __m128i *>(
                        BW24_IQ3S_SIGNS[signs[group * 4 + 2]].data())),
                    _mm_loadl_epi64(reinterpret_cast<const __m128i *>(
                        BW24_IQ3S_SIGNS[signs[group * 4 + 3]].data())));
                const __m256i directions = _mm256_set_m128i(signs_high, signs_low);
                const __m256i decoded_pair = _mm256_sign_epi8(grid, directions);
                const auto integer_dots = dot_i8_16_pair(
                    decoded_pair,
                    activation.blocks[block16(superblock, group * 2)],
                    activation.blocks[block16(superblock, group * 2 + 1)]);
                for (int lane = 0; lane < 2; ++lane) {
                    const auto & input = activation.blocks[
                        block16(superblock, group * 2 + lane)];
                    result += d * (1 + 2 * scale_nibble) * input.scale
                        * static_cast<float>(integer_dots[lane]);
                }
            }
            continue;
#endif
            for (int group = 0; group < 8; ++group) {
                const int scale_nibble = group % 2 == 0
                    ? scales[group / 2] & 15 : scales[group / 2] >> 4;
                for (int half = 0; half < 2; ++half) {
#if defined(__AVX2__) && defined(__SSSE3__)
                    const int first_chunk = half * 2;
                    const std::uint8_t high_bits = high[group];
                    const std::uint8_t * packed = quants + group * 8 + first_chunk * 2;
                    const __m128i indices = _mm_setr_epi32(
                        packed[0] | (((high_bits >> (first_chunk * 2)) & 1) << 8),
                        packed[1] | (((high_bits >> (first_chunk * 2 + 1)) & 1) << 8),
                        packed[2] | (((high_bits >> (first_chunk * 2 + 2)) & 1) << 8),
                        packed[3] | (((high_bits >> (first_chunk * 2 + 3)) & 1) << 8));
                    const __m128i grid = _mm_i32gather_epi32(
                        reinterpret_cast<const int *>(BW24_IQ3S_GRID), indices, 4);
                    const __m128i directions = _mm_unpacklo_epi64(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i *>(
                            BW24_IQ3S_SIGNS[signs[group * 4 + first_chunk]].data())),
                        _mm_loadl_epi64(reinterpret_cast<const __m128i *>(
                            BW24_IQ3S_SIGNS[signs[group * 4 + first_chunk + 1]].data())));
                    store_i8(decoded, _mm_sign_epi8(grid, directions));
#else
                    for (int local_chunk = 0; local_chunk < 2; ++local_chunk) {
                        const int chunk = half * 2 + local_chunk;
                        const int first_index = quants[group * 8 + chunk * 2]
                            | ((static_cast<int>(high[group]) << (8 - 2 * chunk)) & 256);
                        const int second_index = quants[group * 8 + chunk * 2 + 1]
                            | ((static_cast<int>(high[group]) << (7 - 2 * chunk)) & 256);
                        const std::uint32_t first = BW24_IQ3S_GRID[first_index];
                        const std::uint32_t second = BW24_IQ3S_GRID[second_index];
                        const std::uint8_t sign_bits = signs[group * 4 + chunk];
                        for (int index = 0; index < 4; ++index) {
                            const int first_value = (first >> (8 * index)) & 255;
                            const int second_value = (second >> (8 * index)) & 255;
                            decoded[local_chunk * 8 + index] = static_cast<std::int8_t>(
                                sign_bits & (1 << index) ? -first_value : first_value);
                            decoded[local_chunk * 8 + index + 4] = static_cast<std::int8_t>(
                                sign_bits & (1 << (index + 4)) ? -second_value : second_value);
                        }
                    }
#endif
                    result += scaled_dot16(
                        decoded, activation, block16(superblock, group * 2 + half),
                        d * (1 + 2 * scale_nibble));
                }
            }
            continue;
        }
        throw std::runtime_error("missing bw24 CPU dot implementation for " + std::string(spec.name));
    }
    return result;
}

float dot_row_native(
        std::int32_t qtype,
        const std::uint8_t * weights,
        const float * activation_f32,
        const QuantizedActivation * activation_q8,
        int count) {
    if (qtype == QT_F32) {
        const float * values = reinterpret_cast<const float *>(weights);
        float result = 0.0f;
#pragma omp simd reduction(+:result)
        for (int index = 0; index < count; ++index) result += values[index] * activation_f32[index];
        return result;
    }
    if (qtype == QT_BF16) {
        float result = 0.0f;
#pragma omp simd reduction(+:result)
        for (int index = 0; index < count; ++index) {
            result += bf16_to_f32(weights + index * 2) * activation_f32[index];
        }
        return result;
    }
    if (activation_q8 == nullptr) throw std::runtime_error("missing bw24 CPU q8 activation");
    return dot_quantized_row(qtype, weights, *activation_q8, count);
}

struct InodeKey {
    std::uint64_t device = 0;
    std::uint64_t inode = 0;

    bool operator==(const InodeKey & other) const {
        return device == other.device && inode == other.inode;
    }
};

struct InodeKeyHash {
    std::size_t operator()(const InodeKey & key) const {
        std::size_t value = std::hash<std::uint64_t>{}(key.device);
        value ^= std::hash<std::uint64_t>{}(key.inode)
            + 0x9e3779b9 + (value << 6) + (value >> 2);
        return value;
    }
};

struct FileKey {
    InodeKey inode;
    std::uint64_t size = 0;
    std::int64_t ctime_seconds = 0;
    std::int64_t ctime_nanoseconds = 0;

    bool operator==(const FileKey & other) const {
        return inode == other.inode && size == other.size
            && ctime_seconds == other.ctime_seconds
            && ctime_nanoseconds == other.ctime_nanoseconds;
    }
};

struct FileKeyHash {
    std::size_t operator()(const FileKey & key) const {
        std::size_t value = InodeKeyHash {}(key.inode);
        const auto combine = [&value](std::size_t field) {
            value ^= field + 0x9e3779b9 + (value << 6) + (value >> 2);
        };
        combine(std::hash<std::uint64_t> {}(key.size));
        combine(std::hash<std::int64_t> {}(key.ctime_seconds));
        combine(std::hash<std::int64_t> {}(key.ctime_nanoseconds));
        return value;
    }
};

InodeKey inode_key(int fd) {
    struct stat value {};
    if (fstat(fd, &value) != 0) {
        throw std::runtime_error(
            "cannot stat CPU expert source fd: " + std::string(std::strerror(errno)));
    }
    return InodeKey {
        static_cast<std::uint64_t>(value.st_dev),
        static_cast<std::uint64_t>(value.st_ino),
    };
}

FileKey file_key(int fd) {
    struct stat value {};
    if (fstat(fd, &value) != 0) {
        throw std::runtime_error(
            "cannot stat CPU expert source fd: " + std::string(std::strerror(errno)));
    }
    return FileKey {
        InodeKey {
            static_cast<std::uint64_t>(value.st_dev),
            static_cast<std::uint64_t>(value.st_ino),
        },
        static_cast<std::uint64_t>(value.st_size),
        static_cast<std::int64_t>(value.st_ctim.tv_sec),
        static_cast<std::int64_t>(value.st_ctim.tv_nsec),
    };
}

struct CacheKey {
    FileKey file;
    std::uint64_t offset = 0;
    std::size_t len = 0;

    bool operator==(const CacheKey & other) const {
        return file == other.file && offset == other.offset && len == other.len;
    }
};

struct CacheKeyHash {
    std::size_t operator()(const CacheKey & key) const {
        std::size_t value = FileKeyHash{}(key.file);
        value ^= std::hash<std::uint64_t>{}(key.offset)
            + 0x9e3779b9 + (value << 6) + (value >> 2);
        value ^= std::hash<std::size_t>{}(key.len)
            + 0x9e3779b9 + (value << 6) + (value >> 2);
        return value;
    }
};

struct ProjectionRuntime {
    const bw24_cpu_projection_v2 * desc = nullptr;
    const float * activation_f32 = nullptr;
    const struct QuantizedActivation * activation_q8 = nullptr;
    const std::uint8_t * weights = nullptr;
    std::shared_ptr<AlignedBytes> weight_owner;
    bool needs_read = false;
    std::int32_t read_fd = -1;
    std::int32_t alternate_read_fd = -1;
    CacheKey cache_key;
};

struct ExpertRuntime {
    ProjectionRuntime gate;
    ProjectionRuntime up;
    ProjectionRuntime down;
    std::vector<float> gate_output;
    std::vector<float> up_output;
    std::vector<float> activation;
    std::vector<float> down_output;
};

bool direct_io_enabled() {
    static const bool enabled = [] {
        const char * raw = std::getenv("BW24_CPU_EXPERT_IO");
        if (raw == nullptr || std::strcmp(raw, "buffered") == 0) return false;
        if (std::strcmp(raw, "direct") == 0) return true;
        throw std::runtime_error(std::string("invalid BW24_CPU_EXPERT_IO=") + raw
            + " (expected buffered or direct)");
    }();
    return enabled;
}

int io_thread_count(int compute_threads) {
    const char * raw = std::getenv("BW24_CPU_EXPERT_IO_THREADS");
    if (raw == nullptr || *raw == '\0') return compute_threads;
    char * end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || value < 1 || value > 256) {
        throw std::runtime_error(std::string("invalid BW24_CPU_EXPERT_IO_THREADS=") + raw);
    }
    return static_cast<int>(value);
}

class DirectFiles {
public:
    ~DirectFiles() {
        for (const auto & [_, fd] : files_) close(fd);
    }

    int resolve(int source_fd, const InodeKey & identity) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = files_.find(identity);
        if (found != files_.end()) return found->second;
        const std::string path = "/proc/self/fd/" + std::to_string(source_fd);
        const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd < 0) {
            throw std::runtime_error("cannot open O_DIRECT expert source " + path
                + ": " + std::strerror(errno));
        }
        if (!(inode_key(fd) == identity)) {
            close(fd);
            throw std::runtime_error("O_DIRECT expert source changed while opening it");
        }
        files_.emplace(identity, fd);
        return fd;
    }

private:
    std::mutex mutex_;
    std::unordered_map<InodeKey, int, InodeKeyHash> files_;
};

DirectFiles & direct_files() {
    static DirectFiles files;
    return files;
}

class MirrorFiles {
public:
    struct MirrorSpec {
        FileKey source;
        FileKey alternate;
        std::string path;
    };

    struct OpenMirror {
        int fd = -1;
        FileKey generation;
    };

    MirrorFiles() {
        const char * path = std::getenv("BW24_CPU_EXPERT_MIRROR_MAP");
        if (path == nullptr || *path == '\0') return;
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error(std::string("cannot open CPU expert mirror map: ") + path);
        }
        std::string line;
        while (std::getline(input, line)) {
            std::array<std::string, 11> fields;
            std::size_t start = 0;
            for (std::size_t index = 0; index < fields.size() - 1; ++index) {
                const std::size_t end = line.find('\t', start);
                if (end == std::string::npos || end == start) {
                    throw std::runtime_error(
                        "malformed CPU expert mirror map (native runtime requires v2)");
                }
                fields[index] = line.substr(start, end - start);
                start = end + 1;
            }
            if (start == line.size() || line.find('\t', start) != std::string::npos) {
                throw std::runtime_error(
                    "malformed CPU expert mirror map (native runtime requires v2)");
            }
            fields.back() = line.substr(start);
            const auto parse_u64 = [](const std::string & value) {
                std::size_t end = 0;
                const auto parsed = std::stoull(value, &end);
                if (end != value.size()) {
                    throw std::runtime_error("malformed CPU expert mirror-map generation");
                }
                return static_cast<std::uint64_t>(parsed);
            };
            const auto parse_i64 = [](const std::string & value) {
                std::size_t end = 0;
                const auto parsed = std::stoll(value, &end);
                if (end != value.size()) {
                    throw std::runtime_error("malformed CPU expert mirror-map generation");
                }
                return static_cast<std::int64_t>(parsed);
            };
            const MirrorSpec spec {
                FileKey {
                    InodeKey { parse_u64(fields[0]), parse_u64(fields[1]) },
                    parse_u64(fields[2]), parse_i64(fields[3]), parse_i64(fields[4]),
                },
                FileKey {
                    InodeKey { parse_u64(fields[5]), parse_u64(fields[6]) },
                    parse_u64(fields[7]), parse_i64(fields[8]), parse_i64(fields[9]),
                },
                fields[10],
            };
            const auto inserted = paths_.emplace(spec.source.inode, spec);
            if (!inserted.second
                && (!(inserted.first->second.source == spec.source)
                    || !(inserted.first->second.alternate == spec.alternate)
                    || inserted.first->second.path != spec.path)) {
                throw std::runtime_error("conflicting CPU expert mirror-map inode");
            }
        }
        if (input.bad()) throw std::runtime_error("cannot read CPU expert mirror map");
        if (paths_.empty()) throw std::runtime_error("CPU expert mirror map is empty");
        std::fprintf(stderr, "[bw24-cpu] mirrored direct I/O: %zu inode mappings\n", paths_.size());
    }

    ~MirrorFiles() {
        for (const auto & [_, mirror] : files_) close(mirror.fd);
    }

    int resolve(int source_fd, const FileKey & source) {
        if (paths_.empty()) return -1;
        std::lock_guard<std::mutex> lock(mutex_);
        const FileKey current_source = file_key(source_fd);
        if (!(current_source == source)) {
            throw std::runtime_error("CPU expert source changed while resolving its mirror");
        }
        const auto spec = paths_.find(source.inode);
        if (spec == paths_.end()) {
            throw std::runtime_error("CPU expert source inode is absent from mirror map");
        }
        if (!(spec->second.source == source)) {
            throw std::runtime_error("CPU expert source generation differs from mirror map");
        }
        const auto cached = files_.find(source);
        if (cached != files_.end()) {
            if (!(file_key(cached->second.fd) == cached->second.generation)) {
                throw std::runtime_error("CPU expert mirror generation changed after opening");
            }
            return cached->second.fd;
        }
        const int alternate = open(
            spec->second.path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (alternate < 0) {
            throw std::runtime_error(
                "cannot open mirrored CPU expert source " + spec->second.path
                + ": " + std::strerror(errno));
        }
        FileKey alternate_generation;
        try {
            alternate_generation = file_key(alternate);
        } catch (...) {
            close(alternate);
            throw;
        }
        if (!(alternate_generation == spec->second.alternate)
            || alternate_generation.size != source.size
            || alternate_generation.inode.device == source.inode.device) {
            close(alternate);
            throw std::runtime_error(
                "CPU expert mirror generation differs from map or physical filesystem");
        }
        files_.emplace(source, OpenMirror { alternate, alternate_generation });
        return alternate;
    }

private:
    std::mutex mutex_;
    std::unordered_map<InodeKey, MirrorSpec, InodeKeyHash> paths_;
    std::unordered_map<FileKey, OpenMirror, FileKeyHash> files_;
};

MirrorFiles & mirror_files() {
    static MirrorFiles files;
    return files;
}

struct CpuProfile {
    std::atomic<std::uint64_t> calls { 0 };
    std::atomic<std::uint64_t> prepare_ns { 0 };
    std::atomic<std::uint64_t> io_ns { 0 };
    std::atomic<std::uint64_t> insert_ns { 0 };
    std::atomic<std::uint64_t> compute_ns { 0 };
    std::atomic<std::uint64_t> read_projections { 0 };
    std::atomic<std::uint64_t> read_bytes { 0 };
    ~CpuProfile() {
        const auto to_seconds = [](std::uint64_t ns) { return ns / 1.0e9; };
        std::fprintf(
            stderr,
            "[bw24-cpu-profile] calls=%llu prepare=%.6fs io=%.6fs insert=%.6fs "
            "compute=%.6fs read_projections=%llu read_GB=%.3f\n",
            static_cast<unsigned long long>(calls.load(std::memory_order_relaxed)),
            to_seconds(prepare_ns.load(std::memory_order_relaxed)),
            to_seconds(io_ns.load(std::memory_order_relaxed)),
            to_seconds(insert_ns.load(std::memory_order_relaxed)),
            to_seconds(compute_ns.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(read_projections.load(std::memory_order_relaxed)),
            read_bytes.load(std::memory_order_relaxed) / 1.0e9);
    }
};

CpuProfile & cpu_profile() {
    static CpuProfile profile;
    return profile;
}

std::uint64_t elapsed_ns(std::chrono::steady_clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count());
}

class WeightCache {
public:
    WeightCache() : budget_(parse_budget()) {
        std::fprintf(stderr, "[bw24-cpu] normal-RAM expert cache: %.2f GiB policy=lru io=%s\n",
            static_cast<double>(budget_) / (1024.0 * 1024.0 * 1024.0),
            direct_io_enabled() ? "direct" : "buffered");
    }

    ~WeightCache() {
        const std::uint64_t accesses = hits_ + misses_;
        const double hit_rate = accesses == 0
            ? 0.0
            : 100.0 * static_cast<double>(hits_) / static_cast<double>(accesses);
        std::fprintf(
            stderr,
            "[bw24-cpu-cache] hits=%llu misses=%llu hit_rate=%.2f%% read_GB=%.3f "
            "resident_GB=%.3f\n",
            static_cast<unsigned long long>(hits_),
            static_cast<unsigned long long>(misses_),
            hit_rate,
            static_cast<double>(read_bytes_) / 1.0e9,
            static_cast<double>(used_) / 1.0e9);
    }

    std::shared_ptr<AlignedBytes> find(const CacheKey & key) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = entries_.find(key);
        if (found == entries_.end()) {
            ++misses_;
            return {};
        }
        ++hits_;
        lru_.splice(lru_.end(), lru_, found->second.lru);
        return found->second.bytes;
    }

    void insert(const CacheKey & key, const std::shared_ptr<AlignedBytes> & bytes) {
        if (budget_ == 0 || bytes->size() > budget_) return;
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = entries_.find(key);
        if (found != entries_.end()) {
            lru_.splice(lru_.end(), lru_, found->second.lru);
            return;
        }
        lru_.push_back(key);
        entries_.emplace(key, Entry {
            bytes,
            std::prev(lru_.end()),
        });
        used_ += key.len;
        read_bytes_ += key.len;
        while (used_ > budget_ && !lru_.empty()) {
            const auto entry = entries_.find(lru_.front());
            if (entry == entries_.end()) {
                throw std::runtime_error("CPU expert cache eviction index is inconsistent");
            }
            used_ -= entry->first.len;
            lru_.erase(entry->second.lru);
            entries_.erase(entry);
        }
    }

    void snapshot(std::uint64_t * hits, std::uint64_t * misses,
                  std::uint64_t * read_bytes, std::uint64_t * resident_bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (hits != nullptr) *hits = hits_;
        if (misses != nullptr) *misses = misses_;
        if (read_bytes != nullptr) *read_bytes = read_bytes_;
        if (resident_bytes != nullptr) *resident_bytes = used_;
    }

private:
    struct Entry {
        std::shared_ptr<AlignedBytes> bytes;
        std::list<CacheKey>::iterator lru;
    };

    using EntryMap = std::unordered_map<CacheKey, Entry, CacheKeyHash>;

    static std::size_t parse_budget() {
        const char * raw = std::getenv("BW24_CPU_EXPERT_CACHE_GB");
        const double requested_gib = [&] {
            if (raw == nullptr || *raw == '\0') return 16.0;
            char * end = nullptr;
            const double value = std::strtod(raw, &end);
            if (end == raw || *end != '\0' || !std::isfinite(value)
                || value < 0.0 || value > 1024.0) {
                throw std::runtime_error(
                    std::string("invalid BW24_CPU_EXPERT_CACHE_GB=") + raw);
            }
            return value;
        }();
        const auto to_bytes = [](double gib) {
            return static_cast<std::size_t>(gib * 1024.0 * 1024.0 * 1024.0);
        };
        const std::size_t requested = to_bytes(requested_gib);

        // Reserve floor DEFAULTS ON (4 GiB) — 2026-07-20 lesson: a run with the reserve
        // unset/0 pinned 36 GiB of a 37.11 GiB MemAvailable, starved the page cache and
        // thrash-locked the desktop into a hard reboot (journald "Under memory pressure",
        // gnome 1.8s input lag; no OOM-kill because cache pages are "reclaimable" — the
        // kernel refaults forever instead of killing). An explicit env value still wins,
        // but 0 now means "0 on top of nothing": the floor guards the DESKTOP, so going
        // below the default requires saying so with a real number.
        constexpr double kDefaultReserveGib = 4.0;
        const char * reserve_raw = std::getenv("BW24_CPU_EXPERT_RESERVE_GB");
        double reserve_gib = kDefaultReserveGib;
        if (reserve_raw != nullptr && *reserve_raw != '\0') {
            char * reserve_end = nullptr;
            reserve_gib = std::strtod(reserve_raw, &reserve_end);
            if (reserve_end == reserve_raw || *reserve_end != '\0' || !std::isfinite(reserve_gib)
                || reserve_gib < 0.0 || reserve_gib > 1024.0) {
                throw std::runtime_error(
                    std::string("invalid BW24_CPU_EXPERT_RESERVE_GB=") + reserve_raw);
            }
        }

        std::ifstream meminfo("/proc/meminfo");
        std::string key;
        std::string unit;
        std::uint64_t value_kib = 0;
        std::uint64_t available_kib = 0;
        while (meminfo >> key >> value_kib >> unit) {
            if (key == "MemAvailable:") {
                if (unit != "kB") {
                    throw std::runtime_error("/proc/meminfo MemAvailable has an unknown unit");
                }
                available_kib = value_kib;
                break;
            }
        }
        if (available_kib == 0) {
            throw std::runtime_error(
                "BW24_CPU_EXPERT_RESERVE_GB requires /proc/meminfo MemAvailable");
        }
        const std::size_t available = static_cast<std::size_t>(available_kib) * 1024;
        const std::size_t reserve = to_bytes(reserve_gib);
        const std::size_t headroom_budget = available > reserve ? available - reserve : 0;
        const std::size_t effective = std::min(requested, headroom_budget);
        std::fprintf(
            stderr,
            "[bw24-cpu] RAM headroom cap: requested=%.2f GiB available=%.2f GiB "
            "reserve=%.2f GiB effective=%.2f GiB\n",
            requested / (1024.0 * 1024.0 * 1024.0),
            available / (1024.0 * 1024.0 * 1024.0),
            reserve / (1024.0 * 1024.0 * 1024.0),
            effective / (1024.0 * 1024.0 * 1024.0));
        if (effective < requested / 2) {
            std::fprintf(stderr,
                "[bw24-cpu] WARNING: headroom cap cut the cache to %.2f GiB (< half of the "
                "requested %.2f GiB) — the box is memory-tight; expect miss-rate degradation\n",
                effective / (1024.0 * 1024.0 * 1024.0),
                requested / (1024.0 * 1024.0 * 1024.0));
        }
        return effective;
    }

    std::size_t budget_ = 0;
    std::size_t used_ = 0;
    std::uint64_t hits_ = 0;
    std::uint64_t misses_ = 0;
    std::uint64_t read_bytes_ = 0;
    std::mutex mutex_;
    std::list<CacheKey> lru_;
    EntryMap entries_;
};

WeightCache & weight_cache() {
    static WeightCache cache;
    return cache;
}

void copy_error(char * dst, std::size_t capacity, const std::string & message) {
    if (dst == nullptr || capacity == 0) return;
    const std::size_t count = std::min(capacity - 1, message.size());
    std::memcpy(dst, message.data(), count);
    dst[count] = '\0';
}

ProjectionRuntime prepare_projection(
        const bw24_cpu_projection_v2 & desc) {
    if ((desc.weights == nullptr && desc.file_fd < 0)
        || desc.in_features <= 0 || desc.out_features <= 0) {
        throw std::runtime_error("invalid CPU expert projection descriptor");
    }
    const auto spec = quant_spec(desc.qtype);
    if (desc.in_features % spec.block != 0) {
        throw std::runtime_error(std::string("CPU expert width is not block-aligned for ") + spec.name);
    }
    const std::size_t expected_row =
        static_cast<std::size_t>(desc.in_features / spec.block) * spec.bytes;
    if (expected_row != desc.row_bytes) {
        throw std::runtime_error(std::string("CPU expert row-size mismatch for ") + spec.name
            + ": descriptor=" + std::to_string(desc.row_bytes)
            + " bw24=" + std::to_string(expected_row));
    }
    const std::size_t expected_bytes = desc.row_bytes * static_cast<std::size_t>(desc.out_features);
    if (desc.byte_len != expected_bytes) {
        throw std::runtime_error(std::string("CPU expert extent mismatch for ")
            + spec.name + ": descriptor=" + std::to_string(desc.byte_len)
            + " expected=" + std::to_string(expected_bytes));
    }
    ProjectionRuntime runtime;
    runtime.desc = &desc;
    if (desc.file_fd >= 0) {
        const FileKey source = file_key(desc.file_fd);
        const CacheKey key { source, desc.file_offset, desc.byte_len };
        runtime.cache_key = key;
        runtime.weight_owner = weight_cache().find(key);
        if (!runtime.weight_owner) {
            runtime.weight_owner = std::make_shared<AlignedBytes>();
            const bool direct = direct_io_enabled()
                && desc.file_offset % 4096 == 0 && desc.byte_len % 4096 == 0;
            runtime.weight_owner->resize(desc.byte_len, direct ? 4096 : 64);
            runtime.read_fd = direct
                ? direct_files().resolve(desc.file_fd, source.inode)
                : desc.file_fd;
            runtime.alternate_read_fd = direct
                ? mirror_files().resolve(desc.file_fd, source)
                : -1;
            runtime.needs_read = true;
        }
        runtime.weights = static_cast<const std::uint8_t *>(runtime.weight_owner->data);
    } else {
        runtime.weights = desc.weights;
    }
    return runtime;
}

int pread_exact(const ProjectionRuntime & projection, int fd, void * destination,
                std::size_t relative_offset, std::size_t length) {
    const auto & desc = *projection.desc;
    std::size_t done = 0;
    auto * bytes = static_cast<std::uint8_t *>(destination);
    while (done < length) {
        const ssize_t count = pread(
            fd,
            bytes + done,
            length - done,
            static_cast<off_t>(desc.file_offset + relative_offset + done));
        if (count > 0) {
            done += static_cast<std::size_t>(count);
        } else if (count == 0) {
            return EIO;
        } else if (errno != EINTR) {
            return errno;
        }
    }
    return 0;
}

struct ReadRequest {
    ProjectionRuntime * projection;
    int fd;
    std::size_t offset;
    std::size_t length;
};

// ---- asynchronous read pipeline -------------------------------------------------------------
// Default path: expert reads are submitted to a persistent io pool and each expert's compute
// starts as soon as its projections land, while later reads are still in flight. Per-expert
// math and the final expert-index-order accumulation are unchanged, so output stays
// byte-identical to the serial path. BW24_CPU_EXPERT_PIPELINE=0 is the rollback seam.

bool pipeline_enabled() {
    static const bool enabled = [] {
        const char * raw = std::getenv("BW24_CPU_EXPERT_PIPELINE");
        return raw == nullptr || *raw == '\0' || std::strcmp(raw, "0") != 0;
    }();
    return enabled;
}

struct CallIoState {
    std::mutex mutex;
    std::condition_variable ready_cv;
    std::vector<int> ready;             // expert indices whose reads all landed
    std::vector<int> pending_requests;  // per expert index, guarded by mutex
    int outstanding_experts = 0;        // experts with reads still in flight
    int first_error = 0;                // first errno observed by any read
};

struct IoJob {
    ProjectionRuntime * projection = nullptr;
    int fd = -1;
    std::size_t offset = 0;
    std::size_t length = 0;
    int expert_index = -1;
    CallIoState * call = nullptr;
};

class IoPool {
public:
    static IoPool & instance() {
        static IoPool pool;
        return pool;
    }

    ~IoPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        for (auto & worker : workers_) worker.join();
    }

    void ensure_started(int threads) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!workers_.empty()) return;
        const int count = std::max(1, threads);
        workers_.reserve(static_cast<std::size_t>(count));
        for (int index = 0; index < count; ++index) {
            workers_.emplace_back([this] { worker_loop(); });
        }
        apply_cpuset_locked();
    }

    void submit(std::vector<IoJob> && jobs) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto & job : jobs) queue_.push_back(job);
        }
        queue_cv_.notify_all();
    }

private:
    // Optional io-thread pinning (e.g. E-cores) so reads stop competing with compute cores.
    // Unset = inherit the process affinity mask.
    void apply_cpuset_locked() {
        const char * raw = std::getenv("BW24_CPU_EXPERT_IO_CPUSET");
        if (raw == nullptr || *raw == '\0') return;
        cpu_set_t set;
        CPU_ZERO(&set);
        const std::string spec(raw);
        std::size_t position = 0;
        while (position < spec.size()) {
            const std::size_t comma = spec.find(',', position);
            const std::string part = spec.substr(
                position, comma == std::string::npos ? std::string::npos : comma - position);
            const std::size_t dash = part.find('-');
            char * end = nullptr;
            const long lo = std::strtol(part.c_str(), &end, 10);
            long hi = lo;
            if (dash != std::string::npos) {
                hi = std::strtol(part.c_str() + dash + 1, &end, 10);
            }
            if (lo < 0 || hi < lo || hi >= CPU_SETSIZE
                || end == part.c_str() || (end != nullptr && *end != '\0')) {
                throw std::runtime_error(
                    std::string("invalid BW24_CPU_EXPERT_IO_CPUSET=") + raw);
            }
            for (long cpu = lo; cpu <= hi; ++cpu) CPU_SET(static_cast<int>(cpu), &set);
            if (comma == std::string::npos) break;
            position = comma + 1;
        }
        for (auto & worker : workers_) {
            pthread_setaffinity_np(worker.native_handle(), sizeof(set), &set);
        }
    }

    void worker_loop() {
        for (;;) {
            IoJob job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                queue_cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
                if (queue_.empty()) return;
                job = queue_.front();
                queue_.pop_front();
            }
            auto * destination = static_cast<std::uint8_t *>(
                job.projection->weight_owner->data) + job.offset;
            const int status = pread_exact(
                *job.projection, job.fd, destination, job.offset, job.length);
            auto * state = job.call;
            {
                std::lock_guard<std::mutex> lock(state->mutex);
                if (status != 0 && state->first_error == 0) state->first_error = status;
                if (--state->pending_requests[job.expert_index] == 0) {
                    --state->outstanding_experts;
                    state->ready.push_back(job.expert_index);
                }
            }
            state->ready_cv.notify_all();
        }
    }

    std::mutex mutex_;
    std::condition_variable queue_cv_;
    std::deque<IoJob> queue_;
    std::vector<std::thread> workers_;
    bool stopping_ = false;
};

// Blocks scope exit until every submitted read for this call has completed, so read buffers
// (owned by the call's runtime vector) can never be written after the call unwinds.
struct IoDrainGuard {
    CallIoState * state = nullptr;

    ~IoDrainGuard() {
        if (state == nullptr) return;
        std::unique_lock<std::mutex> lock(state->mutex);
        state->ready_cv.wait(lock, [this] { return state->outstanding_experts == 0; });
    }
};

void append_projection_jobs(
        std::vector<IoJob> & jobs,
        ProjectionRuntime & projection,
        int expert_index,
        CallIoState & state) {
    if (!projection.needs_read) return;
    const std::size_t length = projection.desc->byte_len;
    int requests = 0;
    if (projection.alternate_read_fd >= 0 && length >= 8192) {
        const std::size_t split = (length / 2) & ~std::size_t(4095);
        jobs.push_back(IoJob { &projection, projection.read_fd, 0, split, expert_index, &state });
        jobs.push_back(IoJob {
            &projection,
            projection.alternate_read_fd,
            split,
            length - split,
            expert_index,
            &state,
        });
        requests = 2;
    } else {
        jobs.push_back(IoJob { &projection, projection.read_fd, 0, length, expert_index, &state });
        requests = 1;
    }
    state.pending_requests[static_cast<std::size_t>(expert_index)] += requests;
}

void load_projection_weights(std::vector<ProjectionRuntime *> & projections, int threads) {
    std::vector<ReadRequest> reads;
    reads.reserve(projections.size() * 2);
    for (auto * projection : projections) {
        if (!projection->needs_read) continue;
        const std::size_t length = projection->desc->byte_len;
        if (projection->alternate_read_fd >= 0 && length >= 8192) {
            const std::size_t split = (length / 2) & ~std::size_t(4095);
            reads.push_back(ReadRequest { projection, projection->read_fd, 0, split });
            reads.push_back(ReadRequest {
                projection,
                projection->alternate_read_fd,
                split,
                length - split,
            });
        } else {
            reads.push_back(ReadRequest { projection, projection->read_fd, 0, length });
        }
    }
    std::vector<int> read_errors(reads.size(), 0);
    auto & profile = cpu_profile();
    const auto io_start = std::chrono::steady_clock::now();
    if (!reads.empty()) {
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
        for (std::size_t index = 0; index < reads.size(); ++index) {
            const auto & read = reads[index];
            auto * destination = static_cast<std::uint8_t *>(
                read.projection->weight_owner->data) + read.offset;
            read_errors[index] = pread_exact(
                *read.projection, read.fd, destination, read.offset, read.length);
        }
    }
    profile.io_ns.fetch_add(elapsed_ns(io_start), std::memory_order_relaxed);

    const auto insert_start = std::chrono::steady_clock::now();
    std::uint64_t invocation_reads = 0;
    std::uint64_t invocation_read_bytes = 0;
    for (std::size_t index = 0; index < reads.size(); ++index) {
        if (read_errors[index] != 0) {
            throw std::runtime_error(
                "CPU expert pread failed at mirrored request " + std::to_string(index)
                + ": " + std::strerror(read_errors[index]));
        }
    }
    for (auto * projection : projections) {
        if (!projection->needs_read) continue;
        const auto & desc = *projection->desc;
        ++invocation_reads;
        invocation_read_bytes += desc.byte_len;
        weight_cache().insert(projection->cache_key, projection->weight_owner);
    }
    profile.insert_ns.fetch_add(elapsed_ns(insert_start), std::memory_order_relaxed);
    profile.read_projections.fetch_add(invocation_reads, std::memory_order_relaxed);
    profile.read_bytes.fetch_add(invocation_read_bytes, std::memory_order_relaxed);
}

void dot_row(const ProjectionRuntime & projection, int row, float * output) {
    const auto & desc = *projection.desc;
    *output = dot_row_native(
        desc.qtype,
        projection.weights + desc.row_bytes * static_cast<std::size_t>(row),
        projection.activation_f32,
        projection.activation_q8,
        desc.in_features);
}

// Runs the full per-expert chain (gate/up dots, SwiGLU, down-activation quantize, down dots)
// for a subset of the call's experts. Row-level math and per-expert op order are identical for
// every subset partition, so pipelined and serial paths produce byte-identical expert outputs.
void compute_experts(
        const std::vector<int> & subset,
        std::vector<ExpertRuntime> & runtime,
        const bw24_cpu_expert_v2 * experts,
        std::vector<QuantizedActivation> & down_activations,
        std::vector<std::uint8_t> & down_activation_finite,
        int n_ff,
        int n_embd,
        int threads) {
    const int n_subset = static_cast<int>(subset.size());
    if (n_subset == 0) return;
#pragma omp parallel num_threads(threads)
    {
#pragma omp for schedule(dynamic, 16)
        for (int task = 0; task < n_subset * n_ff * 2; ++task) {
            const int expert = subset[static_cast<std::size_t>(task / (n_ff * 2))];
            const int local = task % (n_ff * 2);
            const bool is_up = local >= n_ff;
            const int row = local % n_ff;
            auto & work = runtime[static_cast<std::size_t>(expert)];
            if (is_up) {
                dot_row(work.up, row, &work.up_output[row]);
            } else {
                dot_row(work.gate, row, &work.gate_output[row]);
            }
        }

#pragma omp for schedule(static)
        for (int index = 0; index < n_subset * n_ff; ++index) {
            const int expert = subset[static_cast<std::size_t>(index / n_ff)];
            const int column = index % n_ff;
            const auto & desc = experts[expert];
            auto & work = runtime[static_cast<std::size_t>(expert)];
            const float gate = work.gate_output[column] * desc.gate.scale;
            const float up = work.up_output[column] * desc.up.scale;
            work.activation[column] = (gate / (1.0f + std::exp(-gate))) * up;
        }

#pragma omp for schedule(static)
        for (int index = 0; index < n_subset; ++index) {
            const int expert = subset[static_cast<std::size_t>(index)];
            auto & work = runtime[static_cast<std::size_t>(expert)];
            auto & activation = down_activations[static_cast<std::size_t>(expert)];
            down_activation_finite[static_cast<std::size_t>(expert)] =
                static_cast<std::uint8_t>(
                    activation.quantize(work.activation.data(), work.down.desc->in_features));
            work.down.activation_f32 = work.activation.data();
            work.down.activation_q8 = &activation;
        }

#pragma omp for schedule(dynamic, 16)
        for (int task = 0; task < n_subset * n_embd; ++task) {
            const int expert = subset[static_cast<std::size_t>(task / n_embd)];
            const int row = task % n_embd;
            auto & work = runtime[static_cast<std::size_t>(expert)];
            dot_row(work.down, row, &work.down_output[row]);
        }
    }
}

} // namespace

int bw24_cpu_moe_token_impl(
        const bw24_cpu_expert_v2 * experts,
        std::int32_t expert_count,
        const float * input,
        float * output,
        std::int32_t threads,
        char * error,
        std::size_t error_capacity) try {
    if (experts == nullptr || input == nullptr || output == nullptr || expert_count <= 0) {
        throw std::runtime_error("null or empty CPU expert invocation");
    }
    if (threads <= 0) {
        throw std::runtime_error("CPU expert thread count must be positive");
    }
    auto & profile = cpu_profile();
    const auto prepare_start = std::chrono::steady_clock::now();

    const int n_experts = expert_count;
    const int n_embd = experts[0].gate.in_features;
    const int n_ff = experts[0].gate.out_features;
    if (n_embd <= 0 || n_ff <= 0 || n_embd % 16 != 0 || n_ff % 16 != 0) {
        throw std::runtime_error("CPU expert dimensions must be positive multiples of 16");
    }
    std::vector<ExpertRuntime> runtime(static_cast<std::size_t>(n_experts));
    for (int expert = 0; expert < n_experts; ++expert) {
        const auto & desc = experts[expert];
        if (desc.gate.in_features != n_embd || desc.up.in_features != n_embd
            || desc.gate.out_features != n_ff || desc.up.out_features != n_ff
            || desc.down.in_features != n_ff || desc.down.out_features != n_embd) {
            throw std::runtime_error("inconsistent CPU expert projection dimensions");
        }
        auto & work = runtime[expert];
        work.gate = prepare_projection(desc.gate);
        work.up = prepare_projection(desc.up);
        work.down = prepare_projection(desc.down);
        work.activation.resize(n_ff);
        work.gate_output.resize(n_ff);
        work.up_output.resize(n_ff);
        work.down_output.resize(n_embd);
    }

    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    profile.prepare_ns.fetch_add(elapsed_ns(prepare_start), std::memory_order_relaxed);
    const int io_threads = io_thread_count(threads);

    // Partition the call: cached experts compute immediately while missing experts stream in
    // from the io pool; each missing expert computes as soon as its projections land. Serial
    // fallback (BW24_CPU_EXPERT_PIPELINE=0) keeps the read-everything-then-compute order.
    CallIoState io_state;
    IoDrainGuard drain_guard;
    std::vector<int> cached_experts;
    std::vector<int> missing_experts;
    if (pipeline_enabled()) {
        io_state.pending_requests.assign(static_cast<std::size_t>(n_experts), 0);
        std::vector<IoJob> jobs;
        jobs.reserve(static_cast<std::size_t>(n_experts) * 6);
        for (int expert = 0; expert < n_experts; ++expert) {
            auto & work = runtime[static_cast<std::size_t>(expert)];
            append_projection_jobs(jobs, work.gate, expert, io_state);
            append_projection_jobs(jobs, work.up, expert, io_state);
            append_projection_jobs(jobs, work.down, expert, io_state);
            if (io_state.pending_requests[static_cast<std::size_t>(expert)] == 0) {
                cached_experts.push_back(expert);
            } else {
                missing_experts.push_back(expert);
            }
        }
        io_state.outstanding_experts = static_cast<int>(missing_experts.size());
        drain_guard.state = &io_state;
        if (!jobs.empty()) {
            auto & pool = IoPool::instance();
            pool.ensure_started(io_threads);
            pool.submit(std::move(jobs));
        }
    } else {
        std::vector<ProjectionRuntime *> projections;
        projections.reserve(static_cast<std::size_t>(n_experts) * 3);
        for (auto & expert : runtime) {
            projections.push_back(&expert.gate);
            projections.push_back(&expert.up);
            projections.push_back(&expert.down);
        }
        load_projection_weights(projections, io_threads);
        cached_experts.reserve(static_cast<std::size_t>(n_experts));
        for (int expert = 0; expert < n_experts; ++expert) cached_experts.push_back(expert);
    }

    std::uint64_t compute_elapsed = 0;
    QuantizedActivation input_activation;
    {
        const auto compute_start = std::chrono::steady_clock::now();
        input_activation.prepare(n_embd);
        if (!input_activation.quantize(input, n_embd)) {
            throw std::runtime_error("non-finite bw24 CPU expert input activation");
        }
        compute_elapsed += elapsed_ns(compute_start);
    }
    std::vector<QuantizedActivation> down_activations(static_cast<std::size_t>(n_experts));
    for (auto & activation : down_activations) activation.prepare(n_ff);
    std::vector<std::uint8_t> down_activation_finite(static_cast<std::size_t>(n_experts), 1);
    for (auto & work : runtime) {
        work.gate.activation_f32 = input;
        work.up.activation_f32 = input;
        work.gate.activation_q8 = &input_activation;
        work.up.activation_q8 = &input_activation;
    }

    {
        const auto compute_start = std::chrono::steady_clock::now();
        compute_experts(cached_experts, runtime, experts, down_activations,
                        down_activation_finite, n_ff, n_embd, threads);
        compute_elapsed += elapsed_ns(compute_start);
    }

    // Consume missing experts in read-completion order. io_ns on this path records the wall
    // time compute actually spent blocked on reads (the exposed remainder after overlap).
    std::size_t consumed = 0;
    std::vector<int> ready_now;
    while (consumed < missing_experts.size()) {
        {
            const auto wait_start = std::chrono::steady_clock::now();
            std::unique_lock<std::mutex> lock(io_state.mutex);
            io_state.ready_cv.wait(lock, [&io_state] {
                return !io_state.ready.empty() || io_state.first_error != 0;
            });
            if (io_state.first_error != 0) break;
            ready_now.swap(io_state.ready);
            lock.unlock();
            profile.io_ns.fetch_add(elapsed_ns(wait_start), std::memory_order_relaxed);
        }
        const auto insert_start = std::chrono::steady_clock::now();
        for (const int expert : ready_now) {
            auto & work = runtime[static_cast<std::size_t>(expert)];
            for (auto * projection : { &work.gate, &work.up, &work.down }) {
                if (!projection->needs_read) continue;
                weight_cache().insert(projection->cache_key, projection->weight_owner);
                profile.read_projections.fetch_add(1, std::memory_order_relaxed);
                profile.read_bytes.fetch_add(
                    projection->desc->byte_len, std::memory_order_relaxed);
            }
        }
        profile.insert_ns.fetch_add(elapsed_ns(insert_start), std::memory_order_relaxed);
        const auto compute_start = std::chrono::steady_clock::now();
        compute_experts(ready_now, runtime, experts, down_activations,
                        down_activation_finite, n_ff, n_embd, threads);
        compute_elapsed += elapsed_ns(compute_start);
        consumed += ready_now.size();
        ready_now.clear();
    }
    if (io_state.first_error != 0) {
        throw std::runtime_error(std::string("CPU expert pipelined pread failed: ")
            + std::strerror(io_state.first_error));
    }

    {
        const auto compute_start = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static) num_threads(threads)
        for (int row = 0; row < n_embd; ++row) {
            float sum = 0.0f;
            for (int expert = 0; expert < n_experts; ++expert) {
                const float scale = experts[expert].route_weight * experts[expert].down.scale;
                sum = std::fma(runtime[expert].down_output[row], scale, sum);
            }
            output[row] = sum;
        }
        compute_elapsed += elapsed_ns(compute_start);
    }
    if (std::find(down_activation_finite.begin(), down_activation_finite.end(), 0)
        != down_activation_finite.end()) {
        throw std::runtime_error("non-finite bw24 CPU expert SwiGLU activation");
    }
    profile.compute_ns.fetch_add(compute_elapsed, std::memory_order_relaxed);
    profile.calls.fetch_add(1, std::memory_order_relaxed);
    if (error != nullptr && error_capacity != 0) error[0] = '\0';
    return 0;
} catch (const std::exception & exception) {
    copy_error(error, error_capacity, exception.what());
    return 1;
} catch (...) {
    copy_error(error, error_capacity, "unknown CPU expert failure");
    return 1;
}

extern "C" int bw24_cpu_moe_token_v2(
        const bw24_cpu_expert_v2 * experts,
        std::int32_t expert_count,
        const float * input,
        float * output,
        std::int32_t threads,
        char * error,
        std::size_t error_capacity) {
    return bw24_cpu_moe_token_impl(
        experts, expert_count, input, output, threads, error, error_capacity);
}

// Model-independent correctness hook used by `cpu-native-check`. This intentionally exercises the
// same activation quantizer and row-dot dispatch as production without constructing a full model.
extern "C" int bw24_cpu_dot_v2(
        std::int32_t qtype,
        const std::uint8_t * weights,
        std::size_t row_bytes,
        const float * input,
        std::int32_t count,
        float * output,
        char * error,
        std::size_t error_capacity) try {
    if (weights == nullptr || input == nullptr || output == nullptr || count <= 0) {
        throw std::runtime_error("invalid bw24 CPU dot invocation");
    }
    const auto spec = quant_spec(qtype);
    if (count % spec.block != 0
        || row_bytes != static_cast<std::size_t>(count / spec.block) * spec.bytes) {
        throw std::runtime_error("invalid bw24 CPU dot row layout");
    }
    QuantizedActivation quantized;
    const QuantizedActivation * quantized_ptr = nullptr;
    if (qtype != QT_F32 && qtype != QT_BF16) {
        quantized.prepare(count);
        if (!quantized.quantize(input, count)) {
            throw std::runtime_error("non-finite bw24 CPU dot input activation");
        }
        quantized_ptr = &quantized;
    }
    *output = dot_row_native(qtype, weights, input, quantized_ptr, count);
    if (error != nullptr && error_capacity != 0) error[0] = '\0';
    return 0;
} catch (const std::exception & exception) {
    copy_error(error, error_capacity, exception.what());
    return 1;
} catch (...) {
    copy_error(error, error_capacity, "unknown bw24 CPU dot failure");
    return 1;
}

extern "C" void bw24_cpu_expert_cache_stats_v2(
        std::uint64_t * hits,
        std::uint64_t * misses,
        std::uint64_t * read_bytes,
        std::uint64_t * resident_bytes) noexcept {
    if (hits != nullptr) *hits = 0;
    if (misses != nullptr) *misses = 0;
    if (read_bytes != nullptr) *read_bytes = 0;
    if (resident_bytes != nullptr) *resident_bytes = 0;
    try {
        weight_cache().snapshot(hits, misses, read_bytes, resident_bytes);
    } catch (...) {
        // Stats are diagnostic and the stable C ABI must never propagate a C++ exception.
    }
}

extern "C" void bw24_cpu_expert_profile_stats_v2(
        std::uint64_t * prepare_ns,
        std::uint64_t * io_ns,
        std::uint64_t * insert_ns,
        std::uint64_t * compute_ns) {
    auto & profile = cpu_profile();
    if (prepare_ns != nullptr) {
        *prepare_ns = profile.prepare_ns.load(std::memory_order_relaxed);
    }
    if (io_ns != nullptr) *io_ns = profile.io_ns.load(std::memory_order_relaxed);
    if (insert_ns != nullptr) *insert_ns = profile.insert_ns.load(std::memory_order_relaxed);
    if (compute_ns != nullptr) *compute_ns = profile.compute_ns.load(std::memory_order_relaxed);
}
