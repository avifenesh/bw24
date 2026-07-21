// Benchmark one manifest-described expert projection through llama.cpp's native CPU quant dots.
//
// Build example:
//   c++ -O3 -march=native -fopenmp tools/bench_cpu_expert.cpp \
//     -I/path/to/llama.cpp/ggml/include -L/path/to/llama.cpp/build/bin \
//     -Wl,-rpath,/path/to/llama.cpp/build/bin -lggml-cpu -lggml-base \
//     -o /tmp/bench-cpu-expert
//
// Run:
//   bench-cpu-expert FILE OFFSET QTYPE IN_F OUT_F ROW_BYTES THREADS ITERATIONS

#include "ggml-cpu.h"
#include "ggml.h"

#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

[[noreturn]] void usage(const char * program) {
    std::fprintf(stderr,
        "usage: %s FILE OFFSET QTYPE IN_F OUT_F ROW_BYTES THREADS ITERATIONS\n",
        program);
    std::exit(2);
}

uint64_t parse_u64(const char * value, const char * name) {
    char * end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0') {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return parsed;
}

ggml_type find_type(const char * wanted) {
    for (int raw = 0; raw < GGML_TYPE_COUNT; ++raw) {
        const auto type = static_cast<ggml_type>(raw);
        const char * name = ggml_type_name(type);
        if (name != nullptr && strcasecmp(name, wanted) == 0) {
            return type;
        }
    }
    throw std::runtime_error(std::string("unknown ggml type: ") + wanted);
}

} // namespace

int main(int argc, char ** argv) try {
    if (argc != 9) {
        usage(argv[0]);
    }

    const char * path = argv[1];
    const size_t offset = parse_u64(argv[2], "offset");
    const auto type = find_type(argv[3]);
    const int in_f = static_cast<int>(parse_u64(argv[4], "in_f"));
    const int out_f = static_cast<int>(parse_u64(argv[5], "out_f"));
    const size_t row_bytes = parse_u64(argv[6], "row_bytes");
    const int threads = static_cast<int>(parse_u64(argv[7], "threads"));
    const int iterations = static_cast<int>(parse_u64(argv[8], "iterations"));
    if (in_f <= 0 || out_f <= 0 || threads <= 0 || iterations <= 0) {
        throw std::runtime_error("dimensions, threads, and iterations must be positive");
    }

    ggml_cpu_init();
    ggml_quantize_init(type);
    const auto * traits = ggml_get_type_traits_cpu(type);
    if (traits == nullptr || traits->vec_dot == nullptr) {
        throw std::runtime_error("selected type has no CPU vec_dot");
    }
    const auto * activation_traits = ggml_get_type_traits_cpu(traits->vec_dot_type);
    if (activation_traits == nullptr || activation_traits->from_float == nullptr) {
        throw std::runtime_error("selected type has no activation quantizer");
    }
    const size_t expected_row_bytes = ggml_row_size(type, in_f);
    if (expected_row_bytes != row_bytes) {
        throw std::runtime_error("manifest row_bytes does not match ggml_row_size: "
            + std::to_string(row_bytes) + " vs " + std::to_string(expected_row_bytes));
    }
    const size_t extent_bytes = row_bytes * static_cast<size_t>(out_f);

    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error(std::string("open failed: ") + std::strerror(errno));
    }
    struct stat stat_buf {};
    if (fstat(fd, &stat_buf) != 0) {
        close(fd);
        throw std::runtime_error(std::string("fstat failed: ") + std::strerror(errno));
    }
    if (offset > static_cast<size_t>(stat_buf.st_size)
        || extent_bytes > static_cast<size_t>(stat_buf.st_size) - offset) {
        close(fd);
        throw std::runtime_error("expert extent is outside the file");
    }
    void * mapping = mmap(nullptr, stat_buf.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) {
        throw std::runtime_error(std::string("mmap failed: ") + std::strerror(errno));
    }
    const auto * weights = static_cast<const uint8_t *>(mapping) + offset;

    std::vector<float> activation(in_f);
    for (int i = 0; i < in_f; ++i) {
        activation[i] = 0.1f + 2.0f * std::cos(static_cast<float>(i));
    }
    std::vector<uint8_t> activation_quant(ggml_row_size(traits->vec_dot_type, in_f) + 64);
    void * activation_aligned = reinterpret_cast<void *>(
        (reinterpret_cast<uintptr_t>(activation_quant.data()) + 63) & ~uintptr_t(63));
    activation_traits->from_float(activation.data(), activation_aligned, in_f);
    std::vector<float> output(out_f);

    // Make this a compute/memory-bandwidth measurement, not a page-fault measurement. The real
    // storage lane is measured independently by run-gen's physical read-byte counter.
    volatile uint8_t touch = 0;
    for (size_t i = 0; i < extent_bytes; i += 4096) {
        touch ^= weights[i];
    }
    if (extent_bytes != 0) {
        touch ^= weights[extent_bytes - 1];
    }

    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    const auto run = [&] {
#pragma omp parallel for schedule(static)
        for (int row = 0; row < out_f; ++row) {
            traits->vec_dot(in_f, &output[row], 0, weights + row_bytes * row, 0,
                activation_aligned, 0, 1);
        }
    };
    for (int warmup = 0; warmup < 3; ++warmup) {
        run();
    }

    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        run();
    }
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    double checksum = touch;
    for (float value : output) {
        checksum += value;
    }
    const double ms_per_projection = elapsed * 1000.0 / iterations;
    const double gbps = static_cast<double>(extent_bytes) * iterations / elapsed / 1e9;
    std::printf(
        "type=%s shape=%dx%d row_bytes=%zu extent=%zu threads=%d iterations=%d "
        "ms/projection=%.4f weight_GB/s=%.2f checksum=%.9g\n",
        ggml_type_name(type), in_f, out_f, row_bytes, extent_bytes, threads, iterations,
        ms_per_projection, gbps, checksum);

    munmap(mapping, stat_buf.st_size);
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "ERROR: %s\n", error.what());
    return 1;
}
