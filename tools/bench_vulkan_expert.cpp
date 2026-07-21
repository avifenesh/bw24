// Compare one manifest-described expert projection on llama.cpp CPU quant dots and Vulkan.
// The weight transfer is outside the timed loop; this measures a resident per-token matvec.
//
// Build against a Vulkan-enabled llama.cpp tree:
//   c++ -O3 -march=native tools/bench_vulkan_expert.cpp \
//     -I/path/llama.cpp/ggml/include -L/path/vulkan-build/bin \
//     -Wl,-rpath,/path/vulkan-build/bin \
//     -lggml-vulkan -lggml-cpu -lggml -lggml-base -o /tmp/bw24-bench-vulkan-expert
//
// Run with GGML_VK_VISIBLE_DEVICES=0:
//   bw24-bench-vulkan-expert FILE OFFSET QTYPE IN_F OUT_F ROW_BYTES ITERATIONS

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"
#include "ggml.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <strings.h>
#include <vector>

namespace {

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
        if (name != nullptr && strcasecmp(name, wanted) == 0) return type;
    }
    throw std::runtime_error(std::string("unknown ggml type: ") + wanted);
}

} // namespace

int main(int argc, char ** argv) try {
    if (argc != 8) {
        std::fprintf(stderr,
            "usage: %s FILE OFFSET QTYPE IN_F OUT_F ROW_BYTES ITERATIONS\n", argv[0]);
        return 2;
    }
    const char * path = argv[1];
    const size_t offset = parse_u64(argv[2], "offset");
    const ggml_type type = find_type(argv[3]);
    const int in_f = static_cast<int>(parse_u64(argv[4], "in_f"));
    const int out_f = static_cast<int>(parse_u64(argv[5], "out_f"));
    const size_t row_bytes = parse_u64(argv[6], "row_bytes");
    const int iterations = static_cast<int>(parse_u64(argv[7], "iterations"));
    if (in_f <= 0 || out_f <= 0 || iterations <= 0) {
        throw std::runtime_error("dimensions and iterations must be positive");
    }
    ggml_cpu_init();
    ggml_quantize_init(type);
    if (ggml_row_size(type, in_f) != row_bytes) {
        throw std::runtime_error("row_bytes does not match ggml_row_size");
    }
    const size_t extent = row_bytes * static_cast<size_t>(out_f);

    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) throw std::runtime_error(std::string("open failed: ") + std::strerror(errno));
    struct stat info {};
    if (fstat(fd, &info) != 0) {
        close(fd);
        throw std::runtime_error(std::string("fstat failed: ") + std::strerror(errno));
    }
    if (offset > static_cast<size_t>(info.st_size)
        || extent > static_cast<size_t>(info.st_size) - offset) {
        close(fd);
        throw std::runtime_error("projection extent is outside the file");
    }
    void * mapping = mmap(nullptr, info.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) {
        throw std::runtime_error(std::string("mmap failed: ") + std::strerror(errno));
    }
    const auto * weights = static_cast<const uint8_t *>(mapping) + offset;

    std::vector<float> input(in_f);
    for (int index = 0; index < in_f; ++index) {
        input[index] = 0.1f + 2.0f * std::cos(static_cast<float>(index));
    }
    const auto * traits = ggml_get_type_traits_cpu(type);
    if (traits == nullptr || traits->vec_dot == nullptr) {
        throw std::runtime_error("CPU vec_dot is unavailable for selected qtype");
    }
    const auto * activation_traits = ggml_get_type_traits_cpu(traits->vec_dot_type);
    if (activation_traits == nullptr || activation_traits->from_float == nullptr) {
        throw std::runtime_error("CPU activation quantizer is unavailable for selected qtype");
    }
    std::vector<uint8_t> activation(ggml_row_size(traits->vec_dot_type, in_f) + 64);
    void * activation_aligned = reinterpret_cast<void *>(
        (reinterpret_cast<uintptr_t>(activation.data()) + 63) & ~uintptr_t(63));
    activation_traits->from_float(input.data(), activation_aligned, in_f);
    std::vector<float> cpu_output(out_f);
    for (int row = 0; row < out_f; ++row) {
        traits->vec_dot(in_f, &cpu_output[row], 0,
            weights + row_bytes * static_cast<size_t>(row), 0,
            activation_aligned, 0, 1);
    }

    ggml_backend_t backend = ggml_backend_vk_init(0);
    if (backend == nullptr) throw std::runtime_error("cannot initialize Vulkan device 0");
    ggml_init_params params {
        /* .mem_size = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context * context = ggml_init(params);
    if (context == nullptr) throw std::runtime_error("cannot initialize ggml context");
    ggml_tensor * weight_tensor = ggml_new_tensor_2d(context, type, in_f, out_f);
    ggml_tensor * input_tensor = ggml_new_tensor_1d(context, GGML_TYPE_F32, in_f);
    ggml_tensor * output_tensor = ggml_mul_mat(context, weight_tensor, input_tensor);
    ggml_cgraph * graph = ggml_new_graph_custom(context, 64, false);
    ggml_build_forward_expand(graph, output_tensor);
    if (!ggml_backend_supports_op(backend, output_tensor)) {
        throw std::runtime_error("Vulkan backend does not support this projection");
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) throw std::runtime_error("cannot allocate Vulkan tensor buffer");
    if (ggml_nbytes(weight_tensor) != extent) {
        throw std::runtime_error("ggml tensor extent differs from manifest extent");
    }
    ggml_backend_tensor_set(weight_tensor, weights, 0, extent);
    ggml_backend_tensor_set(input_tensor, input.data(), 0, input.size() * sizeof(float));
    for (int warmup = 0; warmup < 3; ++warmup) {
        const auto status = ggml_backend_graph_compute(backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("Vulkan warmup failed: ")
                + ggml_status_to_string(status));
        }
    }
    ggml_backend_synchronize(backend);
    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto status = ggml_backend_graph_compute(backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("Vulkan compute failed: ")
                + ggml_status_to_string(status));
        }
    }
    ggml_backend_synchronize(backend);
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::vector<float> vulkan_output(out_f);
    ggml_backend_tensor_get(
        output_tensor, vulkan_output.data(), 0, vulkan_output.size() * sizeof(float));

    double max_abs = 0.0;
    double max_rel = 0.0;
    double cpu_checksum = 0.0;
    double vulkan_checksum = 0.0;
    for (int row = 0; row < out_f; ++row) {
        const double cpu = cpu_output[row];
        const double vulkan = vulkan_output[row];
        const double absolute = std::abs(cpu - vulkan);
        max_abs = std::max(max_abs, absolute);
        max_rel = std::max(max_rel, absolute / std::max(1.0e-12, std::abs(cpu)));
        cpu_checksum += cpu;
        vulkan_checksum += vulkan;
    }
    std::printf(
        "type=%s shape=%dx%d extent=%zu iterations=%d ms/projection=%.6f "
        "weight_GB/s=%.3f max_abs=%.9g max_rel=%.9g cpu_checksum=%.12g "
        "vulkan_checksum=%.12g\n",
        ggml_type_name(type), in_f, out_f, extent, iterations,
        seconds * 1000.0 / iterations,
        static_cast<double>(extent) * iterations / seconds / 1.0e9,
        max_abs, max_rel, cpu_checksum, vulkan_checksum);

    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    ggml_backend_free(backend);
    munmap(mapping, info.st_size);
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "ERROR: %s\n", error.what());
    return 1;
}
