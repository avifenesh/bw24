// Measure one real routed token's complete expert-weight read/quant-dot workload in system RAM.
// This is a Fiddler-style CPU-offload feasibility bound: it executes all 79 * top-8 * 3 matrix
// projections from an actual BW24_MOE_TRACE token, but deliberately omits layer dependencies,
// activation functions, and GPU transfers. Those remain costs for an end-to-end backend.
//
// Build:
//   c++ -O3 -march=native -fopenmp tools/bench_cpu_expert_trace.cpp \
//     -I/path/to/llama.cpp/ggml/include -I/path/to/llama.cpp/vendor \
//     -L/path/to/llama.cpp/build/bin -Wl,-rpath,/path/to/llama.cpp/build/bin \
//     -lggml-cpu -lggml-base -o /tmp/bw24-bench-cpu-expert-trace
//
// Run:
//   bw24-bench-cpu-expert-trace REPACK_DIR ROUTE_TRACE THREADS ITERATIONS
// The final 79 trace rows are used, which is one complete Hy3 decode token.

#include "ggml-cpu.h"
#include "ggml.h"
#include "nlohmann/json.hpp"

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
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using json = nlohmann::json;

struct Mapping {
    void * base = MAP_FAILED;
    size_t bytes = 0;

    Mapping() = default;
    Mapping(void * mapped_base, size_t mapped_bytes) : base(mapped_base), bytes(mapped_bytes) {}
    Mapping(const Mapping &) = delete;
    Mapping & operator=(const Mapping &) = delete;
    Mapping(Mapping && other) noexcept : base(other.base), bytes(other.bytes) {
        other.base = MAP_FAILED;
        other.bytes = 0;
    }
    Mapping & operator=(Mapping && other) noexcept {
        if (this != &other) {
            if (base != MAP_FAILED) munmap(base, bytes);
            base = other.base;
            bytes = other.bytes;
            other.base = MAP_FAILED;
            other.bytes = 0;
        }
        return *this;
    }
    ~Mapping() {
        if (base != MAP_FAILED) munmap(base, bytes);
    }
};

struct Activation {
    std::vector<uint8_t> allocation;
    void * aligned = nullptr;
};

struct Task {
    const uint8_t * weights = nullptr;
    const ggml_type_traits_cpu * traits = nullptr;
    const void * activation = nullptr;
    int in_f = 0;
    int out_f = 0;
    size_t row_bytes = 0;
    size_t extent_bytes = 0;
};

uint64_t parse_u64(const char * value, const char * name) {
    char * end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0') {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return parsed;
}

ggml_type find_type(const std::string & wanted) {
    for (int raw = 0; raw < GGML_TYPE_COUNT; ++raw) {
        const auto type = static_cast<ggml_type>(raw);
        const char * name = ggml_type_name(type);
        if (name != nullptr && strcasecmp(name, wanted.c_str()) == 0) {
            return type;
        }
    }
    throw std::runtime_error("unknown ggml type: " + wanted);
}

Mapping map_readonly(const std::string & path) {
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error("open " + path + ": " + std::strerror(errno));
    }
    struct stat info {};
    if (fstat(fd, &info) != 0) {
        const std::string reason = std::strerror(errno);
        close(fd);
        throw std::runtime_error("fstat " + path + ": " + reason);
    }
    void * base = mmap(nullptr, info.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (base == MAP_FAILED) {
        throw std::runtime_error("mmap " + path + ": " + std::strerror(errno));
    }
    return Mapping { base, static_cast<size_t>(info.st_size) };
}

std::vector<std::tuple<int, int, std::vector<int>>> final_token_routes(
        const std::string & trace_path) {
    std::ifstream input(trace_path);
    if (!input) throw std::runtime_error("cannot open route trace: " + trace_path);
    std::vector<std::tuple<int, int, std::vector<int>>> rows;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream stream(line);
        int layer = 0;
        int tokens = 0;
        std::string experts;
        if (!(stream >> layer >> tokens >> experts)) {
            throw std::runtime_error("malformed route trace row");
        }
        std::vector<int> ids;
        std::istringstream list(experts);
        std::string value;
        while (std::getline(list, value, ',')) ids.push_back(std::stoi(value));
        rows.emplace_back(layer, tokens, std::move(ids));
    }
    if (rows.size() < 79) throw std::runtime_error("route trace has fewer than 79 rows");
    rows.erase(rows.begin(), rows.end() - 79);
    for (size_t index = 0; index < rows.size(); ++index) {
        const auto & [layer, tokens, ids] = rows[index];
        if (layer != static_cast<int>(index) + 1 || tokens != 1 || ids.size() != 8) {
            throw std::runtime_error("final trace rows are not one complete 79-layer top-8 token");
        }
    }
    return rows;
}

} // namespace

int main(int argc, char ** argv) try {
    if (argc != 5) {
        std::fprintf(stderr,
            "usage: %s REPACK_DIR ROUTE_TRACE THREADS ITERATIONS\n", argv[0]);
        return 2;
    }
    const std::string repack_dir = argv[1];
    const int threads = static_cast<int>(parse_u64(argv[3], "threads"));
    const int iterations = static_cast<int>(parse_u64(argv[4], "iterations"));
    if (threads <= 0 || iterations <= 0) {
        throw std::runtime_error("THREADS and ITERATIONS must be positive");
    }

    ggml_cpu_init();
    std::ifstream manifest_input(repack_dir + "/manifest.json");
    if (!manifest_input) throw std::runtime_error("cannot open manifest.json");
    const json manifest = json::parse(manifest_input);
    const auto routes = final_token_routes(argv[2]);

    std::map<std::string, Mapping> mappings;
    std::map<std::pair<int, int>, Activation> activations;
    std::vector<Task> tasks;
    tasks.reserve(79 * 8 * 3);
    size_t total_weight_bytes = 0;

    for (const auto & [layer, tokens, ids] : routes) {
        (void) tokens;
        for (const int expert : ids) {
            for (const char * projection : { "gate", "up", "down" }) {
                const std::string name = "blk." + std::to_string(layer) + ".ffn_" + projection
                    + "_exps." + std::to_string(expert) + ".weight";
                const auto & metadata = manifest.at("tensors").at(name);
                const std::string relative_path = metadata.at("file");
                const std::string full_path = repack_dir + "/" + relative_path;
                if (mappings.find(full_path) == mappings.end()) {
                    mappings.emplace(full_path, map_readonly(full_path));
                }
                const auto offset = metadata.at("offset").get<size_t>();
                const auto extent = metadata.at("bytes").get<size_t>();
                const int in_f = metadata.at("ne").at(0).get<int>();
                const int out_f = metadata.at("ne").at(1).get<int>();
                const auto row_bytes = metadata.at("row_bytes").get<size_t>();
                const auto type = find_type(metadata.at("qtype").get<std::string>());
                ggml_quantize_init(type);
                const auto * traits = ggml_get_type_traits_cpu(type);
                if (traits == nullptr || traits->vec_dot == nullptr) {
                    throw std::runtime_error("qtype has no CPU vec_dot: " + name);
                }
                const auto activation_key = std::make_pair(static_cast<int>(traits->vec_dot_type), in_f);
                if (activations.find(activation_key) == activations.end()) {
                    const auto * activation_traits = ggml_get_type_traits_cpu(traits->vec_dot_type);
                    if (activation_traits == nullptr || activation_traits->from_float == nullptr) {
                        throw std::runtime_error("qtype has no activation quantizer: " + name);
                    }
                    std::vector<float> source(in_f);
                    for (int i = 0; i < in_f; ++i) {
                        source[i] = 0.1f + 2.0f * std::cos(static_cast<float>(i));
                    }
                    Activation activation;
                    activation.allocation.resize(ggml_row_size(traits->vec_dot_type, in_f) + 64);
                    activation.aligned = reinterpret_cast<void *>(
                        (reinterpret_cast<uintptr_t>(activation.allocation.data()) + 63) & ~uintptr_t(63));
                    activation_traits->from_float(source.data(), activation.aligned, in_f);
                    activations.emplace(activation_key, std::move(activation));
                }
                const auto & mapping = mappings.at(full_path);
                if (offset > mapping.bytes || extent > mapping.bytes - offset
                    || extent != row_bytes * static_cast<size_t>(out_f)) {
                    throw std::runtime_error("invalid manifest extent: " + name);
                }
                tasks.push_back(Task {
                    static_cast<const uint8_t *>(mapping.base) + offset,
                    traits,
                    activations.at(activation_key).aligned,
                    in_f,
                    out_f,
                    row_bytes,
                    extent,
                });
                total_weight_bytes += extent;
            }
        }
    }

    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    const char * layered_raw = std::getenv("BW24_BENCH_LAYERED_EXPERTS");
    const int layered_experts = layered_raw == nullptr
        ? 0
        : static_cast<int>(parse_u64(layered_raw, "BW24_BENCH_LAYERED_EXPERTS"));
    if (layered_experts < 0 || layered_experts > 8) {
        throw std::runtime_error("BW24_BENCH_LAYERED_EXPERTS must be in 0..=8");
    }
    std::vector<std::vector<float>> layered_outputs;
    if (layered_experts != 0) {
        layered_outputs.resize(tasks.size());
        for (std::size_t index = 0; index < tasks.size(); ++index) {
            layered_outputs[index].resize(tasks[index].out_f);
        }
        total_weight_bytes = 0;
        for (std::size_t layer = 0; layer < 79; ++layer) {
            for (int expert = 0; expert < layered_experts; ++expert) {
                for (int projection = 0; projection < 3; ++projection) {
                    total_weight_bytes += tasks[layer * 24 + expert * 3 + projection].extent_bytes;
                }
            }
        }
    }
    const auto run_flat = [&] {
        double checksum = 0.0;
#pragma omp parallel reduction(+:checksum)
        {
            std::vector<float> output(4096);
#pragma omp for schedule(dynamic, 1)
            for (size_t task_index = 0; task_index < tasks.size(); ++task_index) {
                const auto & task = tasks[task_index];
                for (int row = 0; row < task.out_f; ++row) {
                    task.traits->vec_dot(
                        task.in_f, &output[row], 0,
                        task.weights + task.row_bytes * static_cast<size_t>(row), 0,
                        task.activation, 0, 1);
                }
                for (int row = 0; row < task.out_f; row += 256) checksum += output[row];
            }
        }
        return checksum;
    };
    const auto run_layered = [&] {
        for (std::size_t layer = 0; layer < 79; ++layer) {
            const std::size_t base = layer * 24;
            const int gate_rows = tasks[base].out_f;
            const int down_rows = tasks[base + 2].out_f;
#pragma omp parallel
            {
#pragma omp for schedule(runtime)
                for (int work = 0; work < layered_experts * 2 * gate_rows; ++work) {
                    const int expert = work / (2 * gate_rows);
                    const int local = work % (2 * gate_rows);
                    const int projection = local / gate_rows;
                    const int row = local % gate_rows;
                    const std::size_t index = base + expert * 3 + projection;
                    const auto & task = tasks[index];
                    task.traits->vec_dot(
                        task.in_f, &layered_outputs[index][row], 0,
                        task.weights + task.row_bytes * static_cast<std::size_t>(row), 0,
                        task.activation, 0, 1);
                }
#pragma omp for schedule(runtime)
                for (int work = 0; work < layered_experts * down_rows; ++work) {
                    const int expert = work / down_rows;
                    const int row = work % down_rows;
                    const std::size_t index = base + expert * 3 + 2;
                    const auto & task = tasks[index];
                    task.traits->vec_dot(
                        task.in_f, &layered_outputs[index][row], 0,
                        task.weights + task.row_bytes * static_cast<std::size_t>(row), 0,
                        task.activation, 0, 1);
                }
            }
        }
        double checksum = 0.0;
        for (std::size_t layer = 0; layer < 79; ++layer) {
            for (int expert = 0; expert < layered_experts; ++expert) {
                for (int projection = 0; projection < 3; ++projection) {
                    const std::size_t index = layer * 24 + expert * 3 + projection;
                    for (int row = 0; row < tasks[index].out_f; row += 256) {
                        checksum += layered_outputs[index][row];
                    }
                }
            }
        }
        return checksum;
    };
    const auto run = [&] {
        return layered_experts == 0 ? run_flat() : run_layered();
    };

    // First pass faults the selected 5 GB into RAM. It is intentionally excluded from the bound.
    const double warm_checksum = run();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        const double checksum = run();
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        std::printf(
            "iteration=%d tasks=%zu layered_experts=%d weight_GB=%.3f threads=%d seconds=%.6f "
            "weight_GB/s=%.2f expert_bound_tok/s=%.2f checksum=%.9g warm_checksum=%.9g\n",
            iteration, layered_experts == 0 ? tasks.size() : 79 * layered_experts * 3,
            layered_experts, total_weight_bytes / 1.0e9, threads, seconds,
            total_weight_bytes / seconds / 1.0e9, 1.0 / seconds, checksum, warm_checksum);
    }
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "ERROR: %s\n", error.what());
    return 1;
}
