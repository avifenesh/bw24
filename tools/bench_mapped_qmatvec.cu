// Compare the exact bw24 qmatvec kernel with weights in HBM versus CUDA-mapped pinned RAM.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_120a tools/bench_mapped_qmatvec.cu \
//     -o /tmp/bw24-bench-mapped-qmatvec
//
// Run:
//   bw24-bench-mapped-qmatvec FILE OFFSET BYTES IN_F OUT_F ROW_BYTES QTYPE ITERATIONS
//
// QTYPE is qmatvec.cu's internal QT_* value, not the GGML enum. The benchmark deliberately
// includes the production kernel source so mapped-host and resident arms execute identical math.

#include "../crates/bw24-engine/cu/qmatvec.cu"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace {

void cuda_check(cudaError_t result, const char * operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

uint64_t parse_u64(const char * value, const char * name) {
    char * end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0') {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return parsed;
}

void pread_exact(int fd, uint8_t * destination, size_t length, uint64_t offset) {
    while (length != 0) {
        const ssize_t result = pread(fd, destination, length, static_cast<off_t>(offset));
        if (result < 0 && errno == EINTR) {
            continue;
        }
        if (result <= 0) {
            throw std::runtime_error(result == 0 ? "short positioned read"
                                                  : std::string("pread: ") + std::strerror(errno));
        }
        destination += result;
        length -= static_cast<size_t>(result);
        offset += static_cast<uint64_t>(result);
    }
}

float time_qmatvec(const uint8_t * weights, const float * input, float * output,
                   int in_f, int out_f, int qtype, int64_t row_bytes, int iterations) {
    for (int warmup = 0; warmup < 5; ++warmup) {
        qmatvec_f32<<<dim3(out_f, 1, 1), 256>>>(
            weights, input, output, in_f, out_f, 1, qtype, row_bytes);
    }
    cuda_check(cudaGetLastError(), "qmatvec warmup launch");
    cuda_check(cudaDeviceSynchronize(), "qmatvec warmup sync");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int iteration = 0; iteration < iterations; ++iteration) {
        qmatvec_f32<<<dim3(out_f, 1, 1), 256>>>(
            weights, input, output, in_f, out_f, 1, qtype, row_bytes);
    }
    cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
    cuda_check(cudaEventDestroy(start), "cudaEventDestroy(start)");
    cuda_check(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    return elapsed_ms / static_cast<float>(iterations);
}

float time_staged_qmatvec(const uint8_t * host_weights, uint8_t * device_weights,
                          size_t bytes, const float * input, float * output,
                          int in_f, int out_f, int qtype, int64_t row_bytes, int iterations) {
    for (int warmup = 0; warmup < 5; ++warmup) {
        cuda_check(cudaMemcpyAsync(
            device_weights, host_weights, bytes, cudaMemcpyHostToDevice),
            "cudaMemcpyAsync(staged warmup)");
        qmatvec_f32<<<dim3(out_f, 1, 1), 256>>>(
            device_weights, input, output, in_f, out_f, 1, qtype, row_bytes);
    }
    cuda_check(cudaGetLastError(), "staged qmatvec warmup launch");
    cuda_check(cudaDeviceSynchronize(), "staged qmatvec warmup sync");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(staged start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(staged stop)");
    cuda_check(cudaEventRecord(start), "cudaEventRecord(staged start)");
    for (int iteration = 0; iteration < iterations; ++iteration) {
        cuda_check(cudaMemcpyAsync(
            device_weights, host_weights, bytes, cudaMemcpyHostToDevice),
            "cudaMemcpyAsync(staged)");
        qmatvec_f32<<<dim3(out_f, 1, 1), 256>>>(
            device_weights, input, output, in_f, out_f, 1, qtype, row_bytes);
    }
    cuda_check(cudaEventRecord(stop), "cudaEventRecord(staged stop)");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(staged stop)");
    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop),
               "cudaEventElapsedTime(staged)");
    cuda_check(cudaEventDestroy(start), "cudaEventDestroy(staged start)");
    cuda_check(cudaEventDestroy(stop), "cudaEventDestroy(staged stop)");
    return elapsed_ms / static_cast<float>(iterations);
}

bool q8_expert_supported(int qtype) {
    return qtype == QT_IQ3_S || qtype == QT_IQ4_XS || qtype == QT_Q3_K
        || qtype == QT_Q4_K || qtype == QT_Q6_K || qtype == QT_Q4_0;
}

float time_quantize_q8_1(const float * input, signed char * aq, float * ad,
                         int in_f, int iterations) {
    const dim3 block(256, 1, 1);
    const dim3 grid((in_f + block.x - 1) / block.x, 1, 1);
    for (int warmup = 0; warmup < 5; ++warmup) {
        quantize_q8_1<<<grid, block>>>(input, aq, ad, in_f, 1);
    }
    cuda_check(cudaGetLastError(), "quantize_q8_1 warmup launch");
    cuda_check(cudaDeviceSynchronize(), "quantize_q8_1 warmup sync");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(q8 quantize start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(q8 quantize stop)");
    cuda_check(cudaEventRecord(start), "cudaEventRecord(q8 quantize start)");
    for (int iteration = 0; iteration < iterations; ++iteration) {
        quantize_q8_1<<<grid, block>>>(input, aq, ad, in_f, 1);
    }
    cuda_check(cudaEventRecord(stop), "cudaEventRecord(q8 quantize stop)");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(q8 quantize stop)");
    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop),
               "cudaEventElapsedTime(q8 quantize)");
    cuda_check(cudaEventDestroy(start), "cudaEventDestroy(q8 quantize start)");
    cuda_check(cudaEventDestroy(stop), "cudaEventDestroy(q8 quantize stop)");
    return elapsed_ms / static_cast<float>(iterations);
}

float time_q8_expert(const uint8_t * weights, const signed char * aq, const float * ad,
                     float * output, int in_f, int out_f, int qtype, int64_t row_bytes,
                     int iterations) {
    const dim3 block(32, BW24_MMVQ_ROWS, 1);
    const dim3 grid((out_f + BW24_MMVQ_ROWS - 1) / BW24_MMVQ_ROWS, 1, 1);
    for (int warmup = 0; warmup < 5; ++warmup) {
        qmatvec_expert_q8<<<grid, block>>>(
            weights, aq, ad, output, in_f, out_f, 1, qtype, row_bytes);
    }
    cuda_check(cudaGetLastError(), "q8 expert warmup launch");
    cuda_check(cudaDeviceSynchronize(), "q8 expert warmup sync");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(q8 expert start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(q8 expert stop)");
    cuda_check(cudaEventRecord(start), "cudaEventRecord(q8 expert start)");
    for (int iteration = 0; iteration < iterations; ++iteration) {
        qmatvec_expert_q8<<<grid, block>>>(
            weights, aq, ad, output, in_f, out_f, 1, qtype, row_bytes);
    }
    cuda_check(cudaEventRecord(stop), "cudaEventRecord(q8 expert stop)");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(q8 expert stop)");
    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop),
               "cudaEventElapsedTime(q8 expert)");
    cuda_check(cudaEventDestroy(start), "cudaEventDestroy(q8 expert start)");
    cuda_check(cudaEventDestroy(stop), "cudaEventDestroy(q8 expert stop)");
    return elapsed_ms / static_cast<float>(iterations);
}

} // namespace

int main(int argc, char ** argv) try {
    if (argc != 9) {
        std::fprintf(stderr,
            "usage: %s FILE OFFSET BYTES IN_F OUT_F ROW_BYTES QTYPE ITERATIONS\n", argv[0]);
        return 2;
    }
    const char * path = argv[1];
    const uint64_t offset = parse_u64(argv[2], "offset");
    const size_t bytes = parse_u64(argv[3], "bytes");
    const int in_f = static_cast<int>(parse_u64(argv[4], "in_f"));
    const int out_f = static_cast<int>(parse_u64(argv[5], "out_f"));
    const int64_t row_bytes = static_cast<int64_t>(parse_u64(argv[6], "row_bytes"));
    const int qtype = static_cast<int>(parse_u64(argv[7], "qtype"));
    const int iterations = static_cast<int>(parse_u64(argv[8], "iterations"));
    if (bytes != static_cast<size_t>(out_f) * static_cast<size_t>(row_bytes)) {
        throw std::runtime_error("BYTES must equal OUT_F * ROW_BYTES");
    }

    cuda_check(cudaSetDeviceFlags(cudaDeviceMapHost), "cudaSetDeviceFlags(cudaDeviceMapHost)");
    cuda_check(cudaSetDevice(0), "cudaSetDevice");

    uint8_t * mapped_host = nullptr;
    cuda_check(cudaHostAlloc(
        reinterpret_cast<void **>(&mapped_host), bytes,
        cudaHostAllocMapped | cudaHostAllocWriteCombined), "cudaHostAlloc(mapped)");
    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error(std::string("open: ") + std::strerror(errno));
    }
    pread_exact(fd, mapped_host, bytes, offset);
    close(fd);

    uint8_t * mapped_device = nullptr;
    cuda_check(cudaHostGetDevicePointer(
        reinterpret_cast<void **>(&mapped_device), mapped_host, 0),
        "cudaHostGetDevicePointer");
    uint8_t * resident = nullptr;
    float * input = nullptr;
    float * output = nullptr;
    signed char * aq = nullptr;
    float * ad = nullptr;
    cuda_check(cudaMalloc(&resident, bytes), "cudaMalloc(resident)");
    cuda_check(cudaMalloc(&input, static_cast<size_t>(in_f) * sizeof(float)), "cudaMalloc(input)");
    cuda_check(cudaMalloc(&output, static_cast<size_t>(out_f) * sizeof(float)), "cudaMalloc(output)");
    cuda_check(cudaMalloc(&aq, static_cast<size_t>(in_f)), "cudaMalloc(aq)");
    cuda_check(cudaMalloc(&ad, static_cast<size_t>(in_f / 32) * sizeof(float)), "cudaMalloc(ad)");
    cuda_check(cudaMemcpy(resident, mapped_host, bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(resident)");

    std::vector<float> input_host(in_f);
    for (int i = 0; i < in_f; ++i) {
        input_host[i] = 0.1f + 2.0f * std::cos(static_cast<float>(i));
    }
    cuda_check(cudaMemcpy(input, input_host.data(), input_host.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "cudaMemcpy(input)");

    const float resident_ms = time_qmatvec(
        resident, input, output, in_f, out_f, qtype, row_bytes, iterations);
    std::vector<float> resident_output(out_f);
    cuda_check(cudaMemcpy(resident_output.data(), output, resident_output.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "cudaMemcpy(resident output)");
    const float mapped_ms = time_qmatvec(
        mapped_device, input, output, in_f, out_f, qtype, row_bytes, iterations);
    std::vector<float> mapped_output(out_f);
    cuda_check(cudaMemcpy(mapped_output.data(), output, mapped_output.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "cudaMemcpy(mapped output)");
    const float staged_ms = time_staged_qmatvec(
        mapped_host, resident, bytes, input, output,
        in_f, out_f, qtype, row_bytes, iterations);
    std::vector<float> staged_output(out_f);
    cuda_check(cudaMemcpy(staged_output.data(), output, staged_output.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "cudaMemcpy(staged output)");
    float quantize_q8_ms = -1.0f;
    float q8_expert_ms = -1.0f;
    if (q8_expert_supported(qtype)) {
        quantize_q8_ms = time_quantize_q8_1(input, aq, ad, in_f, iterations);
        q8_expert_ms = time_q8_expert(
            resident, aq, ad, output, in_f, out_f, qtype, row_bytes, iterations);
    }

    const bool mapped_exact = std::memcmp(resident_output.data(), mapped_output.data(),
                                          resident_output.size() * sizeof(float)) == 0;
    const bool staged_exact = std::memcmp(resident_output.data(), staged_output.data(),
                                          resident_output.size() * sizeof(float)) == 0;
    float max_diff = 0.0f;
    double checksum = 0.0;
    for (size_t i = 0; i < mapped_output.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(mapped_output[i] - resident_output[i]));
        checksum += mapped_output[i];
    }
    std::printf(
        "bytes=%zu shape=%dx%d qtype=%d iterations=%d resident_ms=%.4f staged_ms=%.4f "
        "staged_weight_GB/s=%.2f staged_slowdown=%.2fx mapped_ms=%.4f "
        "mapped_weight_GB/s=%.2f mapped_slowdown=%.2fx mapped_exact=%s staged_exact=%s "
        "quantize_q8_ms=%.4f q8_expert_ms=%.4f q8_total_ms=%.4f "
        "maxdiff=%.9g checksum=%.9g\n",
        bytes, in_f, out_f, qtype, iterations, resident_ms, staged_ms,
        static_cast<double>(bytes) / (staged_ms * 1.0e6), staged_ms / resident_ms, mapped_ms,
        static_cast<double>(bytes) / (mapped_ms * 1.0e6), mapped_ms / resident_ms,
        mapped_exact ? "yes" : "no", staged_exact ? "yes" : "no",
        quantize_q8_ms, q8_expert_ms, quantize_q8_ms + q8_expert_ms, max_diff, checksum);

    cudaFree(output);
    cudaFree(ad);
    cudaFree(aq);
    cudaFree(input);
    cudaFree(resident);
    cudaFreeHost(mapped_host);
    return mapped_exact && staged_exact ? 0 : 1;
} catch (const std::exception & error) {
    std::fprintf(stderr, "ERROR: %s\n", error.what());
    return 1;
}
