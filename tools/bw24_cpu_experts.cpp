// Native CPU implementation of one routed MoE token for bw24's experimental Hy3
// CPU/GPU expert split. The stable v1 C ABI is loaded only when BW24_CPU_EXPERT_LIB is set. The
// normal CUDA build has no llama.cpp compile-time or link-time dependency.
//
// Build against a current llama.cpp checkout:
//   c++ -O3 -march=native -fPIC -shared -fopenmp tools/bw24_cpu_experts.cpp
//     -I/path/to/llama.cpp/ggml/include
//     -L/path/to/llama.cpp/build/bin -Wl,-rpath,/path/to/llama.cpp/build/bin
//     -lggml-cpu -lggml-base -o /tmp/libbw24-cpu-experts.so

#include "ggml-cpu.h"
#include "ggml.h"

#include <omp.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <exception>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {

struct bw24_cpu_projection_v1 {
    const std::uint8_t * weights;
    std::int32_t ggml_type;
    std::int32_t in_features;
    std::int32_t out_features;
    std::size_t row_bytes;
    std::size_t byte_len;
    std::int32_t file_fd;
    std::uint64_t file_offset;
    float scale;
};

struct bw24_cpu_expert_v1 {
    bw24_cpu_projection_v1 gate;
    bw24_cpu_projection_v1 up;
    bw24_cpu_projection_v1 down;
    float route_weight;
};

std::uint32_t bw24_cpu_experts_abi_version() {
    return 1;
}

} // extern "C"

namespace {

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

struct CacheKey {
    InodeKey file;
    std::uint64_t offset = 0;
    std::size_t len = 0;

    bool operator==(const CacheKey & other) const {
        return file == other.file && offset == other.offset && len == other.len;
    }
};

struct CacheKeyHash {
    std::size_t operator()(const CacheKey & key) const {
        std::size_t value = InodeKeyHash{}(key.file);
        value ^= std::hash<std::uint64_t>{}(key.offset)
            + 0x9e3779b9 + (value << 6) + (value >> 2);
        value ^= std::hash<std::size_t>{}(key.len)
            + 0x9e3779b9 + (value << 6) + (value >> 2);
        return value;
    }
};

struct ProjectionRuntime {
    const bw24_cpu_projection_v1 * desc = nullptr;
    const ggml_type_traits_cpu * traits = nullptr;
    const ggml_type_traits_cpu * activation_traits = nullptr;
    ggml_type activation_type = GGML_TYPE_COUNT;
    AlignedBytes activation;
    void * activation_data = nullptr;
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

std::once_flag ggml_init_once;

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
    MirrorFiles() {
        const char * path = std::getenv("BW24_CPU_EXPERT_MIRROR_MAP");
        if (path == nullptr || *path == '\0') return;
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error(std::string("cannot open CPU expert mirror map: ") + path);
        }
        std::string line;
        while (std::getline(input, line)) {
            const auto first = line.find('\t');
            const auto second = first == std::string::npos
                ? std::string::npos
                : line.find('\t', first + 1);
            if (first == std::string::npos || second == std::string::npos
                || line.find('\t', second + 1) != std::string::npos
                || second + 1 == line.size()) {
                throw std::runtime_error("malformed CPU expert mirror map");
            }
            std::size_t device_end = 0;
            std::size_t inode_end = 0;
            const auto device = std::stoull(line.substr(0, first), &device_end);
            const auto inode = std::stoull(
                line.substr(first + 1, second - first - 1), &inode_end);
            if (device_end != first || inode_end != second - first - 1) {
                throw std::runtime_error("malformed CPU expert mirror-map inode identity");
            }
            const std::string alternate = line.substr(second + 1);
            const auto inserted = paths_.emplace(InodeKey { device, inode }, alternate);
            if (!inserted.second && inserted.first->second != alternate) {
                throw std::runtime_error("conflicting CPU expert mirror-map inode");
            }
        }
        if (input.bad()) throw std::runtime_error("cannot read CPU expert mirror map");
        if (paths_.empty()) throw std::runtime_error("CPU expert mirror map is empty");
        std::fprintf(stderr, "[bw24-cpu] mirrored direct I/O: %zu inode mappings\n", paths_.size());
    }

    ~MirrorFiles() {
        for (const auto & [_, fd] : files_) close(fd);
    }

    int resolve(int source_fd, const InodeKey & identity) {
        if (paths_.empty()) return -1;
        std::lock_guard<std::mutex> lock(mutex_);
        const auto cached = files_.find(identity);
        if (cached != files_.end()) return cached->second;
        struct stat source {};
        if (fstat(source_fd, &source) != 0) {
            throw std::runtime_error(
                "cannot stat CPU expert source fd: " + std::string(std::strerror(errno)));
        }
        if (!(InodeKey {
                  static_cast<std::uint64_t>(source.st_dev),
                  static_cast<std::uint64_t>(source.st_ino),
              } == identity)) {
            throw std::runtime_error("CPU expert source changed while resolving its mirror");
        }
        const auto path = paths_.find(identity);
        if (path == paths_.end()) {
            throw std::runtime_error("CPU expert source inode is absent from mirror map");
        }
        const int alternate = open(path->second.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (alternate < 0) {
            throw std::runtime_error(
                "cannot open mirrored CPU expert source " + path->second
                + ": " + std::strerror(errno));
        }
        struct stat other {};
        if (fstat(alternate, &other) != 0
            || other.st_size != source.st_size || other.st_dev == source.st_dev) {
            close(alternate);
            throw std::runtime_error("CPU expert mirror has wrong size or physical filesystem");
        }
        files_.emplace(identity, alternate);
        return alternate;
    }

private:
    std::mutex mutex_;
    std::unordered_map<InodeKey, std::string, InodeKeyHash> paths_;
    std::unordered_map<InodeKey, int, InodeKeyHash> files_;
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
        const bw24_cpu_projection_v1 & desc,
        bool allocate_activation) {
    if ((desc.weights == nullptr && desc.file_fd < 0)
        || desc.in_features <= 0 || desc.out_features <= 0) {
        throw std::runtime_error("invalid CPU expert projection descriptor");
    }
    if (desc.ggml_type < 0 || desc.ggml_type >= GGML_TYPE_COUNT) {
        throw std::runtime_error("CPU expert projection has invalid ggml type");
    }
    const auto type = static_cast<ggml_type>(desc.ggml_type);
    ggml_quantize_init(type);
    const auto * traits = ggml_get_type_traits_cpu(type);
    if (traits == nullptr || traits->vec_dot == nullptr) {
        throw std::runtime_error(std::string("CPU vec_dot unavailable for ") + ggml_type_name(type));
    }
    const auto * activation_traits = ggml_get_type_traits_cpu(traits->vec_dot_type);
    if (activation_traits == nullptr || activation_traits->from_float == nullptr) {
        throw std::runtime_error(std::string("CPU activation quantizer unavailable for ")
            + ggml_type_name(type));
    }
    const std::size_t expected_row = ggml_row_size(type, desc.in_features);
    if (expected_row != desc.row_bytes) {
        throw std::runtime_error(std::string("CPU expert row-size mismatch for ")
            + ggml_type_name(type) + ": descriptor=" + std::to_string(desc.row_bytes)
            + " ggml=" + std::to_string(expected_row));
    }
    const std::size_t expected_bytes = desc.row_bytes * static_cast<std::size_t>(desc.out_features);
    if (desc.byte_len != expected_bytes) {
        throw std::runtime_error(std::string("CPU expert extent mismatch for ")
            + ggml_type_name(type) + ": descriptor=" + std::to_string(desc.byte_len)
            + " expected=" + std::to_string(expected_bytes));
    }
    ProjectionRuntime runtime;
    runtime.desc = &desc;
    runtime.traits = traits;
    runtime.activation_traits = activation_traits;
    runtime.activation_type = traits->vec_dot_type;
    if (allocate_activation) {
        runtime.activation.resize(ggml_row_size(traits->vec_dot_type, desc.in_features));
        runtime.activation_data = runtime.activation.data;
    }
    if (desc.file_fd >= 0) {
        const InodeKey identity = inode_key(desc.file_fd);
        const CacheKey key { identity, desc.file_offset, desc.byte_len };
        runtime.cache_key = key;
        runtime.weight_owner = weight_cache().find(key);
        if (!runtime.weight_owner) {
            runtime.weight_owner = std::make_shared<AlignedBytes>();
            const bool direct = direct_io_enabled()
                && desc.file_offset % 4096 == 0 && desc.byte_len % 4096 == 0;
            runtime.weight_owner->resize(desc.byte_len, direct ? 4096 : 64);
            runtime.read_fd = direct
                ? direct_files().resolve(desc.file_fd, identity)
                : desc.file_fd;
            runtime.alternate_read_fd = direct
                ? mirror_files().resolve(desc.file_fd, identity)
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
    projection.traits->vec_dot(
        desc.in_features,
        output,
        0,
        projection.weights + desc.row_bytes * static_cast<std::size_t>(row),
        0,
        projection.activation_data,
        0,
        1);
}

} // namespace

int bw24_cpu_moe_token_impl(
        const bw24_cpu_expert_v1 * experts,
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
    std::call_once(ggml_init_once, [] { ggml_cpu_init(); });
    auto & profile = cpu_profile();
    const auto prepare_start = std::chrono::steady_clock::now();

    const int n_experts = expert_count;
    const int n_embd = experts[0].gate.in_features;
    const int n_ff = experts[0].gate.out_features;
    std::vector<ExpertRuntime> runtime(static_cast<std::size_t>(n_experts));
    for (int expert = 0; expert < n_experts; ++expert) {
        const auto & desc = experts[expert];
        if (desc.gate.in_features != n_embd || desc.up.in_features != n_embd
            || desc.gate.out_features != n_ff || desc.up.out_features != n_ff
            || desc.down.in_features != n_ff || desc.down.out_features != n_embd) {
            throw std::runtime_error("inconsistent CPU expert projection dimensions");
        }
        auto & work = runtime[expert];
        work.gate = prepare_projection(desc.gate, false);
        work.up = prepare_projection(desc.up, false);
        work.down = prepare_projection(desc.down, true);
        work.activation.resize(n_ff);
        work.gate_output.resize(n_ff);
        work.up_output.resize(n_ff);
        work.down_output.resize(n_embd);
    }

    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    std::vector<ProjectionRuntime *> projections;
    projections.reserve(static_cast<std::size_t>(n_experts) * 3);
    for (auto & expert : runtime) {
        projections.push_back(&expert.gate);
        projections.push_back(&expert.up);
        projections.push_back(&expert.down);
    }
    profile.prepare_ns.fetch_add(elapsed_ns(prepare_start), std::memory_order_relaxed);
    const int io_threads = io_thread_count(threads);
    load_projection_weights(projections, io_threads);

    // One persistent OpenMP team covers both input projections, SwiGLU, the down projection,
    // and deterministic expert-order accumulation. Gate and up share one quantized copy of the
    // hidden-state row per activation type instead of re-quantizing it for every expert.
    const auto compute_start = std::chrono::steady_clock::now();
    struct SharedActivation {
        ggml_type type;
        const ggml_type_traits_cpu * traits;
        AlignedBytes values;
    };
    std::vector<SharedActivation> input_activations;
    input_activations.reserve(2);
    for (auto & work : runtime) {
        ProjectionRuntime * input_projections[] = { &work.gate, &work.up };
        for (auto * projection : input_projections) {
            auto shared = std::find_if(
                input_activations.begin(), input_activations.end(),
                [projection](const SharedActivation & candidate) {
                    return candidate.type == projection->activation_type;
                });
            if (shared == input_activations.end()) {
                SharedActivation activation {
                    projection->activation_type,
                    projection->activation_traits,
                    {},
                };
                activation.values.resize(ggml_row_size(activation.type, n_embd));
                activation.traits->from_float(input, activation.values.data, n_embd);
                input_activations.push_back(std::move(activation));
                shared = std::prev(input_activations.end());
            }
            projection->activation_data = shared->values.data;
        }
    }
#pragma omp parallel
    {
#pragma omp for schedule(dynamic, 16)
            for (int task = 0; task < n_experts * n_ff * 2; ++task) {
                const int expert = task / (n_ff * 2);
                const int local = task % (n_ff * 2);
                const bool is_up = local >= n_ff;
                const int row = local % n_ff;
                auto & work = runtime[expert];
                if (is_up) {
                    dot_row(work.up, row, &work.up_output[row]);
                } else {
                    dot_row(work.gate, row, &work.gate_output[row]);
                }
            }

#pragma omp for schedule(static)
            for (int index = 0; index < n_experts * n_ff; ++index) {
                const int expert = index / n_ff;
                const int column = index % n_ff;
                const auto & desc = experts[expert];
                auto & work = runtime[expert];
                const float gate = work.gate_output[column] * desc.gate.scale;
                const float up = work.up_output[column] * desc.up.scale;
                work.activation[column] = (gate / (1.0f + std::exp(-gate))) * up;
            }

#pragma omp for schedule(static)
            for (int expert = 0; expert < n_experts; ++expert) {
                auto & work = runtime[expert];
                work.down.activation_traits->from_float(
                    work.activation.data(), work.down.activation_data,
                    work.down.desc->in_features);
            }

#pragma omp for schedule(dynamic, 16)
            for (int task = 0; task < n_experts * n_embd; ++task) {
                const int expert = task / n_embd;
                const int row = task % n_embd;
                auto & work = runtime[expert];
                dot_row(work.down, row, &work.down_output[row]);
            }

#pragma omp for schedule(static)
            for (int row = 0; row < n_embd; ++row) {
                float sum = 0.0f;
                for (int expert = 0; expert < n_experts; ++expert) {
                    const float scale = experts[expert].route_weight * experts[expert].down.scale;
                    sum = std::fma(runtime[expert].down_output[row], scale, sum);
                }
                output[row] = sum;
            }
    }
    profile.compute_ns.fetch_add(elapsed_ns(compute_start), std::memory_order_relaxed);
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

extern "C" int bw24_cpu_moe_token_v1(
        const bw24_cpu_expert_v1 * experts,
        std::int32_t expert_count,
        const float * input,
        float * output,
        std::int32_t threads,
        char * error,
        std::size_t error_capacity) {
    return bw24_cpu_moe_token_impl(
        experts, expert_count, input, output, threads, error, error_capacity);
}

extern "C" void bw24_cpu_expert_cache_stats_v1(
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

extern "C" void bw24_cpu_expert_profile_stats_v1(
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
