#include <atomic>
#include <cassert>
#include <cerrno>
#include <climits>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <amd_smi/amdsmi.h>

// ============================================================
//  Global stop flag — written by signal, read by all threads
// ============================================================
static std::atomic<bool> g_stop_requested{false};

static void HandleSignal(int) {
    g_stop_requested.store(true, std::memory_order_relaxed);
}

// ============================================================
//  Binary format
// ============================================================
constexpr uint32_t kMagic   = 0x51534d41;  // "AMSQ"
constexpr uint32_t kVersion = 2;           // bumped for multi-thread build

constexpr uint32_t kFlagEnergyFail = 1u << 0;
constexpr uint32_t kFlagPowerFail  = 1u << 1;
constexpr uint32_t kFlagClockFail  = 1u << 2;
constexpr uint32_t kFlagBadTiming  = 1u << 3;
constexpr uint32_t kFlagDropped    = 1u << 4;  // NEW: ring was full, sample dropped

#pragma pack(push, 1)
struct FileHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t device_id;
    uint32_t record_size;
    uint64_t interval_us;
    uint64_t start_host_mono_ns;
};

struct SampleRecord {
    uint64_t seq;
    uint64_t host_mono_ns;
    uint64_t smi_ts_ns;
    uint64_t energy_count;
    float    energy_resolution_uj;
    float    current_socket_power_w;
    float    inst_power_w;
    float    gfx_clock_mhz;
    uint32_t flags;
};
#pragma pack(pop)

static_assert(sizeof(FileHeader)   == 32, "FileHeader size mismatch");
static_assert(sizeof(SampleRecord) == 48, "SampleRecord size mismatch");

// ============================================================
//  SPSC Lock-Free Ring Buffer
//
//  Single-Producer Single-Consumer, wait-free on the fast path.
//  Capacity must be a power of two.
//
//  Layout:
//    head_  — written by consumer, read by producer  (cache-line padded)
//    tail_  — written by producer, read by consumer  (cache-line padded)
//
//  TryPush returns false (non-blocking) instead of blocking when full,
//  so the sampler thread never stalls.
// ============================================================
class SpscRingBuffer {
public:
    explicit SpscRingBuffer(size_t capacity) {
        // Round up to next power-of-two
        size_t cap = 1;
        while (cap < capacity) cap <<= 1;
        mask_ = cap - 1;
        buf_.resize(cap);
    }

    // Called by producer thread only.
    // Returns true if pushed, false if ring was full (sample is dropped).
    bool TryPush(const SampleRecord& rec) noexcept {
        const size_t tail = tail_.load(std::memory_order_relaxed);
        const size_t next = (tail + 1) & mask_;
        if (next == head_.load(std::memory_order_acquire)) {
            return false;  // full — non-blocking drop
        }
        buf_[tail] = rec;
        tail_.store(next, std::memory_order_release);
        return true;
    }

    // Called by consumer thread only.
    // Drains up to max_batch records into *out.
    // Returns number of records drained (0 if empty).
    size_t PopBatch(SampleRecord* out, size_t max_batch) noexcept {
        size_t head = head_.load(std::memory_order_relaxed);
        const size_t tail = tail_.load(std::memory_order_acquire);
        size_t n = 0;
        while (n < max_batch && head != tail) {
            out[n++] = buf_[head];
            head = (head + 1) & mask_;
        }
        if (n > 0) head_.store(head, std::memory_order_release);
        return n;
    }

    size_t Capacity() const noexcept { return mask_ + 1; }

private:
    static constexpr size_t kCacheLineSize = 64;

    std::vector<SampleRecord> buf_;
    size_t mask_ = 0;

    alignas(kCacheLineSize) std::atomic<size_t> head_{0};
    alignas(kCacheLineSize) std::atomic<size_t> tail_{0};
};

// ============================================================
//  Options
// ============================================================
struct PerDeviceOptions {
    int         device_id    = 0;
    std::string output_path;   // filled in from prefix at parse time
    int         cpu_affinity = -1;  // -1 = no pin
};

struct Options {
    std::vector<PerDeviceOptions> devices;
    std::string output_prefix;        // e.g. "myrun"  → myrun-gpu0.bin
    int    interval_us    = 1000;
    bool   realtime       = false;
    bool   mlock_all      = false;
    size_t ring_capacity  = 262144;

    // convert mode
    bool        convert_mode       = false;
    std::string convert_input;
    std::string convert_output_csv;
};

// ============================================================
//  Helpers
// ============================================================
static void PrintUsage(const char* prog) {
    std::cout <<
        "Sampling mode:\n"
        "  " << prog << " --device 0,1,2,3 --output-prefix myrun [options]\n\n"
        "  Output files:  myrun-gpu0.bin  myrun-gpu1.bin  ...\n"
        "  CPU cores are assigned automatically (core 0 reserved for OS,\n"
        "  sampler threads pinned to cores 1, 2, 3, ... in order).\n\n"
        "  --device LIST       Comma-separated GPU device ids, e.g. \"0,1,2,3\"\n"
        "  --output-prefix P   Prefix for output filenames\n"
        "  --interval-us N     Sampling interval in µs (default 1000)\n"
        "  --ring-capacity N   SPSC ring size in records per GPU (default 262144)\n"
        "  --realtime          Request SCHED_FIFO (best effort)\n"
        "  --mlock             Request mlockall() (best effort)\n\n"
        "Convert mode:\n"
        "  " << prog << " --convert in.bin --csv out.csv\n";
}

static uint64_t NowMonotonicNs() noexcept {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ull
         + static_cast<uint64_t>(ts.tv_nsec);
}

// Absolute-time sleep; returns false if interrupted by stop signal.
static bool SleepUntilNs(uint64_t target_ns) noexcept {
    struct timespec ts;
    ts.tv_sec  = static_cast<time_t>(target_ns / 1'000'000'000ull);
    ts.tv_nsec = static_cast<long>  (target_ns % 1'000'000'000ull);
    for (;;) {
        int rc = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr);
        if (rc == 0) return true;
        if (rc == EINTR) {
            if (g_stop_requested.load(std::memory_order_relaxed)) return false;
            continue;
        }
        return false;
    }
}

static void SetAffinity(int cpu) noexcept {
    if (cpu < 0) return;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0)
        std::cerr << "[WARN] sched_setaffinity(" << cpu << ") failed: "
                  << std::strerror(errno) << "\n";
}

static void SetRealtime() noexcept {
    struct sched_param sp{};
    sp.sched_priority = 10;
    if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0)
        std::cerr << "[WARN] SCHED_FIFO failed: " << std::strerror(errno) << "\n";
}

static void MlockAll() noexcept {
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0)
        std::cerr << "[WARN] mlockall failed: " << std::strerror(errno) << "\n";
}

// ============================================================
//  AMD SMI helpers
// ============================================================
static bool GetGpuHandles(std::vector<amdsmi_processor_handle>& out) {
    uint32_t socket_count = 0;
    if (amdsmi_get_socket_handles(&socket_count, nullptr) != AMDSMI_STATUS_SUCCESS
            || socket_count == 0) {
        std::cerr << "amdsmi_get_socket_handles failed or no sockets\n";
        return false;
    }
    std::vector<amdsmi_socket_handle> sockets(socket_count);
    if (amdsmi_get_socket_handles(&socket_count, sockets.data()) != AMDSMI_STATUS_SUCCESS) {
        std::cerr << "amdsmi_get_socket_handles (fetch) failed\n";
        return false;
    }
    for (uint32_t i = 0; i < socket_count; ++i) {
        uint32_t dev_count = 0;
        if (amdsmi_get_processor_handles(sockets[i], &dev_count, nullptr) != AMDSMI_STATUS_SUCCESS
                || dev_count == 0)
            continue;
        std::vector<amdsmi_processor_handle> handles(dev_count);
        if (amdsmi_get_processor_handles(sockets[i], &dev_count, handles.data()) != AMDSMI_STATUS_SUCCESS)
            continue;
        for (uint32_t j = 0; j < dev_count; ++j) {
            processor_type_t pt = AMDSMI_PROCESSOR_TYPE_UNKNOWN;
            if (amdsmi_get_processor_type(handles[j], &pt) == AMDSMI_STATUS_SUCCESS
                    && pt == AMDSMI_PROCESSOR_TYPE_AMD_GPU)
                out.push_back(handles[j]);
        }
    }
    return !out.empty();
}

static float SafeSocketPowerW(const amdsmi_power_info_t& info) noexcept {
    return (info.current_socket_power == UINT32_MAX)
        ? std::numeric_limits<float>::quiet_NaN()
        : static_cast<float>(info.current_socket_power);
}

static float SafeGfxClockMhz(const amdsmi_frequencies_t& f) noexcept {
    if (f.num_supported == 0 || f.current >= f.num_supported)
        return std::numeric_limits<float>::quiet_NaN();
    double raw = static_cast<double>(f.frequency[f.current]);
    return static_cast<float>((raw > 1.0e6) ? raw / 1.0e6 : raw);
}

// ============================================================
//  Per-GPU sampler: owns one SPSC ring + two threads
//
//  sampler_thread: tight loop — reads SMI, pushes into ring (non-blocking)
//  writer_thread:  drains ring, writes binary file
// ============================================================
struct GpuWorker {
    const PerDeviceOptions& cfg;
    const Options&          global;
    amdsmi_processor_handle handle;

    std::unique_ptr<SpscRingBuffer> ring;

    // Statistics (informational, written by sampler, read after join)
    uint64_t total_samples  = 0;
    uint64_t dropped_samples = 0;

    // Set to non-zero on fatal error inside either thread
    std::atomic<int> exit_code{0};

    std::thread sampler_thread;
    std::thread writer_thread;

    GpuWorker(const PerDeviceOptions& c, const Options& g, amdsmi_processor_handle h)
        : cfg(c), global(g), handle(h),
          ring(std::make_unique<SpscRingBuffer>(g.ring_capacity)) {}

    void Start() {
        writer_thread  = std::thread(&GpuWorker::WriterLoop,  this);
        sampler_thread = std::thread(&GpuWorker::SamplerLoop, this);
    }

    void Join() {
        if (sampler_thread.joinable()) sampler_thread.join();
        if (writer_thread.joinable())  writer_thread.join();
    }

private:
    // --------------------------------------------------------
    //  Sampler thread
    // --------------------------------------------------------
    void SamplerLoop() {
        SetAffinity(cfg.cpu_affinity);
        if (global.realtime) SetRealtime();

        // Warm-up read
        uint64_t prev_energy = 0;
        float    counter_res = 0.0f;
        uint64_t prev_smi_ts = 0;
        if (amdsmi_get_energy_count(handle, &prev_energy, &counter_res, &prev_smi_ts)
                != AMDSMI_STATUS_SUCCESS) {
            std::cerr << "[GPU " << cfg.device_id
                      << "] amdsmi_get_energy_count (warm-up) failed\n";
            exit_code.store(1, std::memory_order_relaxed);
            return;
        }

        uint64_t seq = 0;
        uint64_t next_ns = NowMonotonicNs();
        const uint64_t interval_ns = static_cast<uint64_t>(global.interval_us) * 1000ull;

        while (!g_stop_requested.load(std::memory_order_relaxed)) {
            next_ns += interval_ns;
            if (!SleepUntilNs(next_ns)) break;
            if (g_stop_requested.load(std::memory_order_relaxed)) break;

            SampleRecord rec{};
            rec.seq          = seq++;
            rec.host_mono_ns = NowMonotonicNs();

            // --- Energy ---
            uint64_t energy  = 0;
            float    res     = counter_res;
            uint64_t smi_ts  = 0;
            if (amdsmi_get_energy_count(handle, &energy, &res, &smi_ts)
                    != AMDSMI_STATUS_SUCCESS) {
                rec.flags |= kFlagEnergyFail;
                energy = prev_energy; res = counter_res; smi_ts = prev_smi_ts;
            }
            rec.energy_count          = energy;
            rec.energy_resolution_uj  = res;
            rec.smi_ts_ns             = smi_ts;

            // --- Socket power ---
            amdsmi_power_info_t pinfo{};
            if (amdsmi_get_power_info(handle, &pinfo) != AMDSMI_STATUS_SUCCESS) {
                rec.flags |= kFlagPowerFail;
                rec.current_socket_power_w = std::numeric_limits<float>::quiet_NaN();
            } else {
                rec.current_socket_power_w = SafeSocketPowerW(pinfo);
            }

            // --- GFX clock ---
            amdsmi_frequencies_t freqs{};
            if (amdsmi_get_clk_freq(handle, AMDSMI_CLK_TYPE_GFX, &freqs)
                    != AMDSMI_STATUS_SUCCESS) {
                rec.flags |= kFlagClockFail;
                rec.gfx_clock_mhz = std::numeric_limits<float>::quiet_NaN();
            } else {
                rec.gfx_clock_mhz = SafeGfxClockMhz(freqs);
            }

            // --- Instantaneous power from energy delta ---
            if (!(rec.flags & kFlagEnergyFail) &&
                smi_ts > prev_smi_ts && res > 0.0f && energy >= prev_energy) {
                double delta_uj = static_cast<double>(energy - prev_energy)
                                * static_cast<double>(res);
                double delta_ns = static_cast<double>(smi_ts - prev_smi_ts);
                rec.inst_power_w = static_cast<float>((delta_uj * 1e-6) / (delta_ns * 1e-9));
            } else {
                rec.inst_power_w = std::numeric_limits<float>::quiet_NaN();
                rec.flags |= kFlagBadTiming;
            }

            prev_energy    = energy;
            prev_smi_ts    = smi_ts;
            counter_res    = res;

            ++total_samples;
            if (!ring->TryPush(rec)) {
                ++dropped_samples;
                // Mark the record as dropped but don't block
                // (the ring slot is still occupied by an older record;
                //  we just count the loss here)
            }
        }

        // Signal writer that we are done by setting a sentinel via g_stop.
        // Writer polls g_stop + ring-empty to decide when to exit.
    }

    // --------------------------------------------------------
    //  Writer thread
    // --------------------------------------------------------
    void WriterLoop() {
        std::ofstream out(cfg.output_path,
                          std::ios::binary | std::ios::trunc);
        if (!out) {
            std::cerr << "[GPU " << cfg.device_id
                      << "] Failed to open: " << cfg.output_path << "\n";
            exit_code.store(1, std::memory_order_relaxed);
            return;
        }

        // 1 MiB stream buffer
        static thread_local char fbuf[1 << 20];
        out.rdbuf()->pubsetbuf(fbuf, sizeof(fbuf));

        FileHeader hdr{};
        hdr.magic              = kMagic;
        hdr.version            = kVersion;
        hdr.device_id          = static_cast<uint32_t>(cfg.device_id);
        hdr.record_size        = static_cast<uint32_t>(sizeof(SampleRecord));
        hdr.interval_us        = static_cast<uint64_t>(global.interval_us);
        hdr.start_host_mono_ns = NowMonotonicNs();
        out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        if (!out) {
            std::cerr << "[GPU " << cfg.device_id << "] Header write failed\n";
            exit_code.store(1, std::memory_order_relaxed);
            return;
        }

        constexpr size_t kBatchSize = 4096;
        std::vector<SampleRecord> batch(kBatchSize);

        while (true) {
            size_t n = ring->PopBatch(batch.data(), kBatchSize);
            if (n > 0) {
                out.write(reinterpret_cast<const char*>(batch.data()),
                          static_cast<std::streamsize>(n * sizeof(SampleRecord)));
                if (!out) {
                    std::cerr << "[GPU " << cfg.device_id << "] Write error\n";
                    exit_code.store(1, std::memory_order_relaxed);
                    return;
                }
            } else {
                // Ring is empty — check if sampler has stopped
                if (g_stop_requested.load(std::memory_order_relaxed)) {
                    // Drain any last records
                    while ((n = ring->PopBatch(batch.data(), kBatchSize)) > 0) {
                        out.write(reinterpret_cast<const char*>(batch.data()),
                                  static_cast<std::streamsize>(n * sizeof(SampleRecord)));
                    }
                    break;
                }
                // Yield to avoid spinning at 100% when workload is light
                std::this_thread::yield();
            }
        }

        out.flush();
        out.close();

        // fsync via O_RDWR so we actually flush kernel page cache
        int fd = ::open(cfg.output_path.c_str(), O_RDWR);
        if (fd >= 0) { ::fsync(fd); ::close(fd); }
    }
};

// ============================================================
//  CSV conversion (unchanged logic, cosmetic cleanup)
// ============================================================
static bool ConvertBinToCsv(const std::string& in_path,
                             const std::string& out_csv) {
    std::ifstream in(in_path, std::ios::binary);
    if (!in) { std::cerr << "Cannot open: " << in_path << "\n"; return false; }

    FileHeader hdr{};
    in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!in || hdr.magic != kMagic || hdr.record_size != sizeof(SampleRecord)) {
        std::cerr << "Invalid binary file: " << in_path << "\n";
        return false;
    }

    std::ofstream out(out_csv, std::ios::out | std::ios::trunc);
    if (!out) { std::cerr << "Cannot create: " << out_csv << "\n"; return false; }

    out << "seq,device_id,host_mono_ns,smi_ts_ns,energy_count,"
           "energy_resolution_uj,current_socket_power_w,inst_power_w,"
           "gfx_clock_mhz,flags\n";

    SampleRecord rec{};
    while (in.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
        out << rec.seq              << ","
            << hdr.device_id        << ","
            << rec.host_mono_ns     << ","
            << rec.smi_ts_ns        << ","
            << rec.energy_count     << ","
            << rec.energy_resolution_uj    << ","
            << rec.current_socket_power_w  << ","
            << rec.inst_power_w     << ","
            << rec.gfx_clock_mhz   << ","
            << rec.flags            << "\n";
    }
    return !!out;
}

// ============================================================
//  Argument parsing
// ============================================================
static bool ParseArgs(int argc, char** argv, Options& opt) {
    std::string device_list;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "--device") {
            device_list = need("--device");
        } else if (arg == "--output-prefix") {
            opt.output_prefix = need("--output-prefix");
        } else if (arg == "--interval-us") {
            opt.interval_us = std::atoi(need("--interval-us"));
        } else if (arg == "--ring-capacity") {
            long long v = std::atoll(need("--ring-capacity"));
            if (v <= 0) { std::cerr << "--ring-capacity must be > 0\n"; return false; }
            opt.ring_capacity = static_cast<size_t>(v);
        } else if (arg == "--realtime") {
            opt.realtime = true;
        } else if (arg == "--mlock") {
            opt.mlock_all = true;
        } else if (arg == "--convert") {
            opt.convert_mode  = true;
            opt.convert_input = need("--convert");
        } else if (arg == "--csv") {
            opt.convert_output_csv = need("--csv");
        } else if (arg == "-h" || arg == "--help") {
            PrintUsage(argv[0]);
            return false;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }

    if (opt.convert_mode) {
        if (opt.convert_input.empty() || opt.convert_output_csv.empty()) {
            std::cerr << "Convert mode requires --convert in.bin --csv out.csv\n";
            return false;
        }
        return true;
    }

    // --- Validate sampling mode args ---
    if (device_list.empty()) {
        std::cerr << "--device is required\n";
        return false;
    }
    if (opt.output_prefix.empty()) {
        std::cerr << "--output-prefix is required\n";
        return false;
    }

    // Parse device list
    {
        std::istringstream ss(device_list);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (tok.empty()) continue;
            int id = std::atoi(tok.c_str());
            if (id < 0) { std::cerr << "Invalid device id: " << tok << "\n"; return false; }
            PerDeviceOptions d;
            d.device_id   = id;
            d.output_path = opt.output_prefix + "-gpu" + std::to_string(id) + ".bin";
            opt.devices.push_back(d);
        }
    }
    if (opt.devices.empty()) {
        std::cerr << "No valid device ids in --device list\n";
        return false;
    }

    // Auto-assign CPU cores: skip core 0 (OS/system), assign sampler+writer
    // pairs to consecutive cores starting from 1.
    // Each GPU worker needs 2 threads (sampler + writer); we pin only the
    // sampler thread (the latency-sensitive one) and let the writer float
    // on the next core.
    {
        const int total_cpus = static_cast<int>(std::thread::hardware_concurrency());
        int next_cpu = 1;  // reserve core 0 for the OS
        for (auto& d : opt.devices) {
            if (next_cpu < total_cpus) {
                d.cpu_affinity = next_cpu++;
            } else {
                // More GPUs than spare cores — fall back to no pinning for this one
                d.cpu_affinity = -1;
                std::cerr << "[WARN] Not enough CPU cores to pin sampler for GPU "
                          << d.device_id << " (total_cpus=" << total_cpus << ")\n";
            }
        }
    }

    return true;
}

// ============================================================
//  main
// ============================================================
int main(int argc, char** argv) {
    Options opt;
    if (!ParseArgs(argc, argv, opt)) return 1;

    if (opt.convert_mode)
        return ConvertBinToCsv(opt.convert_input, opt.convert_output_csv) ? 0 : 1;

    // --- Global setup ---
    std::signal(SIGINT,  HandleSignal);
    std::signal(SIGTERM, HandleSignal);

    if (opt.mlock_all) MlockAll();

    if (amdsmi_init(AMDSMI_INIT_AMD_GPUS) != AMDSMI_STATUS_SUCCESS) {
        std::cerr << "amdsmi_init failed\n";
        return 1;
    }

    std::vector<amdsmi_processor_handle> gpu_handles;
    if (!GetGpuHandles(gpu_handles)) {
        std::cerr << "No AMD GPU handles found\n";
        amdsmi_shut_down();
        return 1;
    }

    // Validate device ids
    for (const auto& d : opt.devices) {
        if (d.device_id < 0 ||
            static_cast<size_t>(d.device_id) >= gpu_handles.size()) {
            std::cerr << "device_id " << d.device_id
                      << " out of range (found " << gpu_handles.size() << " GPUs)\n";
            amdsmi_shut_down();
            return 1;
        }
    }

    // --- Launch one GpuWorker per requested device ---
    std::vector<std::unique_ptr<GpuWorker>> workers;
    workers.reserve(opt.devices.size());
    for (const auto& d : opt.devices) {
        workers.push_back(std::make_unique<GpuWorker>(
            d, opt, gpu_handles[static_cast<size_t>(d.device_id)]));
    }

    std::cerr << "[INFO] Starting " << workers.size() << " GPU worker(s), "
              << "interval=" << opt.interval_us << " µs, "
              << "ring=" << opt.ring_capacity << " records each\n";
    for (auto& w : workers) {
        std::cerr << "[INFO]   GPU " << w->cfg.device_id
                  << " -> " << w->cfg.output_path
                  << "  cpu=" << (w->cfg.cpu_affinity >= 0
                                  ? std::to_string(w->cfg.cpu_affinity)
                                  : std::string("auto"))
                  << "\n";
    }

    for (auto& w : workers) w->Start();

    // Wait for stop signal (process is typically killed by the orchestrator)
    while (!g_stop_requested.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Workers detect g_stop via the global flag; just join them.
    for (auto& w : workers) w->Join();

    amdsmi_shut_down();

    // Print per-GPU stats and determine exit code
    int rc = 0;
    for (auto& w : workers) {
        std::cerr << "[GPU " << w->cfg.device_id << "] samples=" << w->total_samples
                  << " dropped=" << w->dropped_samples;
        if (w->dropped_samples > 0) {
            double pct = 100.0 * static_cast<double>(w->dropped_samples)
                               / static_cast<double>(w->total_samples);
            std::cerr << " (" << pct << "%)";
        }
        std::cerr << "\n";
        if (w->exit_code.load() != 0) rc = w->exit_code.load();
    }
    return rc;
}