// ROCPROFWRAP_NO_COUNTERS: build telemetry only (power/clock/busy/temp via amd-smi),
// with no rocprofiler-sdk dependency. Needed on machines whose ROCm predates the
// rocprofiler-sdk v1 counter API used below (rocprofiler_counter_config_id_t and
// friends are ROCm 7.x only; ROCm 6.x ships the older profile_config naming and
// librocprofiler-sdk.so.0). The counter CSV columns are still emitted, zero-filled,
// so the output schema is identical either way.
#include "amd_smi/amdsmi.h"
#ifndef ROCPROFWRAP_NO_COUNTERS
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/counter_config.h>
#include <rocprofiler-sdk/device_counting_service.h>
#endif
#include <fstream>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <thread_pool.h>
#include <unistd.h>
#include <deque>
#include <map>
#include <unordered_map>
#include <atomic>
#include <sstream>
#include <stdexcept>
#include <vector>

#define RYML_SINGLE_HDR_DEFINE_NOW
#include <ryml_all.hpp>

// ---------------------------------------------------------------------------
// Macros
// ---------------------------------------------------------------------------
#ifndef ROCPROFWRAP_NO_COUNTERS
#define ROCPROFILER_CALL(result, msg)                                          \
    {                                                                          \
        rocprofiler_status_t CHECKSTATUS = result;                             \
        if (CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS &&                       \
            CHECKSTATUS != ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED) {          \
            const char* _status_msg =                                          \
                rocprofiler_get_status_string(CHECKSTATUS);                    \
            std::cerr << "[" #result "] " << msg                              \
                      << " failed with error code " << CHECKSTATUS            \
                      << ": " << _status_msg << std::endl;                    \
        }                                                                      \
    }
#endif

// ---------------------------------------------------------------------------
// Global variables
// ---------------------------------------------------------------------------
std::ofstream output;
hipDeviceProp_t devProp;

int duration, stop = 0;
uint32_t device;

// amd-smi state
amdsmi_processor_handle gpu_processor_handle;
amdsmi_power_info_t     power_info;
amdsmi_clk_info_t       gfx_clk_info;
amdsmi_clk_info_t       mem_clk_info;
amdsmi_clk_info_t       df_clk_info;
uint32_t                gpu_busy_percent = 0;
amdsmi_gpu_metrics_t    gpu_metrics{};

// Energy tracking
uint64_t currEnergy;
uint64_t prevEnergy = 0;
float    resolution;
float    ePower = 0.0;

// Energy history for windowed average power
// Stores (wall_clock_ns, raw_energy_count) pairs at each 1ms sample
struct EnergySnapshot {
    uint64_t wall_ns;
    uint64_t energy;
};
std::deque<EnergySnapshot> energy_history;

// Timestamps
uint64_t startTime, timeStamp1, timeStamp2, etimeStamp, prevTimeStamp;
uint64_t profItr = 0;

// rocprofiler-sdk state
#ifndef ROCPROFWRAP_NO_COUNTERS
rocprofiler_context_id_t        prof_ctx     = {};
rocprofiler_buffer_id_t         prof_buf     = {};
rocprofiler_counter_config_id_t prof_config  = {.handle = 0};
rocprofiler_agent_id_t          prof_agent_id = {};
std::vector<rocprofiler_counter_record_t> prof_records;
std::map<uint64_t, std::string>           counter_id_to_name;
size_t expected_record_count = 0;
size_t actual_record_count   = 0;   // updated each sample call
#endif
// Kept unconditionally so the getData()/writeData() branches read the same in
// both builds; permanently false when counters are compiled out.
bool   rocprof_initialized   = false;

ThreadPool pool(1);

// Default hardware counters (can be overridden from JSON config)
std::vector<std::string> hwCounters = {
    "SQ_BUSY_CYCLES",
};

// ---------------------------------------------------------------------------
// Utility: read file contents (for YAML parsing)
// ---------------------------------------------------------------------------
template <class CharContainer>
size_t file_get_contents(const char *filename, CharContainer *v) {

  ::FILE *fp = ::fopen(filename, "rb");
  C4_CHECK_MSG(fp != nullptr, "could not open file");
  ::fseek(fp, 0, SEEK_END);
  long sz = ::ftell(fp);
  v->resize(static_cast<typename CharContainer::size_type>(sz));
  if (sz) {
    ::rewind(fp);
    size_t ret = ::fread(&(*v)[0], 1, v->size(), fp);
    C4_CHECK(ret == (size_t)sz);
  }
  ::fclose(fp);
  return v->size();
}

void parseYaml(const char *filename) {

  std::string contents;
  file_get_contents<std::string>(filename, &contents);
  ryml::Tree tree = ryml::parse_in_arena(ryml::to_csubstr(contents));
  std::cout << tree[0]["gpu-ids"].num_children() << std::endl;
}

// ---------------------------------------------------------------------------
// Signal handler
// ---------------------------------------------------------------------------
void signal_callback_handler(int signum) { stop = 1; }

// ---------------------------------------------------------------------------
// amd-smi: Initialization
// ---------------------------------------------------------------------------
void smiInit() {
    amdsmi_status_t status;

    // Initialize amd-smi for AMD GPUs
    status = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
    if (status != AMDSMI_STATUS_SUCCESS) {
        std::cerr << "amdsmi_init failed with status " << status << std::endl;
        exit(1);
    }

    // Get socket handles
    uint32_t socket_count = 0;
    status = amdsmi_get_socket_handles(&socket_count, nullptr);
    if (status != AMDSMI_STATUS_SUCCESS || socket_count == 0) {
        std::cerr << "Failed to get socket count" << std::endl;
        exit(1);
    }

    std::vector<amdsmi_socket_handle> sockets(socket_count);
    status = amdsmi_get_socket_handles(&socket_count, sockets.data());
    if (status != AMDSMI_STATUS_SUCCESS) {
        std::cerr << "Failed to get socket handles" << std::endl;
        exit(1);
    }

    // Enumerate all GPU processors across sockets
    std::vector<amdsmi_processor_handle> all_gpus;
    for (uint32_t s = 0; s < socket_count; s++) {
        uint32_t proc_count = 0;
        status = amdsmi_get_processor_handles(sockets[s], &proc_count, nullptr);
        if (status != AMDSMI_STATUS_SUCCESS) continue;

        std::vector<amdsmi_processor_handle> procs(proc_count);
        status = amdsmi_get_processor_handles(sockets[s], &proc_count, procs.data());
        if (status != AMDSMI_STATUS_SUCCESS) continue;

        for (uint32_t p = 0; p < proc_count; p++) {
            processor_type_t ptype;
            if (amdsmi_get_processor_type(procs[p], &ptype) == AMDSMI_STATUS_SUCCESS) {
                if (ptype == AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
                    all_gpus.push_back(procs[p]);
                }
            }
        }
    }

    if (device >= all_gpus.size()) {
        std::cerr << "GPU device " << device << " not found. Found "
                  << all_gpus.size() << " GPUs." << std::endl;
        exit(1);
    }

    gpu_processor_handle = all_gpus[device];
    std::cout << "amd-smi initialized, using GPU device " << device << std::endl;
}

// ---------------------------------------------------------------------------
// rocprofiler-sdk: Helper functions
// ---------------------------------------------------------------------------
#ifndef ROCPROFWRAP_NO_COUNTERS

// Enumerate available GPU agents
std::vector<rocprofiler_agent_v0_t> get_gpu_agents() {
    std::vector<rocprofiler_agent_v0_t> agents;
    rocprofiler_query_available_agents(
        ROCPROFILER_AGENT_INFO_VERSION_0,
        [](rocprofiler_agent_version_t version, const void** agents_arr,
           size_t num_agents, void* udata) -> rocprofiler_status_t {
            if (version != ROCPROFILER_AGENT_INFO_VERSION_0)
                return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI;
            auto* out = static_cast<std::vector<rocprofiler_agent_v0_t>*>(udata);
            for (size_t i = 0; i < num_agents; ++i) {
                const auto* agent =
                    static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]);
                if (agent->type == ROCPROFILER_AGENT_TYPE_GPU)
                    out->emplace_back(*agent);
            }
            return ROCPROFILER_STATUS_SUCCESS;
        },
        sizeof(rocprofiler_agent_t),
        static_cast<void*>(&agents));
    return agents;
}

// Query supported counters for a given agent
std::unordered_map<std::string, rocprofiler_counter_id_t>
get_supported_counters(rocprofiler_agent_id_t agent) {
    std::unordered_map<std::string, rocprofiler_counter_id_t> out;
    std::vector<rocprofiler_counter_id_t> gpu_counters;

    rocprofiler_iterate_agent_supported_counters(
        agent,
        [](rocprofiler_agent_id_t, rocprofiler_counter_id_t* counters,
           size_t num_counters, void* user_data) -> rocprofiler_status_t {
            auto* vec =
                static_cast<std::vector<rocprofiler_counter_id_t>*>(user_data);
            for (size_t i = 0; i < num_counters; i++)
                vec->push_back(counters[i]);
            return ROCPROFILER_STATUS_SUCCESS;
        },
        static_cast<void*>(&gpu_counters));

    for (auto& counter : gpu_counters) {
        rocprofiler_counter_info_v0_t info;
        if (rocprofiler_query_counter_info(
                counter, ROCPROFILER_COUNTER_INFO_VERSION_0,
                static_cast<void*>(&info)) == ROCPROFILER_STATUS_SUCCESS) {
            out.emplace(info.name, counter);
        }
    }
    return out;
}

// Get number of record instances for a counter
size_t get_counter_record_count(rocprofiler_counter_id_t counter) {
    rocprofiler_counter_info_v1_t info;
    if (rocprofiler_query_counter_info(
            counter, ROCPROFILER_COUNTER_INFO_VERSION_1,
            static_cast<void*>(&info)) == ROCPROFILER_STATUS_SUCCESS) {
        return info.dimensions_instances_count;
    }
    return 1;
}

// ---------------------------------------------------------------------------
// rocprofiler-sdk: Device counting service callback
// ---------------------------------------------------------------------------
void set_profile_callback(rocprofiler_context_id_t        ctx,
                           rocprofiler_agent_id_t,
                           rocprofiler_device_counting_agent_cb_t set_config,
                           void*) {
    if (prof_config.handle != 0) {
        set_config(ctx, prof_config);
    }
}

// ---------------------------------------------------------------------------
// rocprofiler-sdk: Initialization (called inside tool_init)
// ---------------------------------------------------------------------------
void rocprofSetup() {
    auto agents = get_gpu_agents();
    if (agents.empty()) {
        std::cerr << "rocprofiler-sdk: No GPU agents found" << std::endl;
        return;
    }

    if (device >= agents.size()) {
        std::cerr << "rocprofiler-sdk: GPU device " << device
                  << " not found" << std::endl;
        return;
    }

    prof_agent_id = agents[device].id;

    // Look up requested counters by name
    auto supported = get_supported_counters(prof_agent_id);
    std::vector<rocprofiler_counter_id_t> counter_ids;
    expected_record_count = 0;

    for (auto& name : hwCounters) {
        if (name.empty()) continue;
        auto it = supported.find(name);
        if (it != supported.end()) {
            counter_ids.push_back(it->second);
            counter_id_to_name[it->second.handle] = name;
            expected_record_count += get_counter_record_count(it->second);
        } else {
            std::cerr << "Warning: Counter '" << name
                      << "' not supported on this GPU" << std::endl;
        }
    }

    if (counter_ids.empty()) {
        std::cerr << "No valid counters found for profiling" << std::endl;
        return;
    }

    // Create counter configuration (profile)
    ROCPROFILER_CALL(
        rocprofiler_create_counter_config(prof_agent_id, counter_ids.data(),
                                          counter_ids.size(), &prof_config),
        "create counter config");

    // Create context
    ROCPROFILER_CALL(rocprofiler_create_context(&prof_ctx),
                     "create context");

    // Create buffer (required even for synchronous sampling)
    rocprofiler_callback_thread_t client_thread = {};
    ROCPROFILER_CALL(
        rocprofiler_create_buffer(
            prof_ctx, 4096, 2048, ROCPROFILER_BUFFER_POLICY_LOSSLESS,
            [](rocprofiler_context_id_t, rocprofiler_buffer_id_t,
               rocprofiler_record_header_t**, size_t, void*, uint64_t) {},
            nullptr, &prof_buf),
        "create buffer");

    ROCPROFILER_CALL(rocprofiler_create_callback_thread(&client_thread),
                     "create callback thread");
    ROCPROFILER_CALL(rocprofiler_assign_callback_thread(prof_buf, client_thread),
                     "assign callback thread");

    // Configure device counting service
    ROCPROFILER_CALL(
        rocprofiler_configure_device_counting_service(
            prof_ctx, prof_buf, prof_agent_id,
            set_profile_callback, nullptr),
        "configure device counting");

    // Pre-allocate record buffer
    prof_records.resize(expected_record_count);

    // Start context once; keep it running for the lifetime of profiling
    ROCPROFILER_CALL(rocprofiler_start_context(prof_ctx), "start context");

    rocprof_initialized = true;
    std::cout << "rocprofiler-sdk initialized with " << counter_ids.size()
              << " counters (" << expected_record_count << " record slots)"
              << std::endl;
}

// ---------------------------------------------------------------------------
// rocprofiler-sdk: Tool registration callbacks
// ---------------------------------------------------------------------------
int tool_init_callback(rocprofiler_client_finalize_t, void*) {
    // Setup rocprofiler with the globally-set device index
    rocprofSetup();
    return 0;
}

void tool_fini_callback(void*) {
    if (rocprof_initialized) {
        rocprofiler_stop_context(prof_ctx);
        rocprof_initialized = false;
    }
}

// The rocprofiler_configure entry point (discovered by rocprofiler-sdk)
extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id) {
    id->name = "gpuprof";

    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t),
        &tool_init_callback,
        &tool_fini_callback,
        nullptr};

    return &cfg;
}

#endif  // ROCPROFWRAP_NO_COUNTERS

// ---------------------------------------------------------------------------
// CSV header
// ---------------------------------------------------------------------------
void header(std::ofstream &output) {
    // Power columns first, then gfx_clk, mem_clk, df_clk, gpu_busy, and temperature, then counter columns, then timestamp
    output << "socket_power,curr_socket_power,power_from_e,avg_power_2ms,avg_power_5ms,avg_power_10ms,"
              "gfx_clk,mem_clk,df_clk,gpu_busy,temp_edge_C,temp_hotspot_C,temp_mem_C,";

    for (size_t i = 0; i < hwCounters.size(); i++) {
        if (!hwCounters[i].empty())
            output << hwCounters[i] << ",";
    }

    output << "timestamp\n";
}

// ---------------------------------------------------------------------------
// Data collection
// ---------------------------------------------------------------------------
void getData() {
    // --- amd-smi: power, energy ---
    amdsmi_get_power_info(gpu_processor_handle, &power_info);
    amdsmi_get_energy_count(gpu_processor_handle, &currEnergy,
                            &resolution, &etimeStamp);

    // Record snapshot for windowed power calculation.
    // etimeStamp from amd-smi is in nanoseconds (resolution 1 ns per docs).
    // Using the hardware timestamp paired with currEnergy ensures ΔE and Δt
    // are from the same hardware source, avoiding wall-clock poll-jitter error.
    energy_history.push_back({etimeStamp, currEnergy});
    while (energy_history.size() > 1 &&
           etimeStamp - energy_history.front().wall_ns > 20000000ULL)
        energy_history.pop_front();

    // --- amd-smi: GFX clock (MHz) ---
    amdsmi_status_t clk_status = amdsmi_get_clock_info(
        gpu_processor_handle, AMDSMI_CLK_TYPE_GFX, &gfx_clk_info);
    if (clk_status != AMDSMI_STATUS_SUCCESS) {
        gfx_clk_info.clk = 0;
    }

    // --- amd-smi: memory clock (MHz) ---
    amdsmi_status_t mem_clk_status = amdsmi_get_clock_info(
        gpu_processor_handle, AMDSMI_CLK_TYPE_MEM, &mem_clk_info);
    if (mem_clk_status != AMDSMI_STATUS_SUCCESS) {
        mem_clk_info.clk = 0;
    }

    // --- amd-smi: Data Fabric clock (MHz) ---
    amdsmi_status_t df_clk_status = amdsmi_get_clock_info(
        gpu_processor_handle, AMDSMI_CLK_TYPE_DF, &df_clk_info);
    if (df_clk_status != AMDSMI_STATUS_SUCCESS || df_clk_info.clk == UINT32_MAX) {
        // UINT32_MAX = unsupported marker per amd-smi docs (same convention as
        // curr_socket_power below) -- DF clock is not exposed via this call on
        // every ASIC (e.g. gfx950/MI350X).
        df_clk_info.clk = 0;
    }

    // --- amd-smi: GPU busy percent ---
    amdsmi_status_t busy_status = amdsmi_get_gpu_busy_percent(
        gpu_processor_handle, &gpu_busy_percent);
    if (busy_status != AMDSMI_STATUS_SUCCESS) {
        gpu_busy_percent = 0;
    }

    // --- amd-smi: GPU metrics table (temperature) ---
    if (amdsmi_get_gpu_metrics_info(gpu_processor_handle, &gpu_metrics) !=
        AMDSMI_STATUS_SUCCESS) {
        gpu_metrics = {};  // zero-fill on failure, consistent with clk/busy handling above
    }

    // --- rocprofiler-sdk: hardware counters ---
#ifndef ROCPROFWRAP_NO_COUNTERS
    if (rocprof_initialized) {
        actual_record_count = prof_records.size();
        rocprofiler_sample_device_counting_service(
            prof_ctx, {}, ROCPROFILER_COUNTER_FLAG_NONE,
            prof_records.data(), &actual_record_count);
    }
#endif
}

// ---------------------------------------------------------------------------
// Data output (CSV row)
// ---------------------------------------------------------------------------
void writeData(std::ofstream &output) {
    if (profItr > 0) {
        ePower = resolution * (currEnergy - prevEnergy) / 1000000.0 /
                 ((timeStamp1 - prevTimeStamp) / 1000000000.0);
    }

    // Power info from amd-smi
    // socket_power (uint64, W) — general power reading
    // current_socket_power (uint32, W) — MI300+ series
    uint64_t socket_pwr = power_info.socket_power;
    uint32_t curr_pwr   = power_info.current_socket_power;

    // Handle unsupported markers (UINT32_MAX = unsupported per amd-smi docs)
    if (curr_pwr == UINT32_MAX) curr_pwr = 0;

    // Windowed average power over the last N ms.
    // Uses etimeStamp (hardware-paired with currEnergy) as the "now" reference
    // so ΔE and Δt both come from the same hardware source.
    // history is sorted oldest→newest; always excludes the current entry.
    uint64_t hw_now = etimeStamp;  // already in nanoseconds per amd-smi docs
    auto windowed_power = [&](uint64_t window_ns) -> float {
        if (energy_history.size() < 2) return 0.0f;
        uint64_t target = hw_now - window_ns;
        const EnergySnapshot* best = nullptr;
        uint64_t best_dist = UINT64_MAX;
        for (size_t i = 0; i + 1 < energy_history.size(); i++) {
            uint64_t t = energy_history[i].wall_ns;
            uint64_t dist = (t >= target) ? (t - target) : (target - t);
            if (dist < best_dist) { best_dist = dist; best = &energy_history[i]; }
        }
        if (!best || hw_now <= best->wall_ns) return 0.0f;
        uint64_t dt_ns = hw_now - best->wall_ns;
        uint64_t de    = (currEnergy >= best->energy) ? (currEnergy - best->energy) : 0;
        return resolution * (float)de / 1e6f / ((float)dt_ns / 1e9f);
    };

    float avg2  = windowed_power(2000000ULL);   //  2 ms
    float avg5  = windowed_power(5000000ULL);   //  5 ms
    float avg10 = windowed_power(10000000ULL);  // 10 ms

    output << socket_pwr << ","
           << curr_pwr << ","
           << ePower << ","
           << avg2 << ","
           << avg5 << ","
           << avg10 << ","
           << gfx_clk_info.clk << ","
           << mem_clk_info.clk << ","
           << df_clk_info.clk << ","
           << gpu_busy_percent << ",";

    auto temp_or_zero = [](uint16_t raw) -> uint16_t {
        return raw == UINT16_MAX ? 0 : raw;
    };
    output << temp_or_zero(gpu_metrics.temperature_edge) << ","
           << temp_or_zero(gpu_metrics.temperature_hotspot) << ","
           << temp_or_zero(gpu_metrics.temperature_mem) << ",";

    // Hardware counter values (zero-filled when unavailable, so the column
    // layout is the same whether or not counters were compiled in)
#ifndef ROCPROFWRAP_NO_COUNTERS
    if (rocprof_initialized) {
        // Aggregate counter values by counter name (sum across instances)
        std::map<std::string, double> aggregated;
        for (auto& name : hwCounters) {
            if (!name.empty()) aggregated[name] = 0.0;
        }

        for (size_t i = 0; i < actual_record_count; i++) {
            rocprofiler_counter_id_t cid = {.handle = 0};
            rocprofiler_query_record_counter_id(prof_records[i].id, &cid);
            auto it = counter_id_to_name.find(cid.handle);
            if (it != counter_id_to_name.end()) {
                aggregated[it->second] += prof_records[i].counter_value;
            }
        }

        for (auto& name : hwCounters) {
            if (!name.empty()) {
                output << (uint64_t)aggregated[name] << ",";
            }
        }
    } else
#endif
    {
        for (size_t i = 0; i < hwCounters.size(); i++) {
            if (!hwCounters[i].empty()) output << "0,";
        }
    }

    output << timeStamp1 << "\n";
    profItr += 1;
}

// ---------------------------------------------------------------------------
// History update
// ---------------------------------------------------------------------------
void updateHist() {
    prevEnergy = currEnergy;
    prevTimeStamp = timeStamp1;
}