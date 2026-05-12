#include "amd_smi/amdsmi.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

int main(int argc, char* argv[]) {
    int device_idx = (argc >= 2) ? atoi(argv[1]) : 0;

    amdsmi_init(AMDSMI_INIT_AMD_GPUS);

    uint32_t socket_count = 0;
    amdsmi_get_socket_handles(&socket_count, nullptr);
    std::vector<amdsmi_socket_handle> sockets(socket_count);
    amdsmi_get_socket_handles(&socket_count, sockets.data());

    std::vector<amdsmi_processor_handle> gpus;
    for (uint32_t s = 0; s < socket_count; s++) {
        uint32_t proc_count = 0;
        amdsmi_get_processor_handles(sockets[s], &proc_count, nullptr);
        std::vector<amdsmi_processor_handle> procs(proc_count);
        amdsmi_get_processor_handles(sockets[s], &proc_count, procs.data());
        for (uint32_t p = 0; p < proc_count; p++) {
            processor_type_t ptype;
            if (amdsmi_get_processor_type(procs[p], &ptype) == AMDSMI_STATUS_SUCCESS &&
                ptype == AMDSMI_PROCESSOR_TYPE_AMD_GPU)
                gpus.push_back(procs[p]);
        }
    }

    if ((size_t)device_idx >= gpus.size()) {
        printf("GPU %d not found (%zu GPUs total)\n", device_idx, gpus.size());
        return 1;
    }

    printf("Testing etimeStamp on GPU %d — 20 samples at ~1ms interval\n\n", device_idx);
    printf("%-6s  %-22s  %-22s  %-14s  %-14s  energy\n",
           "iter", "wall_clock_ns", "etimeStamp_raw", "wall_delta_ns", "estamp_delta", "raw_count");
    printf("%-6s  %-22s  %-22s  %-14s  %-14s  %-12s\n",
           "------", "----------------------", "----------------------",
           "--------------", "--------------", "------------");

    uint64_t prev_wall = 0, prev_ets = 0, prev_energy = 0;
    for (int i = 0; i < 20; i++) {
        uint64_t wall = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        uint64_t energy = 0, ets = 0;
        float res = 0;
        amdsmi_get_energy_count(gpus[device_idx], &energy, &res, &ets);

        if (i == 0) {
            printf("%-6d  %-22lu  %-22lu  %-14s  %-14s  %lu\n",
                   i, wall, ets, "(baseline)", "(baseline)", energy);
        } else {
            int64_t wall_delta = (int64_t)(wall - prev_wall);
            int64_t ets_delta  = (int64_t)(ets  - prev_ets);
            printf("%-6d  %-22lu  %-22lu  %-14ld  %-14ld  %lu  (Δenergy=%ld, res=%.1fμJ)\n",
                   i, wall, ets, wall_delta, ets_delta,
                   energy, (int64_t)(energy - prev_energy), res);
        }

        prev_wall = wall; prev_ets = ets; prev_energy = energy;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    amdsmi_shut_down();
    return 0;
}
