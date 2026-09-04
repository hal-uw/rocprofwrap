#include <cstdio>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <signal.h>
#include <thread>
#include <unistd.h>
#include <fcntl.h>
#include"counter_config.hpp"
#include "amd-profiling.hpp"


int main(int argc, char * argv[]){
    
    // Checking for the correct number of command line parameters.
    if(argc < 3){
        std::cout << "You need to supply a filename and GPU device ID." << std::endl;
        std::cout << "Example: ./gpuprof output.txt 0" << std::endl;
        std::cout << "Optional: ./gpuprof output.txt 0 counters.json" << std::endl;
        return -1;
    }

    signal(SIGINT, signal_callback_handler);

    // GPU device to collect metrics.
    device = atoi(argv[2]);

    // Open file using string from command line
    output.open(argv[1], std::ios::out);

    // Initialize the hardware counters from JSON config if provided
    if (argc >= 4) {
        if (!CounterConfig::parseJsonConfig(argv[3], hwCounters)) {
            std::cerr << "Warning: Failed to parse counter configuration file. Using default counters." << std::endl;
        }
        // print out the hwCounters at the console not in the file
        std::cout << "Using the following hardware counters:" << std::endl;
        for (const auto& c : hwCounters) {
            std::cout << c << std::endl;
        }
    }

    // Initialize amd-smi (replaces rsmi_init)
    smiInit();

    // Initialize rocprofiler-sdk via force_configure
    // This triggers rocprofiler_configure() -> tool_init_callback() -> rocprofSetup()
    // Must be called BEFORE any HIP call so HSA can be intercepted
    // Suppress SDK permission warnings (ioctl.cpp) from polluting console.
    // The warnings repeat on every sampling call, so keep stderr redirected
    // for the entire program. All useful output goes to stdout or the CSV file.
    // Telemetry-only builds keep stderr, since there is no SDK to be noisy and
    // amd-smi failures are otherwise invisible.
#ifndef ROCPROFWRAP_NO_COUNTERS
    int devnull = open("/dev/null", O_WRONLY);
    if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }

    rocprofiler_force_configure(nullptr);
#endif

    // Write CSV header
    header(output);

    // HIP device properties (this triggers HSA initialization)
    (void)hipGetDeviceProperties(&devProp, device);

    startTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    // Profiling loop
    while(true) {

        timeStamp1 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        getData();
        writeData(output);
        updateHist();

        timeStamp2 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        std::this_thread::sleep_for(std::chrono::nanoseconds(1000000-(timeStamp2-timeStamp1)));

        profItr++;

        if (stop) break;
    }

    // Cleanup rocprofiler-sdk
#ifndef ROCPROFWRAP_NO_COUNTERS
    if (rocprof_initialized) {
        rocprofiler_stop_context(prof_ctx);
        rocprofiler_flush_buffer(prof_buf);
        rocprofiler_destroy_counter_config(prof_config);
    }
#endif

    output.close();

    // Cleanup amd-smi (replaces rsmi_shut_down)
    amdsmi_shut_down();

    printf("gpuProf: Profiling process %d recieved a signal to stop profiling GPU %d.\n", (int) getpid(), device);
    return 0;
}
