#include <cstdio>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <signal.h>
#include <thread>

#include "amd-profiling.hpp"

int main(int argc, char * argv[]){
    
    // Checking for the correct number of command line parameters.
    if(argc != 3){
        std::cout << "You need to supply a filename and GPU device ID." << std::endl;
        std::cout << "Example: ./gpuprof output.txt 0" << std::endl;
        return -1;
    }
   
    signal(SIGINT, signal_callback_handler);
    
    // GPU device to collect metrics.
    device = atoi(argv[2]);

    // Open file using string from command line
    output.open(argv[1], std::ios::out);
    
    //parseYaml("configs/example.yaml");

    hwCounterInit();
    header(output);
  
    // Initialization of rsmi and rocprof
    hipGetDeviceProperties(&devProp, device);
    rocprofiler_initialize();
    
    
    rocprofiler_device_profiling_session_create(&counters[0], counters.size(), &dp_session_id, device, 0);
    rocprofiler_device_profiling_session_start(dp_session_id);


    rsmi_init(0);

    startTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    // Profiling loop
    while(true) {
        
        timeStamp1 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        getData();                
        writeData(output);        
        updateHist();
        
        timeStamp2 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000000-(timeStamp2-timeStamp1)));

        //duration = (int)(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - timeStamp1);
        //printf("Loop Cost:%d,%d\n", duration, (timeStamp2-timeStamp1));
        
        profItr++;
        
        if (stop) break;
    }
   
    rocprofiler_device_profiling_session_stop(dp_session_id);
    rocprofiler_device_profiling_session_destroy(dp_session_id);
    output.close();
    rsmi_shut_down();
    printf("gpuProf: Profiling process %d recieved a signal to stop profiling GPU %d.\n", (int) getpid(), device);
    return 0;
}


