.DEFAULT_GOAL := gpuprof

# Override for a non-default ROCm, e.g. on Frontier: make ROCM=$ROCM_PATH gpuprof_telemetry
ROCM ?= /opt/rocm-7.2.0

gpuprof: amd-profiling.cpp
	hipcc -g -o gpuprof amd-profiling.cpp -I/opt/rocm-7.2.0/include/ -I./include/ -L/opt/rocm-7.2.0/lib/ -lamd_smi -lrocprofiler-sdk -lpthread -lamdhip64 -D__HIP_PLATFORM_AMD__

clean:
	rm -f gpuprof gpuprof_telemetry

gpuprof_in_container: amd-profiling.cpp
	hipcc -g -o gpuprof amd-profiling.cpp -I/opt/rocm/include/ -I./include/ -L/opt/rocm/lib/ -lamd_smi -lrocprofiler-sdk -lpthread -lamdhip64 -D__HIP_PLATFORM_AMD__

# Telemetry-only build: amd-smi power/clock/busy/temp, no rocprofiler-sdk.
# For ROCm versions older than 7.x, whose counter API differs (rocprofiler_counter_config_id_t
# and friends do not exist before 7.0). CSV schema is unchanged — the counter columns are
# still written, zero-filled. Produces ./gpuprof_telemetry, not ./gpuprof.
gpuprof_telemetry: amd-profiling.cpp
	hipcc -g -o gpuprof_telemetry amd-profiling.cpp -I$(ROCM)/include/ -I./include/ -L$(ROCM)/lib/ -lamd_smi -lpthread -lamdhip64 -D__HIP_PLATFORM_AMD__ -DROCPROFWRAP_NO_COUNTERS
