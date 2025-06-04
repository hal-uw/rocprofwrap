.DEFAULT_GOAL := gpuprof

# 1) When building on the host (to sanity‐check), use the host’s ROCm-6.2.1
gpuprof: amd-profiling.cpp
	g++ -g -o gpuprof amd-profiling.cpp -I/opt/rocm-6.2.1/include -I./include -L/opt/rocm-6.2.1/lib -lrocm_smi64 -lrocprofiler64v2 -lpthread -lamdhip64 -D__HIP_PLATFORM_AMD__

clean:
	rm -f gpuprof

# 2) When building inside the container (gpuprof_in_container), ROCm lives at /opt/rocm
gpuprof_in_container: amd-profiling.cpp
	g++ -g -o gpuprof amd-profiling.cpp -I/opt/rocm/include -I./include -L/opt/rocm/lib -lrocm_smi64 -lrocprofiler64v2 -lpthread -lamdhip64 -lhsa-runtime64 -D__HIP_PLATFORM_AMD__
