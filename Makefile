.DEFAULT_GOAL := gpuprof
gpuprof: amd-profiling.cpp
	g++ -g -o gpuprof amd-profiling.cpp -I/opt/rocm-6.4.1/include/ -I./include/ -L/opt/rocm-6.4.1/lib/ -lrocm_smi64 -lrocprofiler64v2 -lpthread -lamdhip64 -D__HIP_PLATFORM_AMD__

clean:
	rm -f gpuprof

gpuprof_in_container: amd-profiling.cpp
	g++ -g -o gpuprof amd-profiling.cpp -I/opt/rocm/include/ -I./include/ -L/opt/rocm/lib/ -lrocm_smi64 -lrocprofiler64v2 -lpthread -lamdhip64 -D__HIP_PLATFORM_AMD__