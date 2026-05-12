.DEFAULT_GOAL := gpuprof
gpuprof: amd-profiling.cpp
	hipcc -g -o gpuprof amd-profiling.cpp -I/opt/rocm-7.2.0/include/ -I./include/ -L/opt/rocm-7.2.0/lib/ -lamd_smi -lrocprofiler-sdk -lpthread -lamdhip64 -D__HIP_PLATFORM_AMD__

clean:
	rm -f gpuprof

gpuprof_in_container: amd-profiling.cpp
	hipcc -g -o gpuprof amd-profiling.cpp -I/opt/rocm/include/ -I./include/ -L/opt/rocm/lib/ -lamd_smi -lrocprofiler-sdk -lpthread -lamdhip64 -D__HIP_PLATFORM_AMD__