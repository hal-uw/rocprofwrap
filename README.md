# rocprofwrap
Lightweight C++ wrapper using RSMI API Library calls for 1ms profiling of power, temperature, frequency and rocprofiler metrics on AMD GPUs 



export MISCOPE_ROOT=<directory to gpuprof>
export HSA_TOOLS_LIB=/opt/rocm/lib/librocprofiler64.so.1
export LD_LIBRARY_PATH=/opt/rocm/lib/:$LD_LIBRARY_PATH
