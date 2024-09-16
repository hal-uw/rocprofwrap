# rocprofwrap
Lightweight C++ wrapper using RSMI API Library calls for 1ms profiling of power, temperature, frequency and rocprofiler metrics on AMD GPUs 

## Build
API calls have been tested with rocm v6.0.2. Compatibility with other rocm versions is not guranteed. 
```
git clone https://github.com/hal-uw/rocprofwrap.git 
module load rocm/6.0.2
make
```

## Export Relevant Paths

```
export WRAPPER_ROOT=<directory to gpuprof>
export HSA_TOOLS_LIB=/opt/rocm-6.0.2/lib/librocprofiler64.so.1
export LD_LIBRARY_PATH=/opt/rocm-6.0.2/lib/:$LD_LIBRARY_PATH
```

Run command
```
python rocprofwrap.py --cmd "/work1/sinclair/crhowarth/CoralGemm/build/gemm R_64F R_64F R_64F R_64F OP_N OP_T 8640 8640 8640 8640 8640 8640 36 10” --gpus 0 --prefix “metrics.csv”
```




