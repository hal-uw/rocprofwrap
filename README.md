# rocprofwrap

Lightweight C++ GPU telemetry tool for 1ms-resolution profiling of power, clock frequency, GPU utilization, and hardware counters on AMD GPUs.

Migrated to **ROCm 7.2.0** with `rocprofiler-sdk` (v3) and `amd-smi` APIs.

## Architecture

The `gpuprof` binary collects all data in a single polling loop (~1ms resolution):

- **Telemetry** (power, energy, GFX clock, GPU busy %) — via `amd-smi` handle-mode API
- **Hardware counters** (e.g. `SQ_BUSY_CYCLES`) — via `rocprofiler-sdk` device counting service

On nodes with `kernel.perf_event_paranoid=2`, hardware counter values may be degraded or return 0 due to permission restrictions. Telemetry data (power, clock, utilization) is unaffected.


## Build

Tested with **ROCm 7.2.0**. Compatibility with other ROCm versions is not guaranteed.

```bash
git clone https://github.com/hal-uw/rocprofwrap.git
cd rocprofwrap

# Ensure ROCm 7.2.0 is available
export LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib/:$LD_LIBRARY_PATH
make
```

## Usage

```bash
python rocprofwrap.py --cmd "<gpu_command>" --gpus 0 --prefix metrics [--counters_file counters.txt]
```

### Standalone `gpuprof` Binary

The `gpuprof` binary can also be run directly for telemetry-only collection:

```bash
# Basic usage: output_file  device_id  [config.json]
./gpuprof telemetry.csv 0
```

The tool runs a ~1ms polling loop, writing one CSV row per sample until the process is killed (`Ctrl+C` or `SIGINT`).

### Configuration

An optional JSON config file can specify which hardware counters to collect:

```json
{
    "counters": ["SQ_BUSY_CYCLES"]
}
```

Pass it as the third argument: `./gpuprof output.csv 0 config.json`

## CSV Output Format

| Column | Source | Unit | Description |
|---|---|---|---|
| `socket_power` | `amdsmi_get_power_info` | W | Total socket power |
| `curr_socket_power` | `amdsmi_get_power_info` | W | Current socket power (MI300+ series) |
| `power_from_e` | `amdsmi_get_energy_count` | W | Power derived from energy counter delta |
| `gfx_clk` | `amdsmi_get_clock_info` | MHz | Current GFX engine clock frequency |
| `gpu_busy` | `amdsmi_get_gpu_busy_percent` | % | GPU activity (0–100) |
| `SQ_BUSY_CYCLES` | `rocprofiler-sdk` | cycles | Hardware counter (0 when permissions restricted) |
| `timestamp` | `std::chrono` | ns | Nanosecond timestamp (epoch) |

Example output:
```csv
socket_power,curr_socket_power,power_from_e,gfx_clk,gpu_busy,SQ_BUSY_CYCLES,timestamp
267,267,294.836,180,1,0,1778619558530016271
999,999,1905.81,2100,100,0,1778619234654231637
```

## Known Limitations

- **`perf_event_paranoid=2`**: On nodes with this kernel setting, SDK-based PMC counters may return 0 or be inaccurate.
- **`power_from_e`**: Energy-derived power shows 0 on the first sample (no previous energy reading to diff against).
