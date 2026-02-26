# rocprofwrap_lt [Claude Code Optimized Version]

Lightweight GPU power sampling wrapper for ROCm 7.0 systems with `amd-smi` API.

Runs a target application and records per-GPU power/clock samples in parallel using AMD SMI (`amdsmi`). All GPUs are handled inside a single multi-threaded binary — no per-GPU processes.

## Files

- `amd_smi_query.cpp`: C++ sampler binary
- `run_profiler.py`: Python launcher that starts/stops the sampler with your workload
- `Makefile`: builds `amd_smi_query`

## What It Does

- Launches **one sampler binary** that spawns one sampler thread + one writer thread per selected GPU
- Launches your application command
- Stops all GPU samplers automatically when the application exits
- Writes binary files like `<prefix>-gpu<id>.bin` (optionally converts to CSV)

## Requirements

- ROCm installed (default path in `Makefile`: `/opt/rocm-7.1.0`)
- AMD SMI runtime/library (`libamd_smi`)
- C++ compiler with C++17 support (`g++` or compatible)
- Python 3

## Build

```bash
make
```

Custom ROCm path:

```bash
make ROCM_DIR=/opt/rocm
```

## Usage

```bash
python3 run_profiler.py -d "<device_ids>" -p <prefix> -- <your_command>
```

Example — profile GPUs 0 and 1 while running a training script:

```bash
python3 run_profiler.py -d "0,1" -p run1 -- python3 gemm.py
```

Example — profile all 4 GPUs and convert to CSV when done:

```bash
python3 run_profiler.py -d "0,1,2,3" -p run1 --post-convert-csv -- ./my_app
```

## Wrapper Options (`run_profiler.py`)

| Option               | Default           | Description                                      |
| -------------------- | ----------------- | ------------------------------------------------ |
| `-d`, `--devices`    | required          | Comma-separated GPU device IDs, e.g. `"0,1,2,3"` |
| `-p`, `--prefix`     | required          | Prefix for output filenames                      |
| `--output-dir`       | `.`               | Directory for output files                       |
| `--query`            | `./amd_smi_query` | Path to sampler binary                           |
| `--interval-ms`      | `1`               | Sampling interval in milliseconds                |
| `--ring-capacity`    | `262144`          | SPSC ring buffer size per GPU (records)          |
| `--realtime`         | off               | Request `SCHED_FIFO` scheduling (best effort)    |
| `--mlock`            | off               | Request `mlockall()` (best effort)               |
| `--post-convert-csv` | off               | Convert `.bin` files to `.csv` after the run     |

CPU core affinity is assigned **automatically** — core 0 is reserved for the OS, sampler threads are pinned to cores 1, 2, 3, … in GPU order.

## Output

Binary output (one file per GPU):

```
<output-dir>/<prefix>-gpu0.bin
<output-dir>/<prefix>-gpu1.bin
...
```

Convert a single file manually:

```bash
./amd_smi_query --convert run1-gpu0.bin --csv run1-gpu0.csv
```

CSV columns:

| Column                   | Description                               |
| ------------------------ | ----------------------------------------- |
| `seq`                    | Sample sequence number                    |
| `device_id`              | GPU device index                          |
| `host_mono_ns`           | Host monotonic timestamp (ns)             |
| `smi_ts_ns`              | SMI energy counter timestamp (ns)         |
| `energy_count`           | Raw energy counter value                  |
| `energy_resolution_uj`   | Energy counter resolution (µJ per count)  |
| `current_socket_power_w` | SMI-reported socket power (W)             |
| `inst_power_w`           | Instantaneous power from energy delta (W) |
| `gfx_clock_mhz`          | GFX clock frequency (MHz)                 |
| `flags`                  | Bitmask: see below                        |

### Flags bitmask

| Bit | Meaning                               |
| --- | ------------------------------------- |
| `0` | Energy counter read failed            |
| `1` | Power info read failed                |
| `2` | Clock read failed                     |
| `3` | Bad timing (energy delta unusable)    |
| `4` | Sample dropped (ring buffer was full) |

## How It Works

### Sampler (`amd_smi_query.cpp`)

One **sampler thread** per GPU runs a tight fixed-interval loop using `clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)` to avoid cumulative drift. Each iteration calls three AMD SMI APIs:

- `amdsmi_get_energy_count` — energy counter + timestamp
- `amdsmi_get_power_info` — socket power
- `amdsmi_get_clk_freq` — GFX clock

Instantaneous power is computed from the energy counter delta:

```
inst_power_W = (delta_energy_uJ × 1e-6) / (delta_time_ns × 1e-9)
```

Samples are pushed into a **lock-free SPSC ring buffer** (power-of-two capacity, cache-line aligned head/tail). If the ring is full the sample is dropped and flagged — the sampler thread **never blocks**.

A dedicated **writer thread** per GPU drains the ring in batches of 4096 records and writes binary output. After the run, files are `fsync`'d.

### Wrapper (`run_profiler.py`)

- Starts a single `amd_smi_query` process covering all requested GPUs
- Starts your target application
- Waits for the application to finish, then sends `SIGINT` to the sampler
- Optionally converts all `.bin` files to `.csv`

## Notes

- Only tested on AMD HPCFund ROCm 7.1.0
- `--realtime` and `--mlock` require appropriate OS privileges (e.g. `CAP_SYS_NICE`, `CAP_IPC_LOCK`); failures are non-fatal warnings