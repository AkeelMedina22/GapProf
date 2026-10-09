# GapProf

GapProf is a low-overhead, phase-aware GPU profiler for LLM inference. It pairs asynchronous CUPTI activity records with a continuously polled NVML power signal, attributes device execution to the prefill and decode phases under CUDA graph capture, and does so without serializing the workload or recompiling it. It was built to measure the cost of KV cache offloading.

This work is a part of my MSc dissertation at the University of Edinburgh, EPCC. Contact me for further information.

## Features

* **Process Injection.** Attaches to an unmodified CUDA binary via `LD_PRELOAD` and `CUDA_INJECTION64_PATH`.
* **Lock-free telemetry.** A bounded MPSC ring decouples high-frequency event ingestion from the application's critical path. Measured overhead is 2.5% on H200 and 1.7% on GH200.
* **Phase attribution under CUDA graphs.** Host-side NVTX ranges are collapsed into causal decode regions, so device work is attributed to the step that submitted it rather than the window in which the launch returned.
* **Offline analysis.** A Python post-processor aligns the GPU and host clock domains, reconstructs the power curve, and emits an energy report plus a Perfetto trace.

## Layout

```
include/gapprof/   headers
src/               telemetry sources
tests/             unit tests, queue microbenchmarks, NVML probes
submodules/        llama.cpp, Tracy
gapprof            Python wrapper and post-processor 

```

## Prerequisites

* Linux, NVIDIA driver R535 or later (for `NVML_FI_DEV_POWER_INSTANT`)
* CUDA Toolkit including CUPTI and NVML
* CMake 3.17+, a C++20 compiler
* Python 3.11+ with the versions pinned in `requirements.txt`

## Build

```bash
git clone git@github.com:AkeelMedina22/GapProf.git
cd GapProf

cmake -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
pip install -r requirements.txt

git submodule update --init --recursive
cmake -B submodules/llama.cpp/build -S submodules/llama.cpp \
    -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build submodules/llama.cpp/build --config Release -j
```

`ctest` runs the lock-free queue tests twice, once for correctness and once under the thread sanitizer.

Building with `-DUSE_TRACY=ON` instruments GapProf's own code paths for self-profiling. This needs a local Tracy checkout; set the path in `CMakeLists.txt` first.

## Usage

`gapprof` wraps a target binary, sets up injection, and post-processes the resulting trace. Using `llama.cpp`, this run will offload the KV cache to host memory:

```bash
chmod +x gapprof

CUDA_VISIBLE_DEVICES=0 ./gapprof submodules/llama.cpp/build/bin/llama-completion -m /path/to/model/ -f /path/to/prompt/ -ngl 999 -nkvo -fa on -n 200 -c 49152 -no-cnv --no-warmup --ignore-eos
```

This writes three files:

| File | Contents |
| --- | --- |
| `run.csv` | raw event stream (kernels, memcpys, NVTX markers, power samples) |
| `run.rpt` | phase durations, energy, J/token, busy/stall split |
| `run.trace.json` | Perfetto trace, open at https://ui.perfetto.dev |

To analyse a trace collected earlier, pass `--skip-run`.

`test_prof` is a synthetic workload that emits `Phase_Prefill` and `Phase_Decode` NVTX ranges, useful for checking the toolchain before pointing GapProf at a real engine.

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `GAPPROF_POLL_US` | `10000` | NVML polling interval in microseconds |
| `GAPPROF_OUTPUT_CSV` | `gapprof_results.csv` | output path (overridden by `--csv`) |
| `GAPPROF_POWER_SCOPE` | `0` | NVML scope ID; `0` is GPU-only, non-zero selects module scope on Grace |

Polling faster than the sensor updates does not add resolution. The hardware sensor delivers a new value roughly 10 times per second regardless of the polling rate, which is why per-phase power cannot be resolved within a decode step on this hardware. 

## License

MIT. See the `LICENSE` file at the repository root. 