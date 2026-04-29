# GGUF Loader — Parked 2026-04-29

## Status

Parked. NOT in scope for v2.0-alpha.

## Why parked

Path X (prebuilt wheel) is dead-end on this hardware:

- abetlen's index publishes Windows cp312 wheels for `llama-cpp-python` only up to **0.3.4** (cu118 / cu121 / cu122 / cu123 / cu124 indexes, all carry the same three wheels: 0.2.88, 0.2.90, 0.3.4)
- cu125 / cu126 wheel indexes do not exist
- 0.3.4 was built around late 2024 against CUDA 12.4 — links against `cudart64_12.dll` / `cublas64_12.dll`
- Host runtime here is CUDA 13.0 (torch 2.10.0+cu130 ships only the 13.x runtime DLLs)
- `llama.dll` therefore fails to load: `FileNotFoundError: Could not find module ... llama.dll (or one of its dependencies)`
- Even if the runtime DLLs were bridged via `nvidia-cuda-runtime-cu12==12.6.*`, the compiled `ggml-cuda.dll` only knows about archs up to sm_90; PTX-JIT to sm_120 is the driver's job and is theoretically possible but unverified for this exact wheel
- The only deterministic path is Path Y (full source build with `CMAKE_CUDA_ARCHITECTURES=120`), which requires the Windows build toolchain (CUDA 13 Toolkit + VS Build Tools 2022 with C++ workload + CMake) — none currently installed

## Unblock conditions

Reopen this ticket when **either** is true:

1. abetlen publishes a Windows cp312 wheel for `llama-cpp-python >= 0.3.5` against cu126 or cu130 with sm_120 PTX baked in. Verifiable by:
   ```
   curl https://abetlen.github.io/llama-cpp-python/whl/cu126/llama-cpp-python/
   curl https://abetlen.github.io/llama-cpp-python/whl/cu130/llama-cpp-python/
   ```
   and grepping for `cp312-cp312-win_amd64.whl` entries past 0.3.4.

2. Jeffrey schedules a free 90-min window and proceeds with Path Y:
   - Install CUDA 13.0 Toolkit (`https://developer.nvidia.com/cuda-13-0-0-download-archive`, ~3 GB)
   - Install VS Build Tools 2022 with "Desktop development with C++" workload (`https://visualstudio.microsoft.com/visual-cpp-build-tools/`, ~3 GB)
   - Install CMake 3.30+ (`https://cmake.org/download/`, ~50 MB)
   - Verify: `nvcc --version`, `cmake --version`, `where cl` all return cleanly
   - Run the source build per `OTR_GGUF_GO_FORWARD_PLAN.md` v1.1 phase 1.3 with venv interpreter:
     ```
     set CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120
     C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pip install llama-cpp-python --no-binary llama-cpp-python --force-reinstall --verbose
     ```

## State at park

- Branch `feat/gguf-creative-loader` was NOT created (env check failed before branching)
- Working tree on `v2.0-alpha` HEAD `fa83ee2` (Remove Gemma 2 family + log BUG-110)
- `llama-cpp-python` and `diskcache` uninstalled from venv
- `scripts/_env_check_gguf.bat`, `scripts/_gguf_install_step1.bat`, `scripts/_probe_wheel_indexes.bat`, `scripts/_bug_log_move.py` left in place as scratch artifacts (gitignore candidates or manual cleanup later)
- `OTR_GGUF_GO_FORWARD_PLAN.md` v1.1 retained as authoritative reference for the unblock path

## What we did NOT do

- Build llama-cpp-python from source (no toolchain)
- Bridge with `nvidia-cuda-runtime-cu12` (declined: even if it loads, sm_120 fallback path is unverified and the test gate is messy)
- Add any `OTR_GGUFLoader` node code (would have shipped dead code)
- Touch the dropdown (no GGUF entries to add)

## Forward-looking

When unblocked, the Phase 2/3/4 work in `OTR_GGUF_GO_FORWARD_PLAN.md` v1.1 stands as written. The OTR-side integration (loader node, dropdown tag `(CREATIVE/GGUF)`, routing in `story_orchestrator.py` to bypass transformers when the tag is detected) is independent of which install path eventually succeeds.

Today's prompt-engineering work on Mistral-Nemo / Gemma 4 (separate ticket #69) is independent of GGUF and proceeds.
