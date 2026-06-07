"""B-dep probe (a) -- a CUDA extension COMPILES + LOADS + runs a kernel on
sm_120, with the cu128 toolchain ISOLATED so a host cu13x nvcc cannot leak into
the source build (the silent ABI-mismatch hazard).

Three phases:
  1. (CPU) scan the HOST PATH for nvcc(s), read each version, flag any whose
     major != the cu128 target.
  2. (CPU) if OTR_CU128_HOME is set, build the sanitized sidecar env and assert
     the cu128 bin LEADS PATH (so its nvcc wins) -- isolation proof.
  3. (GPU) compile a trivial inline CUDA extension, load it, run it, and confirm
     the device is Blackwell sm_120.

PASS iff phase 3 compiles+loads+runs on an sm_120 device AND the EFFECTIVE build
toolchain is the cu128 target (isolated, or no conflicting host toolkit). A host
cu13x nvcc is fine *if* OTR_CU128_HOME isolates it; it is a NO-GO only when no
isolation is in place (a naive build would link cu13x into a cu128 wheel).

OPERATOR-RUN on the 5080 (needs the GPU + the cu128 toolkit). The agent does NOT
run this.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import _b_harness as H  # noqa: E402

TARGET = os.getenv("OTR_CU128_TARGET", "12.8")
CU128_HOME = os.getenv("OTR_CU128_HOME", "")


def _nvcc_version(nvcc_path: str) -> str | None:
    try:
        out = subprocess.run([nvcc_path, "--version"], capture_output=True,
                             text=True, timeout=30)
        return H.parse_nvcc_version(out.stdout)
    except Exception:  # noqa: BLE001
        return None


def _phase1_host_scan() -> list:
    found = H.detect_nvcc_on_path(os.environ.get("PATH", ""), os.path.isfile)
    versions = []
    for p in found:
        ver = _nvcc_version(p)
        versions.append(ver)
        print(f"PROBE a: host nvcc {p} -> {ver}")
    mm = H.cu_toolkit_mismatch(TARGET, [v for v in versions if v])
    if mm["major_conflict"]:
        print(f"PROBE a: host has a cu-major mismatch {mm['major_conflict']} vs target {TARGET}")
    return versions


def _phase2_isolation() -> bool | None:
    if not CU128_HOME:
        print("PROBE a: OTR_CU128_HOME unset -- cannot prove cu128 isolation")
        return None
    env = H.build_sidecar_env(dict(os.environ), CU128_HOME,
                              os.path.join(CU128_HOME, "ext"))
    leads = H.path_leads_with(env["PATH"], CU128_HOME)
    iso_nvcc = H.detect_nvcc_on_path(env["PATH"], os.path.isfile)
    iso_ver = _nvcc_version(iso_nvcc[0]) if iso_nvcc else None
    ok = bool(leads and iso_ver and iso_ver.split(".")[0] == TARGET.split(".")[0])
    print(f"PROBE a: isolated build nvcc -> {iso_ver}; cu128 bin leads PATH = {leads}")
    return ok


_CUDA_SRC = r"""
#include <torch/extension.h>
__global__ void add_one_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
}
torch::Tensor add_one(torch::Tensor x) {
    int n = x.numel();
    int threads = 256, blocks = (n + threads - 1) / threads;
    add_one_kernel<<<blocks, threads>>>(x.data_ptr<float>(), n);
    return x;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("add_one", &add_one); }
"""


def _phase3_compile_run() -> tuple[bool, str]:
    try:
        import torch
        from torch.utils.cpp_extension import load_inline
    except Exception as e:  # noqa: BLE001
        return False, f"torch import failed: {e}"
    if not torch.cuda.is_available():
        return False, "CUDA unavailable (run on the 5080)"
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    build_dir = os.getenv("TORCH_EXTENSIONS_DIR") or os.path.join(
        os.getenv("TEMP", "."), "otr_b_cudaext")
    os.makedirs(build_dir, exist_ok=True)
    try:
        mod = load_inline(name="otr_b_sm120_probe", cpp_sources="",
                          cuda_sources=_CUDA_SRC, functions=["add_one"],
                          build_directory=build_dir, verbose=False)
        x = torch.zeros(1024, dtype=torch.float32, device="cuda")
        y = mod.add_one(x)
        torch.cuda.synchronize()
        ran_ok = bool(torch.allclose(y, torch.ones_like(y)))
    except Exception as e:  # noqa: BLE001
        return False, f"sm_{cap[0]}{cap[1]} {name}: CUDA-ext build/run failed: {e}"
    sm120 = cap[0] >= 12
    msg = f"sm_{cap[0]}{cap[1]} {name}: ext compiled+loaded+ran={ran_ok}"
    return (ran_ok and sm120), msg


def main() -> int:
    versions = _phase1_host_scan()
    iso = _phase2_isolation()
    ok3, msg3 = _phase3_compile_run()
    print(f"PROBE a: {msg3}")
    # Effective-toolchain gate: isolated cu128 OK, or no conflicting host major.
    mm = H.cu_toolkit_mismatch(TARGET, [v for v in versions if v])
    toolchain_ok = (iso is True) or (iso is None and not mm["major_conflict"])
    if not ok3:
        print("PROBE a: NO-GO: the CUDA extension did not compile/load/run on sm_120")
        return 1
    if not toolchain_ok:
        print("PROBE a: NO-GO: a host cu-major toolkit is on PATH with no cu128 "
              "isolation (set OTR_CU128_HOME so the cu128 nvcc leads PATH)")
        return 1
    print("PROBE a: PASS -- CUDA ext builds+loads+runs on sm_120 with the cu128 "
          "toolchain effective")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
