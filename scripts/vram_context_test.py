"""vram_context_test.py -- Measure VRAM peak vs context length per model.

Loads each LLM in the OTR dropdown, runs a single inference at
progressively longer prompt lengths, and records the peak CUDA memory
allocation. Output is a markdown table appended to
docs/2026-04-29-vram-context-test.md so future context_cap raises can
be informed by hard data instead of guesswork.

PRECONDITIONS
- ComfyUI not running (this script will eat all VRAM during model load)
- HF_HUB_CACHE / HF_HOME pointing at C:\\ComfyUI-Models\\huggingface
- venv: C:\\Users\\jeffr\\Documents\\ComfyUI\\.venv\\Scripts\\python.exe

USAGE (Windows venv)
    cd /d C:\\Users\\jeffr\\Documents\\ComfyUI\\custom_nodes\\ComfyUI-OldTimeRadio
    C:\\Users\\jeffr\\Documents\\ComfyUI\\.venv\\Scripts\\python.exe scripts\\vram_context_test.py

OUTPUT
    Appends a markdown table to docs/2026-04-29-vram-context-test.md and
    also prints to stdout.

WARNING: each model loads into 4-bit NF4 for measurement. Total runtime
~10-30 min depending on which models you have cached. Skip models you
don't have cached locally; the script will note them as SKIP rather
than downloading.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Models to test. Conservative list -- only those Jeffrey has actually
# cached on his machine as of 2026-04-29.
MODELS = [
    "mistralai/Mistral-Nemo-Instruct-2407",
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    # 2026-08-25: the three commented-out rows here (Qwen2.5-14B,
    # Captain-Eris, MN-12B-Mag-Mell) were deleted rather than left as
    # "uncomment to include" -- all three are gone from the catalog, so
    # uncommenting one would only produce an unresolvable repo_id.
]

# Prompt lengths to probe (in tokens). Each step doubles roughly so
# we get power-of-two coverage from "tight" to "stress".
PROMPT_LENGTHS = [2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768]

# Build a long Lorem-style filler that we can slice to any token count.
# Uses a real-ish paragraph so the tokenizer doesn't degenerate.
FILLER_PARAGRAPH = """In the static-locked decade after the broadcast collapse, every \
working radio receiver was a sealed container of hope and grief in equal \
measure, transmitting only the sound of weather across thirty-seven dead \
frequencies, although the operators at Listening Post 4 occasionally \
swore they could hear the faint syncopated rhythm of a Morse fragment \
underneath the carrier wave, three short and one long, repeating, never \
quite resolving into anything they could prove had been deliberately \
sent rather than gathered up out of solar interference and pareidolia. \
"""


def _nvml_device_used_bytes(device_index: int = 0) -> int:
    """Total VRAM used on this GPU device by ALL processes, via NVML.

    This is the number you'd see in `nvidia-smi` and includes:
      - PyTorch's caching allocator (this process)
      - CUDA driver overhead (kernel images, contexts, cuBLAS/cuDNN)
      - Any OTHER process using the GPU (browser GPU accel, etc.)

    This is the TRUTH of VRAM consumption on the card. Compare against
    torch.cuda.max_memory_allocated() to see how much is "ours" vs.
    "driver/other." Returns 0 if NVML / pynvml unavailable.

    NOTE this measures ONLY GPU memory (VRAM). It is COMPLETELY
    SEPARATE from CPU/system RAM.
    """
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        return int(info.used)
    except Exception:
        return 0


def _process_cpu_ram_bytes() -> int:
    """Resident set size (CPU RAM) of THIS process in bytes.

    Reported separately from VRAM numbers so the host-side cost of
    loading + quantizing a model is visible but NEVER confused with
    GPU VRAM. The two are different memory spaces. Returns 0 if
    psutil is unavailable.
    """
    try:
        import psutil  # type: ignore
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return 0


def _measure_one(model_id: str, prompt_token_lengths: list[int]) -> list[dict]:
    """Load model in 4-bit NF4, measure VRAM (PyTorch + NVML) and CPU RAM
    at each prompt length.

    The script reports THREE separate numbers per measurement:

      - peak_torch_gb: torch.cuda.max_memory_allocated() in GB.
        This is GPU VRAM allocated by PyTorch's caching allocator
        for THIS process. Does NOT include CUDA driver overhead,
        cuBLAS/cuDNN workspace, or other processes on the GPU.

      - nvml_used_gb: nvidia-smi-equivalent total VRAM used on the
        device by ALL processes. This is the TRUTH of GPU memory
        pressure. Always >= peak_torch_gb.

      - cpu_ram_gb: this process's CPU/system RAM resident set size.
        SEPARATE from VRAM. Reported for visibility (large quantized
        loads can spike CPU RAM during the 4-bit conversion) but
        NEVER mixed into the VRAM decision.

    Cap-tuning decisions in docs/2026-04-29-vram-context-test.md
    should use the nvml_used_gb column as the authoritative VRAM
    metric. peak_torch_gb is informative but undercounts. cpu_ram_gb
    is for host-side capacity planning and never bleeds into VRAM
    accounting.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\n=== {model_id} ===")
    results = []

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"  Loading tokenizer + model (NF4)...")
    t0 = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map={"": "cuda:0"},
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        print(f"  SKIP load failed: {type(exc).__name__}: {exc}")
        return [{"model_id": model_id, "skip": str(exc)}]

    load_s = time.time() - t0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_torch_gb = torch.cuda.memory_allocated() / 1e9
    base_nvml_gb = _nvml_device_used_bytes() / 1e9
    base_cpu_gb = _process_cpu_ram_bytes() / 1e9
    print(
        f"  Loaded in {load_s:.1f}s. "
        f"base_torch={base_torch_gb:.2f}GB  "
        f"base_nvml={base_nvml_gb:.2f}GB  "
        f"base_cpu={base_cpu_gb:.2f}GB"
    )

    for n_target in prompt_token_lengths:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Build a prompt of approximately n_target tokens
            n_paras = max(1, n_target // 80)
            text = (FILLER_PARAGRAPH * n_paras)
            ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=n_target).input_ids.to("cuda:0")
            actual_len = ids.shape[1]

            # Run a single 16-token generation
            t1 = time.time()
            with torch.no_grad():
                _ = mdl.generate(ids, max_new_tokens=16, do_sample=False)
            gen_s = time.time() - t1

            # Three SEPARATE measurements -- never mixed:
            # peak_torch  = PyTorch's allocator, this process, GPU only
            # nvml_used   = nvidia-smi equivalent, all processes, GPU only
            # cpu_ram     = this process's host RAM, NOT GPU
            peak_torch_gb = torch.cuda.max_memory_allocated() / 1e9
            nvml_used_gb = _nvml_device_used_bytes() / 1e9
            cpu_ram_gb = _process_cpu_ram_bytes() / 1e9

            print(
                f"  ctx={actual_len:>6}t  "
                f"torch={peak_torch_gb:.2f}GB  "
                f"nvml={nvml_used_gb:.2f}GB  "
                f"cpu={cpu_ram_gb:.2f}GB  "
                f"gen_s={gen_s:.2f}"
            )
            results.append({
                "model_id":      model_id,
                "ctx_target":    n_target,
                "ctx_actual":    actual_len,
                "peak_torch_gb": round(peak_torch_gb, 3),
                "nvml_used_gb":  round(nvml_used_gb, 3),
                "cpu_ram_gb":    round(cpu_ram_gb, 3),
                "base_torch_gb": round(base_torch_gb, 3),
                "gen_s":         round(gen_s, 2),
            })
        except Exception as exc:
            print(f"  ERROR at ctx={n_target}: {type(exc).__name__}: {exc}")
            results.append({
                "model_id": model_id,
                "ctx_target": n_target,
                "error": str(exc),
            })
            break  # if one length OOMs, larger will too -- stop probing

    # Tear down
    del mdl, tok
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return results


def main() -> int:
    out_doc = Path(__file__).resolve().parents[1] / "docs" / "2026-04-29-vram-context-test.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    all_results: list[dict] = []
    for m in MODELS:
        try:
            all_results.extend(_measure_one(m, PROMPT_LENGTHS))
        except KeyboardInterrupt:
            print("\n[INTERRUPTED]")
            break
        except Exception as exc:
            print(f"  TOP-LEVEL ERROR for {m}: {type(exc).__name__}: {exc}")
            all_results.append({"model_id": m, "skip": str(exc)})

    # Render markdown table.
    # Three SEPARATE columns for VRAM (PyTorch + NVML) and a fourth for
    # CPU RAM. NVML is the cap-tuning truth; PyTorch is informative;
    # CPU is for visibility only and explicitly NEVER mixed with VRAM.
    print("\n\n=== RESULTS ===")
    out_doc.parent.mkdir(parents=True, exist_ok=True)
    with open(out_doc, "a", encoding="utf-8") as f:
        f.write(f"\n\n## Run {timestamp}\n\n")
        f.write("All three memory columns are reported separately:\n\n")
        f.write("- **VRAM nvml** -- nvidia-smi-equivalent device-wide GPU RAM (truth, includes driver overhead + other processes). USE THIS FOR CAP DECISIONS.\n")
        f.write("- **VRAM torch** -- PyTorch caching allocator only, this process. Informative; undercounts driver overhead.\n")
        f.write("- **CPU RAM** -- this process's resident-set size (host RAM). SEPARATE memory space from VRAM. Reported for visibility only.\n\n")
        f.write("| Model | Context (target / actual) | VRAM nvml (GB) | VRAM torch (GB) | CPU RAM (GB) | Gen 16t (sec) | Note |\n")
        f.write("|---|---:|---:|---:|---:|---:|---|\n")
        for r in all_results:
            mid = r.get("model_id", "?").split("/")[-1]
            if "skip" in r:
                line = f"| {mid} | -- | -- | -- | -- | -- | SKIP: {r['skip'][:40]} |"
            elif "error" in r:
                line = (
                    f"| {mid} | {r['ctx_target']} / -- | OOM | OOM | -- | -- | "
                    f"OOM: {r['error'][:60]} |"
                )
            else:
                line = (
                    f"| {mid} | {r['ctx_target']} / {r['ctx_actual']} | "
                    f"{r['nvml_used_gb']:.2f} | "
                    f"{r['peak_torch_gb']:.2f} | "
                    f"{r['cpu_ram_gb']:.2f} | "
                    f"{r['gen_s']:.2f} | |"
                )
            print(line)
            f.write(line + "\n")
    print(f"\nResults appended to: {out_doc}")
    print("REMEMBER: VRAM nvml is the cap-tuning truth.")
    print("VRAM torch is informative.  CPU RAM is host-side and SEPARATE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
