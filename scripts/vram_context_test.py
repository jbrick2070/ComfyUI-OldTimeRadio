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
    # Qwen / EXPERIMENTAL skipped by default to keep runtime tractable;
    # uncomment to include in a longer measurement run.
    # "Qwen/Qwen2.5-14B-Instruct",
    # "Nitral-AI/Captain-Eris_Violet-V0.420-12B",
    # "inflatebot/MN-12B-Mag-Mell-R1",
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


def _measure_one(model_id: str, prompt_token_lengths: list[int]) -> list[dict]:
    """Load model in 4-bit NF4, measure peak VRAM at each prompt length."""
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
    base_alloc_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  Loaded in {load_s:.1f}s, base allocated: {base_alloc_gb:.2f} GB")

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

            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"  ctx={actual_len:>6}t  peak={peak_gb:.2f}GB  gen_s={gen_s:.2f}")
            results.append({
                "model_id": model_id,
                "ctx_target": n_target,
                "ctx_actual": actual_len,
                "peak_gb": round(peak_gb, 3),
                "base_gb": round(base_alloc_gb, 3),
                "gen_s": round(gen_s, 2),
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

    # Render markdown table
    print("\n\n=== RESULTS ===")
    out_doc.parent.mkdir(parents=True, exist_ok=True)
    with open(out_doc, "a", encoding="utf-8") as f:
        f.write(f"\n\n## Run {timestamp}\n\n")
        f.write("| Model | Context (target / actual) | Peak VRAM (GB) | Gen 16t (sec) | Note |\n")
        f.write("|---|---:|---:|---:|---|\n")
        for r in all_results:
            mid = r.get("model_id", "?").split("/")[-1]
            if "skip" in r:
                line = f"| {mid} | -- | -- | -- | SKIP: {r['skip'][:40]} |"
            elif "error" in r:
                line = (
                    f"| {mid} | {r['ctx_target']} / -- | OOM | -- | "
                    f"OOM: {r['error'][:40]} |"
                )
            else:
                line = (
                    f"| {mid} | {r['ctx_target']} / {r['ctx_actual']} | "
                    f"{r['peak_gb']:.2f} (base {r['base_gb']:.2f}) | "
                    f"{r['gen_s']:.2f} | |"
                )
            print(line)
            f.write(line + "\n")
    print(f"\nResults appended to: {out_doc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
