"""vram_context_test.py -- OTR_VRAMContextTest ComfyUI node.

Production-accurate VRAM measurement at varying prompt context lengths,
running INSIDE ComfyUI via the same canonical loader surface
(`_otr_model_loader.request_slot` + `make_generate_fn`) that real
renders use. Replaces the earlier headless standalone script
(`scripts/vram_context_test.py`), which loaded models outside ComfyUI
and undercounted runtime overhead.

Why an in-workflow node:

  - ComfyUI's model-management framework, custom-node imports, and any
    other co-resident state add ~500 MB-1 GB of VRAM the headless
    measurement misses. Cap-tuning decisions need numbers from the
    real path.
  - The node calls `request_slot` and `make_generate_fn` directly, so
    quantization config, attention backend selection, tokenizer setup,
    and prompt-truncation logic exactly match what
    LLMScriptWriter.write_script() does.
  - Results are stamped into the ledger via the early-init / per-clip
    pattern used elsewhere in OTR, so they're inspectable post-run
    via /otr/latest_ledger or any standard ledger tooling.

Workflow placement:

  Drop the OTR_VRAMContextTest node into a small smoke workflow
  (NO LLMScriptWriter / Bark / FLUX / HuMo) so the test runs without
  competing for VRAM. The node loads the chosen LLM once, probes
  every prompt length in sequence, then unloads.

Output:

  - report (STRING): markdown table summarising each probe's VRAM
    + CPU RAM + generation time. Append to docs/2026-04-29-vram-
    context-test.md or pipe to a Save Text node.
  - Side effect: appends entries to ledger.meta.vram_test_results[]
    via the existing early-ledger init + Ledger.save() merge.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

log = logging.getLogger("OTR.vram_context_test")


# Per-prompt-length filler paragraph. Sliced to the target token count so
# every probe sees a similar prompt structure. Same paragraph the headless
# script used so cross-comparison is straightforward.
_FILLER_PARAGRAPH = (
    "In the static-locked decade after the broadcast collapse, every working "
    "radio receiver was a sealed container of hope and grief in equal measure, "
    "transmitting only the sound of weather across thirty-seven dead frequencies, "
    "although the operators at Listening Post 4 occasionally swore they could hear "
    "the faint syncopated rhythm of a Morse fragment underneath the carrier wave, "
    "three short and one long, repeating, never quite resolving into anything they "
    "could prove had been deliberately sent rather than gathered up out of solar "
    "interference and pareidolia. "
)


# Same model-id list the LLMScriptWriter dropdown exposes. Kept in sync
# manually -- if the orchestrator dropdown gains/loses entries, mirror
# them here so the test surface matches the production surface.
# 2026-08-25: re-synced. This list had drifted badly -- it still named
# Captain-Eris and MN-12B-Mag-Mell, pruned from the catalog on 2026-05-23,
# and Qwen2.5-14B, pruned 2026-08-25. Three of its six entries were models
# the catalog no longer offered, which is what "kept in sync manually"
# decays into. The probe surface now matches the real curated local rows.
_LLM_MODEL_CHOICES = [
    "mistralai/Mistral-Nemo-Instruct-2407",
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "google/gemma-4-12b-it",
    "google/gemma-2-2b-it",
]

# Default prompt-length probe set. Comma-separated for the widget so the
# user can edit per-run without recompiling. Powers-of-two-ish coverage
# from "tight" to "stress" with a focus around the production cap range
# (8K-16K) since that's what we're actually trying to tune.
_DEFAULT_PROBE_LENGTHS = "2048,4096,6144,8192,12288,16384,20480,24576"


def _accel_state() -> str:
    """Live accelerator for the probe: ``cuda`` | ``mps`` | ``none``.

    S0 portability (2026-07-10): every ``torch.cuda.*`` counter this node
    used was UNGUARDED, so the diagnostics node crashed on exactly the
    hosts it is most useful for. MPS gets the ``torch.mps`` equivalents
    where they exist; a pure-CPU host still measures CPU RAM + wall time
    with the VRAM columns honestly 0.
    """
    import torch
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "none"


def _accel_empty_cache(accel: str) -> None:
    import torch
    if accel == "cuda":
        torch.cuda.empty_cache()
    elif accel == "mps" and hasattr(getattr(torch, "mps", None), "empty_cache"):
        torch.mps.empty_cache()


def _accel_reset_peak(accel: str) -> None:
    import torch
    if accel == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # MPS has no peak-reset API; _accel_peak_gb reports current allocation.


def _accel_allocated_gb(accel: str) -> float:
    import torch
    if accel == "cuda":
        return torch.cuda.memory_allocated() / 1e9
    if accel == "mps" and hasattr(getattr(torch, "mps", None),
                                  "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / 1e9
    return 0.0


def _accel_peak_gb(accel: str) -> float:
    import torch
    if accel == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if accel == "mps" and hasattr(getattr(torch, "mps", None),
                                  "current_allocated_memory"):
        # No max_memory_allocated on MPS -- current allocation is the
        # honest floor (labelled as such in the report).
        return torch.mps.current_allocated_memory() / 1e9
    return 0.0


def _nvml_used_bytes(device_index: int = 0) -> int:
    """Total VRAM used on this device by ALL processes (nvidia-smi truth)."""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        return int(info.used)
    except Exception:
        return 0


def _process_cpu_ram_bytes() -> int:
    """This process's resident-set size on the host (CPU RAM, NOT VRAM)."""
    try:
        import psutil  # type: ignore
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return 0


def _parse_lengths(s: str) -> list[int]:
    """Parse a comma-separated string of token-length probe targets."""
    out: list[int] = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            n = int(tok)
            if n >= 64:
                out.append(n)
        except ValueError:
            continue
    return out or [int(x) for x in _DEFAULT_PROBE_LENGTHS.split(",")]


class VRAMContextTest:
    """Measure VRAM (PyTorch + NVML) and CPU RAM at each probe prompt length.

    Calls `_otr_model_loader.request_slot` and `make_generate_fn` so
    measurements reflect the same canonical code path LLMScriptWriter
    uses in production renders. Results stamp into
    ledger.meta.vram_test_results[] and a markdown report is returned
    as the node's STRING output.
    """

    CATEGORY = "OTR/v2/Diagnostics"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    OUTPUT_NODE = True
    # S30 B6 opt-in: VRAMContextTest is a DIAGNOSTIC node that
    # measures LLM VRAM consumption at varying prompt lengths. It
    # picks an LLM by widget on purpose (the whole point of the
    # diagnostic is "pick a model + measure it"). It is NOT a
    # production consumer surface; the two-slot LLM rule does not
    # apply. Marker borrowed from the non-LLM media-node exemption
    # since the structural guard rule is the same.
    NON_LLM_MODEL_WIDGET_OK = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": (_LLM_MODEL_CHOICES, {
                    "default": "mistralai/Mistral-Nemo-Instruct-2407",
                    "tooltip": (
                        "Which LLM to probe. Mirrors the LLMScriptWriter "
                        "dropdown so production-relevant models are first-"
                        "class. Suffix tags ([ALPHA], (EXPERIMENTAL)) are "
                        "stripped before the HF lookup -- same behaviour as "
                        "the writer's request_slot path."
                    ),
                }),
                "probe_lengths": ("STRING", {
                    "default": _DEFAULT_PROBE_LENGTHS,
                    "tooltip": (
                        "Comma-separated token-length targets. Each probe "
                        "builds a prompt of approximately this many tokens "
                        "and runs a single 16-token generation, capturing "
                        "VRAM nvml + VRAM torch + CPU RAM peaks. Default "
                        f"probes: {_DEFAULT_PROBE_LENGTHS}"
                    ),
                }),
                "max_new_tokens": ("INT", {
                    "default": 16, "min": 4, "max": 256, "step": 4,
                    "tooltip": (
                        "Generation length per probe. 16 is the smallest "
                        "useful sample for VRAM measurement (KV cache "
                        "growth is the relevant scaling factor, not "
                        "generation length itself). Raise to 64-128 if "
                        "you want to also bench tokens/sec at this "
                        "context size."
                    ),
                }),
            },
            "optional": {
                "optimization_profile": (
                    ["Pro (Ultra Quality)", "Standard", "Obsidian (UNSTABLE/4GB)"],
                    {"default": "Standard",
                     "tooltip": "Same widget as LLMScriptWriter -- "
                                "controls 4-bit NF4 vs full precision."},
                ),
                "measurement_label": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Optional tag for this run, stamped into the "
                        "ledger entry. Useful when running the same node "
                        "across multiple workflows or after VRAM-affecting "
                        "config changes."
                    ),
                }),
            },
        }

    def execute(
        self,
        model_id: str,
        probe_lengths: str,
        max_new_tokens: int = 16,
        optimization_profile: str = "Standard",
        measurement_label: str = "",
    ):
        # Late imports so the node loads cleanly even if the loader
        # module has heavy dependencies missing in some environments.
        # S31 B1: switched off legacy `_SO._load_llm` / `_SO._generate_with_llm`
        # onto the canonical `_otr_model_loader.request_slot` +
        # `make_generate_fn` surface (Hard rule #5).
        try:
            from . import _otr_model_loader as _OTRML
        except ImportError:
            import sys
            from pathlib import Path
            _NODES_DIR = Path(__file__).resolve().parent
            if str(_NODES_DIR) not in sys.path:
                sys.path.insert(0, str(_NODES_DIR))
            import _otr_model_loader as _OTRML  # type: ignore

        try:
            import torch
        except ImportError:
            return ("torch unavailable; cannot run VRAM probe.",)

        lengths = _parse_lengths(probe_lengths)
        log.info(
            "[VRAMContextTest] model=%s probes=%s max_new_tokens=%d label=%s",
            model_id, lengths, max_new_tokens, measurement_label or "<none>",
        )

        # Pre-load the model once via the canonical lifecycle helper.
        # request_slot caches by (slot, model_id); the first call is the
        # "real" load and subsequent generations reuse the cached entry.
        # S31 B1: VRAMContextTest is a diagnostic surface that picks a
        # single LLM by widget for measurement, so it claims the
        # "technical" slot. NON_LLM_MODEL_WIDGET_OK = True keeps this
        # exempt from the two-slot writer rule (see class marker).
        accel = _accel_state()
        log.info("[VRAMContextTest] pre-loading %s (accelerator=%s) ...",
                 model_id, accel)
        _accel_empty_cache(accel)
        _accel_reset_peak(accel)
        load_t0 = time.time()
        try:
            # S1: diagnostics probe the nv50 BASELINE policy explicitly
            # (the same resolved values production defaults to).
            from ._otr_shared.llm_policy import BASELINE_POLICY
            cache_entry = _OTRML.request_slot(
                "technical", model_id, policy=BASELINE_POLICY)
        except Exception as exc:
            msg = (
                f"## VRAM Context Test\n\n"
                f"FAILED to load `{model_id}`: `{type(exc).__name__}: {exc}`\n"
            )
            log.exception("[VRAMContextTest] load failed: %s", exc)
            return (msg,)
        load_s = time.time() - load_t0
        base_torch_gb = _accel_allocated_gb(accel)
        base_nvml_gb = _nvml_used_bytes() / 1e9
        base_cpu_gb = _process_cpu_ram_bytes() / 1e9
        log.info(
            "[VRAMContextTest] loaded in %.1fs  base_torch=%.2fGB  "
            "base_nvml=%.2fGB  base_cpu=%.2fGB",
            load_s, base_torch_gb, base_nvml_gb, base_cpu_gb,
        )

        # Probe each prompt length via the canonical generate surface
        # (Hard rule #5): request_slot + make_generate_fn. Captures three
        # orthogonal memory numbers per probe (see
        # docs/2026-04-29-vram-context-test.md for the column
        # definitions). VRAM nvml is the cap-tuning truth.
        gen_fn = _OTRML.make_generate_fn(cache_entry)
        results: list[dict] = []
        for n_target in lengths:
            _accel_empty_cache(accel)
            _accel_reset_peak(accel)

            n_paras = max(1, n_target // 80)
            prompt = _FILLER_PARAGRAPH * n_paras

            t0 = time.time()
            try:
                _ = gen_fn(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as exc:
                log.exception(
                    "[VRAMContextTest] probe failed at ctx=%d: %s",
                    n_target, exc,
                )
                results.append({
                    "ctx_target": n_target,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                # If one length OOMs, larger will too -- stop probing
                if "OutOfMemoryError" in type(exc).__name__ or "out of memory" in str(exc).lower():
                    break
                else:
                    continue
            gen_s = time.time() - t0

            peak_torch_gb = _accel_peak_gb(accel)
            nvml_gb = _nvml_used_bytes() / 1e9
            cpu_gb = _process_cpu_ram_bytes() / 1e9
            log.info(
                "[VRAMContextTest] ctx=%d  torch=%.2fGB  nvml=%.2fGB  "
                "cpu=%.2fGB  gen_s=%.2f",
                n_target, peak_torch_gb, nvml_gb, cpu_gb, gen_s,
            )
            results.append({
                "ctx_target":    n_target,
                "peak_torch_gb": round(peak_torch_gb, 3),
                "nvml_used_gb":  round(nvml_gb, 3),
                "cpu_ram_gb":    round(cpu_gb, 3),
                "gen_s":         round(gen_s, 2),
            })

        # Stamp ledger.meta.vram_test_results[] for post-run inspection
        # via /otr/latest_ledger or any standard ledger tooling. Best-
        # effort; never blocks the report return.
        try:
            from .production_ledger import get_ledger
            led = get_ledger()
            led.data.setdefault("meta", {}).setdefault("vram_test_results", []).append({
                "label":                measurement_label or None,
                "model_id":             model_id,
                "optimization_profile": optimization_profile,
                "max_new_tokens":       int(max_new_tokens),
                "load_s":               round(load_s, 2),
                "base_torch_gb":        round(base_torch_gb, 3),
                "base_nvml_gb":         round(base_nvml_gb, 3),
                "base_cpu_gb":          round(base_cpu_gb, 3),
                "probes":               results,
            })
            led.save()
        except Exception as ledger_exc:  # noqa: BLE001
            log.warning(
                "[VRAMContextTest] ledger stamp failed (non-fatal): %s",
                ledger_exc,
            )

        # Build markdown report.
        lines = [
            f"## VRAM Context Test -- `{model_id}`",
            "",
            f"- **label**: {measurement_label or '_<none>_'}",
            f"- **profile**: {optimization_profile}",
            f"- **accelerator**: {accel}"
            + ("" if accel == "cuda" else
               " (no CUDA -- VRAM torch column is "
               + ("MPS current-allocation, no peak API" if accel == "mps"
                  else "0; CPU RAM + wall time are the signal")
               + ")"),
            f"- **load**: {load_s:.1f}s "
            f"(base torch {base_torch_gb:.2f} / nvml {base_nvml_gb:.2f} / "
            f"cpu {base_cpu_gb:.2f} GB)",
            "",
            "Three memory columns are reported separately. Use **VRAM nvml** "
            "for cap-tuning decisions -- it includes CUDA driver overhead "
            "and any other GPU process. **VRAM torch** is PyTorch allocator "
            "only (this process). **CPU RAM** is host-side and NEVER mixed "
            "into VRAM accounting.",
            "",
            "| Context (target) | VRAM nvml (GB) | VRAM torch (GB) | "
            "CPU RAM (GB) | Gen sec | Note |",
            "|---:|---:|---:|---:|---:|---|",
        ]
        for r in results:
            if "error" in r:
                lines.append(
                    f"| {r['ctx_target']} | OOM | OOM | -- | -- | "
                    f"{r['error'][:50]} |"
                )
            else:
                lines.append(
                    f"| {r['ctx_target']} | {r['nvml_used_gb']:.2f} | "
                    f"{r['peak_torch_gb']:.2f} | {r['cpu_ram_gb']:.2f} | "
                    f"{r['gen_s']:.2f} | |"
                )

        report = "\n".join(lines) + "\n"

        # Side-effect: persist the report to ComfyUI's standard output
        # directory so it survives the session and can be diffed across
        # runs. Use folder_paths (BUG-01.02 rule -- output nodes must
        # NEVER hardcode paths). Best-effort; failure here doesn't
        # block returning the in-memory report to the workflow.
        try:
            import folder_paths  # type: ignore
            out_root = Path(folder_paths.get_output_directory()) / "otr" / "vram_tests"
            out_root.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            safe_model = model_id.replace("/", "_").replace(":", "_")
            out_path = out_root / f"vram_test_{stamp}_{safe_model}.md"
            out_path.write_text(report, encoding="utf-8")
            log.info("[VRAMContextTest] report written: %s", out_path)
        except Exception as save_exc:  # noqa: BLE001
            log.warning(
                "[VRAMContextTest] report save failed (non-fatal): %s",
                save_exc,
            )

        log.info("[VRAMContextTest] complete (%d probes)", len(results))
        return (report,)


__all__ = ["VRAMContextTest"]
