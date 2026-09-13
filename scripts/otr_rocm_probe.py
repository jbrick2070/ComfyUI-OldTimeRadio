"""The five-minute AMD probe: what we need to know before anyone renders.

WHY THIS EXISTS. `ROCM_MISSION_IMPOSSIBLE.md` asks a volunteer for six commands,
a full ComfyUI install, several gigabytes of weights and a complete episode.
That is the right ASK eventually and the wrong FIRST ask: it is a long evening
for a stranger, and if it dies at step 1 we learn almost nothing except that it
died. This answers most of the open AMD questions in about five minutes, with NO
models downloaded and NO render, so the expensive ask only goes to someone whose
card has already cleared the cheap one.

WHAT IS ACTUALLY UNKNOWN, and every check below maps to one of these. Measured
2026-09-12: `docs/dropdown_matrix.json` carries an AMD verdict for ZERO of its
61 engines, and the `amd` machine class is `status: draft` with no receipts at
all. That column is not weak, it is empty.

  1. Does a ROCm torch present itself as `cuda`, the way every profile assumes?
  2. Does ComfyUI's own vendor detection say AMD -- `is_amd()` is what this
     pack's device resolution branches on, and it has never run on an AMD card.
  3. Does OUR device code do the right thing there? `device_options()` and
     `resolve_device()` shipped on 2026-09-12 and have only ever run on NVIDIA
     and Apple silicon. If they are wrong on ROCm, every AMD graph is wrong
     before a single weight loads.
  4. Is bitsandbytes importable? Every non-CUDA profile ships
     `quant_policy: none` on the belief that it is not. That belief is a POLICY
     nobody has measured on real ROCm hardware.
  5. Does a plain bf16 matmul actually execute on the card?

WHAT THIS DELIBERATELY DOES NOT DO: download a model, render anything, or judge
whether an engine fits. Those need the full mission.

Run it from a ComfyUI checkout that has this pack in `custom_nodes/`:

    python custom_nodes/ComfyUI-OldTimeRadio/scripts/otr_rocm_probe.py

Then paste the whole output into the issue. Every line is a fact about the
machine; nothing here reports anything about the operator's files or system
beyond the GPU and the library versions.
"""
from __future__ import annotations

import os
import platform
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACK = os.path.dirname(_HERE)
_CUSTOM_NODES = os.path.dirname(_PACK)


def _find_comfy_root():
    """Where `import comfy` can actually work, or None.

    NOT just the pack's grandparent. On a real install the pack often sits
    under a JUNCTION or a symlink, so `<pack>/../..` is a directory that LOOKS
    like a ComfyUI root and has no `comfy/` in it -- which is exactly what
    happened the first time this probe was run, on 2026-09-12. The probe then
    reported the pack's core-absent FALLBACK values as though they were this
    machine's answer: `vendor() = unknown` and `resolve_device('default') = cpu`
    on a box with a working GPU. A volunteer reporting that would have sent us
    chasing a defect that does not exist.

    So: look for a directory that really contains `comfy/model_management.py`,
    honour an explicit override first, and tell the truth when there is none.
    """
    env = os.environ.get("COMFYUI_ROOT") or os.environ.get("OTR_COMFYUI_ROOT")
    candidates = [env] if env else []
    candidates.append(os.path.dirname(_CUSTOM_NODES))
    candidates.append(os.getcwd())
    here = _PACK
    for _ in range(6):                       # walk up, for odd layouts
        here = os.path.dirname(here)
        if not here or here == os.path.dirname(here):
            break
        candidates.append(here)
        candidates.append(os.path.join(here, "ComfyUI"))
    for cand in candidates:
        if cand and os.path.isfile(os.path.join(cand, "comfy", "model_management.py")):
            return os.path.abspath(cand)
    return None


_COMFY = _find_comfy_root()
for _p in (_PACK, _COMFY):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

LINES: list = []


def say(label, value):
    LINES.append("%-34s %s" % (label, value))
    print("%-34s %s" % (label, value), flush=True)


def section(title):
    LINES.append("")
    LINES.append("-- %s " % title + "-" * max(0, 60 - len(title)))
    print("\n-- %s " % title + "-" * max(0, 60 - len(title)), flush=True)


def attempt(label, fn):
    """Run one check. A failure is a RESULT, never the end of the probe."""
    try:
        say(label, fn())
    except Exception as exc:                       # noqa: BLE001 -- reporting
        say(label, "FAILED: %s: %s" % (type(exc).__name__, str(exc)[:120]))


def main() -> int:
    print(__doc__.splitlines()[0])
    section("the machine")
    say("platform", "%s %s" % (platform.system(), platform.release()))
    say("python", sys.version.split()[0])

    section("1. does ROCm torch present itself as cuda")
    try:
        import torch
    except Exception as exc:                       # noqa: BLE001
        say("torch", "NOT IMPORTABLE: %s" % exc)
        print("\nNothing else can be checked without torch. Install the ROCm "
              "build first -- see ROCM_MISSION_IMPOSSIBLE.md step 1.")
        return 2
    say("torch", torch.__version__)
    say("torch.version.hip", getattr(torch.version, "hip", None))
    say("torch.version.cuda", getattr(torch.version, "cuda", None))
    attempt("torch.cuda.is_available()", lambda: torch.cuda.is_available())
    attempt("device count", lambda: torch.cuda.device_count())
    attempt("device name", lambda: torch.cuda.get_device_name(0))
    attempt("total VRAM (GiB)", lambda: round(
        torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 2))

    section("2. what ComfyUI's own detection says")
    say("ComfyUI root found", _COMFY or "*** NONE ***")
    if _COMFY is None:
        print(
            "\n*** STOP. `comfy` is not importable from here, so sections 2 "
            "and 3 below report this pack's CORE-ABSENT FALLBACK, not your "
            "machine. Those numbers are meaningless for AMD and must not be "
            "reported as results. Run the probe from inside your ComfyUI "
            "checkout, or set COMFYUI_ROOT to the directory that contains "
            "`comfy/model_management.py`, and run it again. ***\n",
            flush=True)

    def _mm():
        import comfy.model_management as mm
        return mm

    attempt("comfy import", lambda: _mm().__name__)
    attempt("is_amd()", lambda: _mm().is_amd())
    attempt("is_nvidia()", lambda: _mm().is_nvidia())
    attempt("get_torch_device()", lambda: str(_mm().get_torch_device()))
    attempt("get_gpu_device_options()",
            lambda: list(_mm().get_gpu_device_options()))

    section("3. what THIS pack's device code resolves to")
    say("(these shipped 2026-09-12 and", "have never run on an AMD card)")

    def _devopts():
        # LOADED BY PATH, NOT BY `import nodes...`, and this is not fussiness.
        # ComfyUI's own root contains a module called `nodes.py`, and the root
        # has to be on sys.path for `import comfy` to work at all -- so a plain
        # `from nodes._otr_shared import ...` resolves to ComfyUI's file and
        # dies with "'nodes' is not a package". Measured on 2026-09-12: the
        # probe reported this pack's device code as broken on a machine where
        # it is fine, which is precisely the false alarm a volunteer would have
        # spent an evening on.
        import importlib.util
        name = "_otr_probe_device_options"
        if name in sys.modules:
            return sys.modules[name]
        path = os.path.join(_PACK, "nodes", "_otr_shared", "device_options.py")
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    attempt("device_options()", lambda: _devopts().device_options())
    attempt("vendor()", lambda: _devopts().vendor())
    attempt("resolve_device('default')",
            lambda: _devopts().resolve_device("default"))
    attempt("resolve_device('cuda')", lambda: _devopts().resolve_device("cuda"))
    attempt("resolve_device('tpu') [must pass through untouched]",
            lambda: _devopts().resolve_device("tpu"))

    section("4. is bitsandbytes real here")
    say("(every non-CUDA profile ships", "quant_policy 'none' assuming it is not)")

    def _bnb():
        import bitsandbytes as bnb
        return getattr(bnb, "__version__", "imported, no __version__")

    attempt("bitsandbytes", _bnb)

    def _bnb_backends():
        import bitsandbytes as bnb
        fn = getattr(bnb, "supported_torch_devices", None)
        return sorted(fn()) if callable(fn) else "no supported_torch_devices()"

    attempt("bitsandbytes backends", _bnb_backends)

    section("5. does the card actually compute")

    def _matmul():
        a = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")
        c = (a @ b).float()
        torch.cuda.synchronize()
        return "bf16 512x512 matmul OK, finite=%s" % bool(c.isfinite().all())

    attempt("bf16 matmul on the card", _matmul)

    def _fp16():
        a = torch.randn(512, 512, dtype=torch.float16, device="cuda")
        c = (a @ a).float()
        torch.cuda.synchronize()
        return "fp16 OK, finite=%s" % bool(c.isfinite().all())

    attempt("fp16 matmul on the card", _fp16)

    section("what to do with this")
    print("""
Paste EVERYTHING above into the issue. Three lines decide what happens next:

  * `is_amd()` -- if this is not True, our device resolution will treat the card
    as NVIDIA and every quantisation decision downstream is made on a wrong
    premise. That is the single most important line here.
  * `vendor()` -- must read `amd`. If it reads `nvidia` or `unknown`, the pack
    cannot tell your card apart from a GeForce and we have a real bug to fix
    before you spend an evening on a render.
  * `bitsandbytes` -- if it imports AND lists a usable backend, the AMD profiles
    are leaving performance on the table by shipping `quant_policy: none`, and
    that is a good problem worth knowing about.

If those look sane, the full mission in ROCM_MISSION_IMPOSSIBLE.md is worth your
evening. If they do not, you have saved yourself one and taught us more than a
failed render would have.
""")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                              # noqa: BLE001
        traceback.print_exc()
        print("\nThe probe itself crashed. That is a finding too -- please "
              "paste this traceback into the issue.")
        sys.exit(1)
