"""ONE device vocabulary for this pack, borrowed from ComfyUI rather than invented.

**Operator ruling, 2026-09-12:** *"if comfy supports auto detection great, use
that, not rolling your own custom solution"*, and *"if comfy core has a way to be
sure our dropdowns are filtered by capability great, but I don't want us
hand-rolling our own thing to limit dropdowns -- only if it's tried and true
comfy approved."*

It does, and this module is a thin wrapper over it. Nothing here detects
hardware; core does that.

    comfy.model_management.get_gpu_device_options()        the option list
    comfy.model_management.resolve_gpu_device_option(opt)  option -> torch.device
    comfy.model_management.get_torch_device()              the host default
    comfy.model_management.is_nvidia() / is_amd()          the vendor ROCm hides

**THE PRECEDENT IS IN CORE ITSELF.** `comfy_extras/nodes_multigpu.py` builds its
device combos from exactly these functions (three nodes, at :203, :257 and :309).
This is the sanctioned pattern with in-tree usage, not a private helper.

WHY THIS BEATS WHAT WE HAD. Three widgets each hand-typed
``["cuda", "cpu", "mps"]``, and two more places re-derived the same answer with
their own torch probes. That vocabulary is wrong three ways: it NAMES a vendor,
so whichever machine last saved the canonical stamps its own hardware into the
file everyone else pulls; it cannot tell AMD from NVIDIA, because ROCm reports
itself as cuda; and it silently omits every other backend core supports --
DirectML, Intel XPU, Ascend NPU, Cambricon MLU.

WHY THE LEGACY NAMES SURVIVE, and this is the part that must not be dropped.
ComfyUI validates a combo by MEMBERSHIP (`execution.py`, ``val not in
combo_options``). Core's labels are ``default`` / ``cpu`` / ``gpu:N``; ours were
``cuda`` / ``cpu`` / ``mps``. Shipping core's list ALONE would make every saved
graph holding ``cuda`` or ``mps`` invalid -- our own canonical, all 93 variants,
and any workflow a user saved yesterday. So the shipped list is core's options
PLUS the legacy names, which keep working and resolve to the same devices they
always did. New graphs get ``default``; old graphs load unchanged.

WHAT THIS DOES **NOT** CHANGE, because the principle was right. The tooltips say
an EXPLICIT device fails loud rather than silently downgrading, and a ledger that
records what actually ran beats an auto-fallback nobody can see. That stands:
only ``default`` is ever resolved for you, an explicitly chosen device is never
second-guessed, and :func:`resolve_device` returns a CONCRETE name so the caller
stamps the truth rather than the word "default".

PURE and total: never raises, and degrades to the legacy triple if core is
absent (tests, tooling, a bare interpreter). UTF-8 no BOM, ASCII only.
"""
from __future__ import annotations

#: The vocabulary this pack understood before 2026-09-12. Kept FOREVER for the
#: membership reason in the module docstring -- removing one of these breaks
#: every saved graph that holds it.
LEGACY_DEVICE_NAMES: "tuple[str, ...]" = ("cuda", "cpu", "mps")

#: What a fresh graph should save. Core's word for "ask the host", and the only
#: value this module will ever resolve on the caller's behalf.
DEFAULT_DEVICE_OPTION = "default"


def _core():
    """comfy.model_management, or None outside a ComfyUI runtime."""
    try:
        import comfy.model_management as mm  # noqa: PLC0415 -- runtime only
        return mm
    except Exception:  # noqa: BLE001 -- absent in tests and bare tooling
        return None


def device_options() -> "list[str]":
    """The dropdown list: core's host-detected options, then the legacy names.

    Core yields ``["default", "cpu"]`` plus a ``gpu:N`` entry per device when the
    host has more than one -- so the list is genuinely capability-filtered by
    core, not by us. The legacy names follow so old saved values stay legal.
    """
    opts: "list[str]" = []
    mm = _core()
    if mm is not None:
        try:
            opts = [str(o) for o in mm.get_gpu_device_options()]
        except Exception:  # noqa: BLE001 -- never let a dropdown fail to build
            opts = []
    if not opts:
        opts = [DEFAULT_DEVICE_OPTION, "cpu"]
    for legacy in LEGACY_DEVICE_NAMES:
        if legacy not in opts:
            opts.append(legacy)
    return opts


def resolve_device(option, *, fallback: str = "cpu") -> str:
    """One option string -> the CONCRETE device name to record and to use.

    ``default`` and ``gpu:N`` go to core. A legacy name is honoured as written,
    because an explicit choice is never second-guessed -- that is the fail-loud
    principle the tooltips promise, and downgrading it silently is exactly what
    the 2026-07-09 portability ruling forbids.

    Returns a plain string (``"cuda"``, ``"cuda:1"``, ``"mps"``, ``"cpu"``) so
    every caller can stamp what actually ran. Never returns ``"default"``.
    """
    opt = str(option or "").strip().lower()
    if opt in LEGACY_DEVICE_NAMES:
        return opt

    # AN UNRECOGNISED VALUE IS PASSED THROUGH UNTOUCHED, NOT RESOLVED, and this
    # is the most important line in the module (caught by
    # test_resolve_inputs_rejects_bad_policy_enum, 2026-09-12). The first cut
    # sent anything it did not recognise to core and got back the host device,
    # so `llm_device="tpu"` silently became "cuda" instead of raising
    # LLMPolicyError. That turned a loud, correct refusal into exactly the
    # silent-wrong-value behaviour this pack forbids -- and it would have hidden
    # a typo in a profile or a hand-edited graph forever.
    #
    # Translating only what we RECOGNISE leaves every existing validator doing
    # its job: junk arrives at LLMRuntimePolicy / CastLock still spelled wrong,
    # and still fails loud there.
    if opt not in device_options():
        return str(option)

    mm = _core()
    if mm is None:
        return fallback

    try:
        if opt.startswith("gpu:"):
            dev = mm.resolve_gpu_device_option(opt)
            if dev is not None:
                return _name(dev)
        dev = mm.get_torch_device()
        return _name(dev)
    except Exception:  # noqa: BLE001 -- a resolver must never fail a render
        return fallback


def _name(dev) -> str:
    """A torch.device as the string our ledger and loaders already speak."""
    try:
        kind = getattr(dev, "type", None) or str(dev).split(":")[0]
        index = getattr(dev, "index", None)
        if kind == "cuda" and index not in (None, 0):
            return "cuda:%d" % int(index)
        return str(kind)
    except Exception:  # noqa: BLE001
        return "cpu"


def vendor() -> str:
    """``nvidia`` / ``amd`` / ``apple`` / ``cpu`` / ``unknown``.

    THE REASON THIS EXISTS SEPARATELY FROM THE DEVICE: quantisation travels with
    the vendor and the device does not determine it. AMD under ROCm and NVIDIA
    BOTH report ``cuda``, and they need different quant policies -- a Mac or AMD
    user who resolves the device correctly but keeps a bitsandbytes policy still
    dies on a missing library. Core draws the distinction for us with
    ``is_amd()`` / ``is_nvidia()``; without it we would be reading
    ``torch.version.hip`` by hand, which is the wheel this module refuses to
    reinvent.
    """
    mm = _core()
    if mm is None:
        return "unknown"
    try:
        if mm.is_amd():
            return "amd"
        if mm.is_nvidia():
            return "nvidia"
        dev = _name(mm.get_torch_device())
        if dev == "mps":
            return "apple"
        if dev == "cpu":
            return "cpu"
    except Exception:  # noqa: BLE001
        pass
    return "unknown"
