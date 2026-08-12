"""NAMED BOOT CONTRACTS -- what a lane needs the SERVER to have been started with.

Spec S8 (`docs/2026-08-09-SPEC-lab-findings-into-otr.md`). Some lanes only fit
under the 14.5 GiB gate if the server was launched with particular allocator
flags. HuMo 14B measures 14.98 GiB unclamped and 13.06 GiB under a reserve-VRAM
clamp with pinned memory disabled -- same graph, same weights, 1.9 GiB apart.
That is a BOOT fact, and boot facts cannot be fixed at render time: by the time
a beat renders, the server has been up for an hour.

So a contract is NAMED here, a profile SELECTS one, a launcher APPLIES it, and
preflight PROVES it -- against the running server, never against the config.

Three rules this module exists to enforce, each of them a defect that already
happened:

**The contract rides `launch.env`, never `launch.extra_args`.** `extra_args` is
written only into a markdown documentation string by `scripts/build_variants.py`;
no launcher ever turns it into argv, and `--disable-pinned-memory` appeared in
ZERO non-doc files repo-wide before this build. A boot pin placed there clamps
nothing while looking configured (lesson L6).

**Enforcement probes the RUNNING SERVER, not the profile text.** A check that
reads the same config file the launcher was supposed to honour cannot tell
"applied" from "written down", so a dead channel would also produce a falsely
passing check. :func:`running_server_boot_state` reads
``comfy.cli_args.args`` -- what this process was actually started with.

**Enforcement happens EARLY.** `assert_usable` runs inside the render phase,
after the writer, TTS, the master freeze and every still have already been paid
for. A wrong-boot refusal that arrives there costs an episode's work to learn
something knowable before the first token. The ShotLock preflight is the right
seam; the render-time check stays as defence in depth.

Only a lane that has NEVER shipped under `default` may REQUIRE a contract.
`humo_1.7B` declares `default` AND `humo_diet` compatible, so requiring the diet
would regress a shipping lane -- the diet is how a PROFILE casts it, not a
property the engine may demand.

stdlib-only, cold-import clean (V-12): importing this pulls in nothing. The one
ComfyUI touch is inside a function, guarded, and reports UNKNOWN rather than
raising when there is no server (the CPU suite must be able to import this).
UTF-8, no BOM, ASCII-only.
"""
from __future__ import annotations

#: The stock boot: no clamps. Every lane that shipped before this build is
#: compatible with it, and a profile that names no contract means this one.
DEFAULT = "default"

#: The HuMo diet (lab HUMO_DIET + ENVELOPE_LADDERS job A). Reserve VRAM away
#: from model loading and disable pinned host memory; measured to take 14B FP8
#: from 14.98 GiB to 13.06 GiB warm at 832x480x97 with the graph UNCHANGED, and
#: to take host RAM from 61.3 to roughly 26 GiB. 2.921 is not a round number
#: because it is a measurement, not a preference.
HUMO_DIET = "humo_diet"

#: The MiniMax H3 boot: sage-free (Sage silently turns H3 output to noise --
#: Comfy-Org/ComfyUI#15263, and the per-model KJ probe FAILED on sm_120) plus
#: pinned memory disabled. Declared here ahead of its adapters so the mechanism
#: and its second consumer land without a schema change.
#:
#: **`reserve_vram_gb` MOVED None -> 12.0 in lane 19 (2026-08-12), on the lab
#: receipts, and it is the difference between this lane fitting and not.** The
#: contract was drafted from the spec's prose before any H3 measurement was read
#: back. Every MiniMax H3 leg on this box that PASSED the 14.5 GiB gate was
#: booted with `--reserve-vram 12`, including the trained 1344x768 canvas at
#: model f124 (9.15 GiB absolute). The one I2V leg booted WITHOUT it --
#: `h3_i2v_canonical_832x480_f107` -- peaked at **15.39 GiB and FAILED**, on a
#: canvas 2.6x SMALLER and a length SHORTER than the leg that passed at 9.15.
#: A smaller, shorter render peaking 70% higher is not a canvas effect; the boot
#: is the only structural difference, and the mechanism is plain -- reserving 12
#: GiB away from model loading is what forces the 21 GB DiT to stream instead of
#: attempting residency on a 16 GB card.
#:
#: Left at None this contract would have named the sage and pinned-memory knobs,
#: passed its own check, and still let the lane load its way over the ceiling --
#: a contract that constrains the two knobs that were easy to write down and not
#: the one that decides the outcome.
H3 = "h3"

#: The LTX-AV diet (row 7b, operator ruling 2026-08-11). `ltx_audio_in` cleared
#: the 14.5 GiB gate on the stock boot by **35 MB (0.24%)** at 1024x576x193 --
#: a property of that minute on a lived-in desktop, not a property of the lane,
#: since a browser opening moves the idle baseline back over it. The operator
#: refused to wave that through and ruled: prove the lever.
#:
#: `reserve_vram_gb` is deliberately **None**, and that is the whole design
#: decision here rather than an omission. This adapter already holds its own
#: reserve in-process: `_ltx_av_vram_reserve` bumps ComfyUI's
#: `EXTRA_RESERVED_VRAM` to `OTR_LTX_AV_RESERVE_VRAM_GB` (default 4.0) across
#: the graph run, and it bumps only when its target EXCEEDS the current value.
#: A boot `--reserve-vram 2.921` would therefore be overwritten by the
#: adapter's own 4.0 for the entire render window -- a knob that reaches
#: nothing (L6), while looking in the profile like it did something.
#:
#: So the untested lever on THIS lane is the pinned-memory one, which is also
#: the half of the HuMo diet that could not be reproduced by `--reserve-vram`
#: alone. Same shape as H3, without the Sage constraint (LTX-AV boots on the
#: Sage-free LTX token already, but that is the token's business, not this
#: contract's -- a contract that forbids an unrelated flag is how one lane's
#: needs become another lane's mystery refusal).
LTX_AV_DIET = "ltx_av_diet"

#: name -> the boot state a server must be in. ``None`` means "this contract
#: does not constrain that knob" -- distinct from False, which would REQUIRE it
#: off. Being able to say "don't care" is what keeps a contract from silently
#: forbidding an unrelated flag.
BOOT_CONTRACTS = {
    DEFAULT: {
        "reserve_vram_gb": None,
        "disable_pinned_memory": None,
        "sage_attention": None,
    },
    HUMO_DIET: {
        "reserve_vram_gb": 2.921,
        "disable_pinned_memory": True,
        "sage_attention": None,
    },
    H3: {
        "reserve_vram_gb": 12.0,     # see the H3 note above: measured, not tidy
        "disable_pinned_memory": True,
        "sage_attention": False,
    },
    LTX_AV_DIET: {
        "reserve_vram_gb": None,     # the adapter's own in-process 4.0 dominates
        "disable_pinned_memory": True,
        "sage_attention": None,
    },
}

#: The LIVE launcher channel: `scripts/_otr_soak_server_launch.cmd` turns these
#: into argv. Any contract knob that has no env row here is unreachable, which
#: is the whole point of keeping the mapping in one visible place.
CONTRACT_ENV = {
    "reserve_vram_gb": "OTR_HEADLESS_RESERVE_VRAM_GB",
    "disable_pinned_memory": "OTR_HEADLESS_DISABLE_PINNED",
}


class BootContractError(RuntimeError):
    """The running server was not started for the contract this lane needs.

    A RuntimeError, not an EngineUnusable: the engine is fine and the weights
    are fine. What is wrong is the process, and the fix is a restart with
    different flags -- so the message names the contract, the knob, what was
    asked for and what is actually set.
    """

    reason_code = "boot_contract_unmet"


def known_contract(name) -> bool:
    return str(name or DEFAULT) in BOOT_CONTRACTS


def contract_spec(name) -> dict:
    """The required boot state for ``name``. Unknown names raise -- a typo in a
    profile must not resolve to "no constraints"."""
    key = str(name or DEFAULT)
    if key not in BOOT_CONTRACTS:
        raise BootContractError(
            "unknown boot contract %r; known contracts are %s"
            % (key, ", ".join(sorted(BOOT_CONTRACTS))))
    return dict(BOOT_CONTRACTS[key])


def launch_env_for(name) -> dict:
    """The ``launch.env`` rows a profile must carry to REACH this contract.

    Values are strings because that is what the env channel and the profile
    schema both hold. `sage_attention` is deliberately absent: no launcher
    passes an attention flag today, so emitting an env row for it would be
    another configured-knob-that-reaches-nothing (lesson L6). A sage-sensitive
    lane refuses at `assert_usable` via `assert_sage_not_patched` instead, which
    is enforcement that actually runs.
    """
    spec = contract_spec(name)
    env = {}
    if spec["reserve_vram_gb"] is not None:
        env[CONTRACT_ENV["reserve_vram_gb"]] = "%g" % spec["reserve_vram_gb"]
    if spec["disable_pinned_memory"]:
        env[CONTRACT_ENV["disable_pinned_memory"]] = "1"
    return env


def running_server_boot_state() -> dict:
    """What THIS process was actually started with.

    ``{"available": False, "error": <named reason>}`` off a ComfyUI server --
    the CPU suite has no `comfy.cli_args`, and reporting that honestly is what
    stops a headroom-gated branch from being silently unreachable while the
    tests stay green. Never raises.
    """
    try:
        from comfy.cli_args import args  # ComfyUI runtime only
    except Exception as exc:  # noqa: BLE001
        return {"available": False,
                "error": "comfy.cli_args unavailable (%s)" % type(exc).__name__}
    state = {"available": True}
    reserve = getattr(args, "reserve_vram", None)
    state["reserve_vram_gb"] = None if reserve is None else float(reserve)
    state["disable_pinned_memory"] = bool(
        getattr(args, "disable_pinned_memory", False))
    try:
        # RELATIVE, and it has to be (lane 19, 2026-08-12 -- found by a live
        # smoke, not by a test). This read `from nodes._otr_video_engines...`,
        # an ABSOLUTE import that resolves `nodes` against sys.path. In the CPU
        # suite the repo root is on sys.path, so `nodes` IS this package and the
        # probe worked. Inside a running ComfyUI server it is NOT: ComfyUI has
        # its own top-level `nodes` module (the node registry), OTR lives under
        # custom_nodes/ComfyUI-OldTimeRadio, and the import raised
        # ModuleNotFoundError on every boot.
        #
        # The failure was silent until this lane because the probe's error is
        # only CONSULTED when a contract constrains Sage, and `h3` is the first
        # one that does -- so the bug shipped dormant behind a `default` and a
        # `humo_diet` that both say "don't care" about Sage. It then presented
        # as an unrenderable lane: UNKNOWN is correctly not a pass, so the H3
        # lane refused on every server, for a reason that was about the import
        # and not about Sage.
        #
        # A relative import resolves through THIS package's own __name__, which
        # is correct under both names.
        from .._otr_video_engines.motion_common import sageattention_patched
        state["sage_attention"] = bool(sageattention_patched())
    except Exception as exc:  # noqa: BLE001
        state["sage_attention"] = None
        state["sage_probe_error"] = type(exc).__name__
    return state


def check_running_server(name, state=None) -> list:
    """Every way the running server FAILS ``name``, as sentences. Empty = met.

    Returns rather than raises so a caller can report several lanes at once,
    and so the "no server" case is a decision the caller makes rather than an
    exception it has to catch. A reserve-VRAM comparison is a >= : reserving
    MORE than the contract asks for is still inside the envelope the contract
    was measured in, while reserving less is not.
    """
    spec = contract_spec(name)
    state = running_server_boot_state() if state is None else dict(state)
    constrained = [k for k in ("reserve_vram_gb", "disable_pinned_memory",
                               "sage_attention") if spec.get(k) is not None]
    if not state.get("available"):
        # UNKNOWN IS NOT SATISFIED (retro bug hunt r1, 2026-08-11 -- both
        # reviewers reached this independently). This returned a bare `[]` and
        # the comment said "the caller decides". No caller decides:
        # `assert_running_server` treats an empty list as MET and raises
        # nothing, so a contract that constrains real clamps evaluated as
        # compliant on any box where `comfy.cli_args` cannot be imported.
        #
        # A contract that constrains NOTHING (`default`) is genuinely satisfied
        # by an unknown server -- there is nothing to violate -- so the
        # distinction is drawn on what the SPEC asks for, not on whether we
        # happened to be able to look.
        if not constrained:
            return []
        return ["boot contract %r constrains %s, but the running server's boot "
                "state could not be read (%s) -- UNKNOWN is not satisfied: a "
                "clamp that cannot be verified may not be assumed"
                % (name, ", ".join(constrained),
                   state.get("unavailable_reason") or "comfy.cli_args "
                   "unavailable")]
    problems = []
    want = spec["reserve_vram_gb"]
    if want is not None:
        got = state.get("reserve_vram_gb")
        if got is None or float(got) + 1e-6 < float(want):
            problems.append(
                "boot contract %r needs --reserve-vram >= %g GiB (via %s); the "
                "server was started with %r"
                % (name, want, CONTRACT_ENV["reserve_vram_gb"], got))
    want = spec["disable_pinned_memory"]
    if want is not None and bool(state.get("disable_pinned_memory")) != bool(want):
        problems.append(
            "boot contract %r needs --disable-pinned-memory %s (via %s); the "
            "server has it %s"
            % (name, "ON" if want else "OFF",
               CONTRACT_ENV["disable_pinned_memory"],
               "ON" if state.get("disable_pinned_memory") else "OFF"))
    want = spec["sage_attention"]
    got = state.get("sage_attention")
    if want is not None:
        if got is None:
            # `running_server_boot_state` records `sage_probe_error` when the
            # probe raises and NOTHING read it -- so a failed probe left
            # `sage_attention = None` and this comparison skipped entirely,
            # silently passing a Sage-constrained contract. Recording an error
            # nobody reads is the same as swallowing it (retro bug hunt r1).
            problems.append(
                "boot contract %r needs SageAttention %s, but the probe could "
                "not determine it (%s) -- an unverifiable Sage state is not a "
                "pass on a lane Sage silently corrupts"
                % (name, "ACTIVE" if want else "ABSENT",
                   state.get("sage_probe_error") or "no reason recorded"))
        elif bool(got) != bool(want):
            problems.append(
                "boot contract %r needs SageAttention %s; it is %s on this "
                "server" % (name, "ACTIVE" if want else "ABSENT",
                            "ACTIVE" if got else "ABSENT"))
    return problems


def assert_running_server(name, state=None) -> None:
    """Raise :class:`BootContractError` naming every unmet knob, or return."""
    problems = check_running_server(name, state=state)
    if problems:
        raise BootContractError(
            "the running server does not satisfy boot contract %r -- this is "
            "not fixable at render time, the server must be restarted with the "
            "contract's launch.env applied: %s" % (name, "; ".join(problems)))


def contract_for_profile(profile) -> str:
    """The contract a profile SELECTS, defaulting to ``default``.

    Reads ``launch.boot_contract``. A profile that names none is on the stock
    boot, which is what every profile shipped before this build meant.
    """
    launch = (profile or {}).get("launch") or {}
    return str(launch.get("boot_contract") or DEFAULT)


def compatible_contracts_for_engine(engine) -> tuple:
    """The contracts an ENGINE will run under.

    An adapter that declares nothing is compatible with everything, which
    preserves every lane that shipped before this mechanism existed. An adapter
    that declares a tuple is stating the boots it has been proven on -- and only
    a lane that never shipped under ``default`` may leave ``default`` out.
    """
    declared = getattr(engine, "compatible_boot_contracts", None)
    if not declared:
        return tuple(sorted(BOOT_CONTRACTS))
    return tuple(declared)


def check_engine_against_profile(engine, profile) -> list:
    """Every reason this engine may not run on this profile's boot, as
    sentences. Empty = compatible. Pure: no server, no CUDA, no imports."""
    name = contract_for_profile(profile)
    if not known_contract(name):
        return ["profile selects unknown boot contract %r; known contracts are "
                "%s" % (name, ", ".join(sorted(BOOT_CONTRACTS)))]
    allowed = compatible_contracts_for_engine(engine)
    if name not in allowed:
        return ["engine %r is proven on boot contract(s) %s, and this profile "
                "selects %r" % (getattr(engine, "name", "?"),
                                ", ".join(allowed), name)]
    return []


__all__ = [
    "DEFAULT", "HUMO_DIET", "H3", "BOOT_CONTRACTS", "CONTRACT_ENV",
    "BootContractError", "known_contract", "contract_spec", "launch_env_for",
    "running_server_boot_state", "check_running_server",
    "assert_running_server", "contract_for_profile",
    "compatible_contracts_for_engine", "check_engine_against_profile",
]
