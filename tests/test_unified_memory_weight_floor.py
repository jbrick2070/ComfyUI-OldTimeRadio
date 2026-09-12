"""The unified-memory weight floor -- the guard that turns a machine crash into
a sentence.

WHY THIS FILE EXISTS. On 2026-09-08 an Apple Silicon session loaded ``wan_ti2v``
with its fp16 UNET on a 16 GB M4. It loaded cleanly on Metal -- WanTEModel
10835 MB, WanVAE 1344 MB, then ``WAN22 ... loaded completely; 9536.40 MB, full
load: True`` -- and then the OS killed everything. On unified memory an OOM is
not a failed render with a traceback to read; the offload device is the same
physical RAM, so there is nowhere to spill and the whole machine goes down.

Nothing in the pack objected, because nothing was asking whether the weights
fit. ``QUALIFIED_COST_ROWS`` is empty by operator ruling, so the calibrated cost
model refuses nothing; that is deliberate and this guard does not touch it. This
is a separate, uncalibrated, file-sizes-only floor check.

THE CASES BELOW ARE GROUND TRUTH, NOT FIXTURES. Every row was observed on one
physical machine, and the guard is only worth having if it reproduces all six.
Two of them are failures that actually happened; four are configurations that
actually ran. A change that makes any row flip has broken the guard, and the
ltx_8gb row is the one to watch -- the FIRST implementation summed every
declared artifact and refused it, which is a false positive on a proven-working
lane. That is the error direction that matters: a false refusal blocks work that
succeeds, a false allowance only leaves things as they were.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes._otr_video_engines import motion_common as mc  # noqa: E402


#: The budget for a 16 GB Mac, as :func:`motion_common.unified_memory_budget_mb`
#: computes it: physical RAM x 1.15, the observed swap tolerance.
#:
#: THIS USED TO BE THE METAL WORKING-SET CEILING (12123.7 MiB) AND THAT WAS THE
#: WRONG POOL -- PBUG-20260908-02. ComfyUI parks a text encoder on its "offload
#: device: cpu", where `torch.mps.recommended_max_memory()` cannot see it and it
#: consumes the same physical pages regardless. Budgeting against the Metal
#: ceiling therefore ignored exactly the memory that killed the machine.
M4_16GB_BUDGET_MB = 16384.0 * 1.15

#: (label, concurrent_resident_mb, must_refuse).
#:
#: EVERY ARTIFACT IS SUMMED, because on this platform every artifact IS
#: concurrently resident -- PBUG-20260908-02. The earlier version of this table
#: gave each row an eviction credit (`max(encoder, sum(rest))`) and it was
#: fiction: wan_ti2v passed free_after_use=True and logged "0 models unloaded"
#: right before the load that killed the box, flux2_klein held a 7.67 GB encoder
#: through sampling, and ltx_8gb never attempted an unload at all.
GROUND_TRUTH = [
    # --- the two that actually failed on this machine -------------------
    ("wan_ti2v fp16: umt5 10835 + UNET 9536 + VAE 1344, ALL RESIDENT "
     "(log: '0 models unloaded') -- KILLED THE MACHINE",
     10835.0 + 9536.0 + 1344.0, True),
    ("a hypothetical lane at 20 GiB, no single artifact near the line",
     7000.0 + 7000.0 + 6480.0, True),
    # --- the four that ran ----------------------------------------------
    # NOTE ltx_8gb: 15.75 GiB EXCEEDS this machine's 15.9 GiB of physical RAM
    # and it still published episodes, on swap, slowly. That is precisely why
    # the budget is physical RAM x 1.15 and not physical RAM: a bare-RAM
    # threshold refuses a lane with receipts.
    ("ltx_8gb: t5xxl 9787 + ckpt 6340 both resident -- PUBLISHED EPISODES",
     9787.0 + 6340.0, False),
    ("flux2_klein: qwen_3_4b 7672 + Q4 2592 + vae 336 -- RAN at 23.5 s/step",
     7672.0 + 2592.0 + 336.0, False),
    ("wan_ti2v GGUF: the SHIPPED set, Q5_K_M 3810 + umt5 3860 + VAE 1310",
     3810.0 + 3860.0 + 1310.0, False),
    ("animatediff haunted: sd15 1990 + mm 1560 + adapter 95 (4060: 4.9 GB)",
     1990.0 + 1560.0 + 95.0, False),
]


@pytest.mark.parametrize("label,peak_mb,must_refuse", GROUND_TRUTH,
                         ids=[c[0].split(":")[0] for c in GROUND_TRUTH])
def test_the_guard_reproduces_what_the_hardware_did(label, peak_mb,
                                                    must_refuse):
    """Six configurations, six known outcomes, no calibration."""
    verdict = mc.unified_memory_weight_refusal("eng", peak_mb, M4_16GB_BUDGET_MB)
    if must_refuse:
        assert verdict, (
            "%s -- this configuration DID fail on the hardware and the guard "
            "let it through. A missed refusal here is a machine crash, not a "
            "failed test run." % label)
        assert "MACHINE down" in verdict, (
            "the refusal must say what is actually at stake; a generic "
            "out-of-memory message reads as a render problem")
    else:
        assert verdict is None, (
            "%s -- this configuration DID run on the hardware and the guard "
            "refused it. A false refusal blocks working work, which is worse "
            "than the guard not existing. Got: %s" % (label, verdict))


def test_ltx_survives_only_because_the_budget_allows_swap():
    """ltx_8gb holds 15.75 GiB concurrently on a machine with 15.9 GiB of
    physical RAM, and it published episodes anyway -- slowly, on swap.

    This pins the reason the budget is not simply physical RAM: that threshold
    would refuse a lane that demonstrably works. It also pins the other side --
    the tolerance is not unlimited, and wan_ti2v's 21.2 GiB is past it."""
    physical_only = 16384.0
    assert mc.unified_memory_weight_refusal(
        "ltx_8gb", 9787.0 + 6340.0, physical_only - 1000.0), (
        "sanity: against a bare-RAM budget LTX would be refused, which is the "
        "mistake the swap tolerance exists to avoid")
    assert mc.unified_memory_weight_refusal(
        "ltx_8gb", 9787.0 + 6340.0, M4_16GB_BUDGET_MB) is None


@pytest.mark.parametrize("weight_mb,free_mb", [
    (None, M4_16GB_BUDGET_MB),          # nothing resolved
    (99999.0, None),                  # no probe (CPU box, older torch)
    (0.0, M4_16GB_BUDGET_MB),           # engine declares no weights
    (-5.0, M4_16GB_BUDGET_MB),          # nonsense size
    (99999.0, 0.0),                   # probe returned zero
    (99999.0, -1.0),                  # probe returned nonsense
    (float("nan"), M4_16GB_BUDGET_MB),  # NaN size
    (99999.0, float("inf")),          # infinite budget
    ("big", M4_16GB_BUDGET_MB),         # non-numeric
])
def test_it_fails_open_on_every_input_it_cannot_trust(weight_mb, free_mb):
    """Allow whenever the number is not real.

    An uncalibrated guard gets exactly one acceptable error direction. Every
    path that cannot produce an exact figure must allow, so the worst outcome
    of a bad input is today's behaviour rather than a blocked render."""
    assert mc.unified_memory_weight_refusal("eng", weight_mb, free_mb) is None


def test_the_headroom_is_reserved_not_ignored():
    """A model that fits the raw budget but not the headroom is refused."""
    free = 10000.0
    assert mc.unified_memory_weight_refusal("eng", 9000.0, free,
                                            headroom_mb=0.0) is None
    assert mc.unified_memory_weight_refusal("eng", 9000.0, free,
                                            headroom_mb=1536.0)


@pytest.mark.parametrize("bad", ["x", float("nan"), float("inf"), -1.0, -9e9])
def test_a_broken_headroom_override_falls_back_rather_than_disarming(bad):
    """``OTR_UNIFIED_MEMORY_HEADROOM_MB`` is an escape hatch, not a kill switch.

    The negative cases are the point, and they are why this test exists at a
    threshold that a headroom of zero would ALLOW. The first implementation did
    ``max(0.0, float(headroom_mb))``, so ``-1`` became a reservation of zero and
    the escape hatch silently became a disable switch. Asserting on a model that
    is only refused BECAUSE of the reservation is what catches that; asserting
    on one that is refused anyway would have passed against the bug."""
    just_over_with_reservation = M4_16GB_BUDGET_MB - 800.0   # fits raw, not with 1.5 GiB
    verdict = mc.unified_memory_weight_refusal("eng", just_over_with_reservation,
                                               M4_16GB_BUDGET_MB,
                                               headroom_mb=bad)
    assert verdict, ("a malformed headroom override (%r) disarmed the "
                     "reservation -- it must fall back to the default" % (bad,))


def test_headroom_none_means_the_default_not_a_broken_override():
    """``None`` is the ordinary "operator said nothing" path, not garbage; it
    reads the env/default rather than being rejected."""
    just_over = M4_16GB_BUDGET_MB - 800.0
    assert mc.unified_memory_weight_refusal("eng", just_over, M4_16GB_BUDGET_MB,
                                            headroom_mb=None)
    assert mc.unified_memory_weight_refusal("eng", 1000.0, M4_16GB_BUDGET_MB,
                                            headroom_mb=None) is None


def test_the_refusal_names_the_way_out():
    """A refusal a reader cannot act on is a dead end. It has to say that a
    quantised build is the fix, because for every lane refused so far one
    exists and is the SHIPPED configuration."""
    verdict = mc.unified_memory_weight_refusal("wan_ti2v", 20000.0,
                                               M4_16GB_BUDGET_MB)
    assert verdict
    assert "quantised" in verdict
    assert "otr_fetch_lane_weights" in verdict
    assert "wan_ti2v" in verdict


def test_a_discrete_card_is_left_alone():
    """The CUDA/Metal asymmetry is the load-bearing design decision here.

    On a discrete card, weights larger than VRAM are SURVIVABLE -- ComfyUI
    offloads to host RAM, so refusing them would break working NVIDIA
    configurations. The guard must therefore be a strict no-op whenever CUDA is
    present, and this test is what stops someone "simplifying" it into a
    universal check and regressing every 5080 lane that relies on offload."""
    seen = {}

    class _FakeCuda:
        @staticmethod
        def is_available():
            seen["asked_cuda"] = True
            return True

    class _FakeMps:
        @staticmethod
        def is_available():  # pragma: no cover -- must never be reached
            seen["asked_mps"] = True
            return True

    class _FakeBackends:
        mps = _FakeMps

    fake = type(sys)("torch")
    fake.cuda = _FakeCuda
    fake.backends = _FakeBackends
    real = sys.modules.get("torch")
    sys.modules["torch"] = fake
    try:
        assert mc._unified_memory_backend() is False
    finally:
        if real is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = real
    assert seen.get("asked_cuda") is True
    assert "asked_mps" not in seen, (
        "CUDA was present and the guard still asked about Metal -- on a "
        "discrete card this check must short-circuit")


def test_the_guard_never_raises_anything_but_its_own_error():
    """A guard that can itself crash the render is worse than no guard.

    ``refuse_if_weights_exceed_unified_memory`` swallows everything except its
    own MotionBudgetError, so an unregistered engine, a missing folder_paths,
    or a torch that raises on import cannot take a render down."""
    for name in (None, "", "no_such_engine_anywhere", 12345):
        mc.refuse_if_weights_exceed_unified_memory(name)


def test_free_vram_mb_reports_a_real_number_on_metal():
    """The probe this guard reads. It returned ``None`` on every Mac until
    2026-09-08, which is why the frame budget was silently disabled there --
    ``compute_real_frame_budget`` treats None as "no budget known" and stops
    predicting."""
    try:
        import torch  # noqa: F401
    except ImportError:  # pragma: no cover
        pytest.skip("torch not installed")
    if not getattr(getattr(torch.backends, "mps", None), "is_available",
                   lambda: False)():
        pytest.skip("not an Apple Silicon host")
    free = mc.free_vram_mb()
    assert free is not None, (
        "free_vram_mb() returned None on a Metal host -- the frame budget and "
        "the weight floor are both blind again")
    assert math.isfinite(free) and free > 0
    assert free < 1024.0 * 1024.0, "implausible: is this bytes rather than MiB?"


# ---------------------------------------------------------------------------
# THE RESOLVER. Everything above tests arithmetic on numbers handed in by the
# test. That is exactly how the first version of this guard shipped BROKEN and
# green: `resolved_weight_mb` read `model_requirements`, which holds S5 wizard
# asset ids rather than filenames -- `wan_ti2v` declares "wan2.2-ti2v-5b" while
# its loader consumes "Wan2.2-TI2V-5B-Q5_K_M.gguf" -- so folder_paths resolved
# nothing, the guard fail-opened on the one engine that had just killed the
# machine, and every assertion above still passed. A cursor review caught it.
# These tests exist so that cannot recur silently.
# ---------------------------------------------------------------------------

def test_the_resolver_reads_loader_filenames_not_wizard_asset_ids():
    """The regression that made the guard inert, pinned at its source."""
    split = mc._loader_filenames("wan_ti2v")
    assert split, "wan_ti2v must be resolvable -- it is the reason this exists"
    encoders, resident = split
    names = list(encoders) + list(resident)
    assert all(("." in n) for n in names), (
        "a resolved name with no extension is a wizard asset id, not a file "
        "the loader will open: %r" % (names,))
    assert any("TI2V" in n or "ti2v" in n for n in resident)
    assert any("vae" in n.lower() for n in resident), (
        "the VAE must be in the RESIDENT set -- it is concurrent with the "
        "UNET, and that concurrency is what took the machine down")
    assert encoders, "the text encoder must be tracked, in its own phase"

    from nodes._otr_video_engines import registry as vreg
    wizard = list((vreg.CAPABILITIES.get("wan_ti2v") or {}).get(
        "model_requirements") or [])
    assert wizard and not set(wizard) & set(names), (
        "model_requirements and the loader names must stay distinct; if they "
        "ever coincide, this test is no longer proving anything")


def test_the_resolver_puts_the_text_encoder_in_its_own_phase():
    """ltx_8gb is the shape that matters: a big encoder and a smaller model
    that are never resident together."""
    encoders, resident = mc._loader_filenames("ltx_8gb")
    assert any("t5" in n.lower() for n in encoders)
    assert any("ltxv" in n.lower() for n in resident)
    assert not any("t5" in n.lower() for n in resident), (
        "the T5 encoder leaked into the resident set -- summed with the "
        "checkpoint it refuses a lane with published episodes")


def test_an_engine_with_no_weights_is_not_guarded_at_all():
    """Procgen and still lanes carry nothing to weigh; the guard must not
    invent a number for them."""
    for name in ("viz_camera", "viz_green", "still_flat", "still_pan"):
        assert mc._loader_filenames(name) is None
        assert mc.resolved_weight_mb(name) is None


def test_the_resolver_follows_the_env_override_the_loader_follows(monkeypatch):
    """THE CRASH, REPRODUCED AS A UNIT TEST.

    The machine died because OTR_WAN_TI2V_UNET_NAME pointed at the fp16 build
    instead of the shipped GGUF. The guard is only useful if it weighs what
    will ACTUALLY load, so it has to honour the same env the loader does -- a
    guard that weighs the defaults while the loader opens an override is
    worse than none, because it reports safety it has not checked."""
    monkeypatch.setenv("OTR_WAN_TI2V_UNET_NAME",
                       "wan2.2_ti2v_5B_fp16.safetensors")
    monkeypatch.setenv("OTR_WAN_TI2V_CLIP_NAME", "umt5_xxl_fp16.safetensors")
    _encoders, resident = mc._loader_filenames("wan_ti2v")
    assert "wan2.2_ti2v_5B_fp16.safetensors" in resident, (
        "the guard is still weighing the default GGUF while the loader would "
        "open the fp16 override -- this is the crash, unguarded")


def _fake_folder_paths(tmp_path, encoder_name):
    class _FakeFolderPaths:
        @staticmethod
        def get_full_path(category, name):
            is_enc = category in ("text_encoders", "clip")
            if is_enc != (name == encoder_name):
                return None
            p = tmp_path / name
            return str(p) if p.exists() else None
    return _FakeFolderPaths


def _stage(tmp_path, sizes):
    for name, mb in sizes.items():
        with open(tmp_path / name, "wb") as fh:
            fh.truncate(mb * 1024 * 1024)


SIZES = {"enc.safetensors": 5000, "unet.safetensors": 6000,
         "vae.safetensors": 1000}


def test_on_unified_memory_every_artifact_is_charged(monkeypatch, tmp_path):
    """PBUG-20260908-02: the eviction does not happen on Metal, so the encoder
    is charged alongside the model rather than given its own phase.

    5000 + 6000 + 1000 = 12000. A two-phase answer would be max(5000, 7000) =
    7000, and the gap between them is an entire text encoder -- which is exactly
    the amount by which the machine was under-charged when it died."""
    _stage(tmp_path, SIZES)
    monkeypatch.setitem(sys.modules, "folder_paths",
                        _fake_folder_paths(tmp_path, "enc.safetensors"))
    monkeypatch.setattr(mc, "_loader_filenames",
                        lambda n: (["enc.safetensors"],
                                   ["unet.safetensors", "vae.safetensors"]))
    monkeypatch.setattr(mc, "_unified_memory_backend", lambda: True)
    assert mc.resolved_weight_mb("anything") == pytest.approx(12000, rel=0.01)


def test_on_a_discrete_card_the_two_phase_credit_still_applies(monkeypatch,
                                                               tmp_path):
    """CUDA is the platform where free_after_use demonstrably WORKS -- the 5080
    measured a 5-7 GB lower peak from it (PBUG-20260902-01). So the eviction
    credit is correct there and must not be removed by the Metal fix.

    max(encoder 5000, resident 6000 + 1000) = 7000."""
    _stage(tmp_path, SIZES)
    monkeypatch.setitem(sys.modules, "folder_paths",
                        _fake_folder_paths(tmp_path, "enc.safetensors"))
    monkeypatch.setattr(mc, "_loader_filenames",
                        lambda n: (["enc.safetensors"],
                                   ["unet.safetensors", "vae.safetensors"]))
    monkeypatch.setattr(mc, "_unified_memory_backend", lambda: False)
    assert mc.resolved_weight_mb("anything") == pytest.approx(7000, rel=0.01)


def test_one_unresolvable_artifact_makes_the_whole_answer_none(monkeypatch):
    """All-or-nothing. A partial sum looks like an answer and is not one, and
    this guard's only value is that its number is exact."""
    class _FakeFolderPaths:
        @staticmethod
        def get_full_path(category, name):
            return None

    monkeypatch.setitem(sys.modules, "folder_paths", _FakeFolderPaths)
    monkeypatch.setattr(mc, "_loader_filenames",
                        lambda n: ([], ["missing.safetensors"]))
    assert mc.resolved_weight_mb("anything") is None


# ---------------------------------------------------------------------------
# RESIDENCY IS A PER-ENGINE CONTRACT. Two engines in this pack do opposite
# things and the guard has to read which is which, because guessing wrong in
# the HuMo direction under-counts by the size of a text encoder -- an ALLOW on
# a configuration that takes the machine down.
# ---------------------------------------------------------------------------

def test_humo_is_fully_resident_and_every_artifact_is_summed():
    """``eng_humo._session_node_ids``: "HuMo renders FULLY RESIDENT by contract
    (BUG-265: forcing inter-node eviction fragmented the allocator into an
    OOM)". Its umt5, whisper, UNET, LoRA and VAE are held at once, so none of
    them may be given a phase of its own."""
    encoders, resident = mc._loader_filenames("humo")
    assert encoders == [], (
        "HuMo put an artifact in the encoder phase -- that artifact would be "
        "max()'d instead of summed, and it is resident the whole render")
    assert any("umt5" in n for n in resident), (
        "the umt5 encoder must be in the RESIDENT set for a fully-resident lane")
    assert any("whisper" in n for n in resident)
    assert mc._encoder_is_evicted("humo") is False


def test_wan_and_ltx_are_two_phase_because_they_evict():
    """Both call ``run_graph(..., free_after_use=True)``; wan_ti2v's own comment
    says it exists "so the umt5 text encode frees before the 5B UNET"."""
    for name in ("ltx_8gb", "wan_ti2v", "fastwan_8gb"):
        assert mc._encoder_is_evicted(name) is True, name
        encoders, _resident = mc._loader_filenames(name)
        assert encoders, "%s evicts, so its encoder gets its own phase" % name


def test_eviction_is_parsed_not_grepped():
    """THE BUG THIS REPLACED, pinned.

    The first implementation asked ``"free_after_use=True" in source``. That
    matched ``eng_humo._session_node_ids``' docstring -- which contains the
    sentence "WAN renders with ``free_after_use=True``" while explaining that
    HuMo does the opposite -- and so reported the pack's heaviest lane as
    evicting. Prose about another engine read as this engine's behaviour."""
    import inspect

    from nodes._otr_video_engines import eng_humo
    src = inspect.getsource(eng_humo)
    assert "free_after_use=True" in src, (
        "this test is only meaningful while eng_humo's source still MENTIONS "
        "the string; if that prose is gone, the trap is gone with it")
    assert mc._encoder_is_evicted("humo") is False, (
        "a substring search over the source is back -- it is reading HuMo's "
        "description of WAN as a statement about HuMo")


def test_a_placeholder_loader_name_is_not_treated_as_a_file():
    """``eng_humo``'s optional LoRA slot returns the string "none". Under
    all-or-nothing resolution an unresolvable name returns None for the WHOLE
    engine, so one placeholder silently disarms the guard for that lane."""
    for name in ("humo", "humo_1.7B"):
        split = mc._loader_filenames(name)
        assert split, name
        _encoders, resident = split
        assert not any(n.strip().lower() in mc._LOADER_NAME_PLACEHOLDERS
                       for n in resident), (
            "%s leaked a placeholder into the resolvable set: %r"
            % (name, resident))


# ---------------------------------------------------------------------------
# END TO END. Everything above can pass against an impure half that never
# refuses -- agy's point was that replacing the body of
# refuse_if_weights_exceed_unified_memory with `return` left all 33 tests
# green. These two are what make that mutation fail.
# ---------------------------------------------------------------------------

def _fake_unified_torch():
    class _Cuda:
        @staticmethod
        def is_available():
            return False

    class _Mps:
        @staticmethod
        def is_available():
            return True

    class _Backends:
        mps = _Mps

    fake = type(sys)("torch")
    fake.cuda = _Cuda
    fake.backends = _Backends
    return fake


def test_the_impure_half_actually_raises_on_an_oversized_engine(monkeypatch):
    """The mutation test: stub the guard's body to `return` and this fails.

    PINNED TO THE 16 GB BUDGET (2026-09-12): the guard budgets against
    `unified_memory_budget_mb`, which reads PHYSICAL RAM (PBUG-20260908-02),
    not `free_vram_mb`; on the 63 GB reference box a 20 GB weight fits and
    nothing raised, so this test failed in every run here."""
    monkeypatch.setitem(sys.modules, "torch", _fake_unified_torch())
    monkeypatch.setattr(mc, "free_vram_mb", lambda: M4_16GB_BUDGET_MB)
    monkeypatch.setattr(mc, "unified_memory_budget_mb", lambda: M4_16GB_BUDGET_MB)
    monkeypatch.setattr(mc, "resolved_weight_mb", lambda name: 20000.0)
    with pytest.raises(mc.MotionBudgetError) as excinfo:
        mc.refuse_if_weights_exceed_unified_memory("some_engine")
    assert "MACHINE down" in str(excinfo.value)


def test_the_impure_half_allows_an_engine_that_fits(monkeypatch):
    """The other half of the mutation test: a guard hard-wired to raise fails
    here, and a guard hard-wired to return fails above."""
    monkeypatch.setitem(sys.modules, "torch", _fake_unified_torch())
    monkeypatch.setattr(mc, "free_vram_mb", lambda: M4_16GB_BUDGET_MB)
    monkeypatch.setattr(mc, "unified_memory_budget_mb", lambda: M4_16GB_BUDGET_MB)
    monkeypatch.setattr(mc, "resolved_weight_mb", lambda name: 3000.0)
    mc.refuse_if_weights_exceed_unified_memory("some_engine")


def test_a_discrete_card_is_not_refused_even_when_the_weights_are_huge(
        monkeypatch):
    """The CUDA no-op, end to end rather than via the backend probe alone.

    A 20 GB model on a discrete card is SLOW (ComfyUI offloads to host RAM),
    not fatal. Refusing it here would break working NVIDIA lanes, which is the
    one outcome this guard must never produce."""
    class _Cuda:
        @staticmethod
        def is_available():
            return True

    class _Backends:
        mps = type("_M", (), {"is_available": staticmethod(lambda: True)})

    fake = type(sys)("torch")
    fake.cuda = _Cuda
    fake.backends = _Backends
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(mc, "free_vram_mb", lambda: 8000.0)
    monkeypatch.setattr(mc, "resolved_weight_mb", lambda name: 28000.0)
    mc.refuse_if_weights_exceed_unified_memory("some_engine")
