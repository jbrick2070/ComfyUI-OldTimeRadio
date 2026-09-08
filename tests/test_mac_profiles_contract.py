"""The two Apple Silicon profiles, pinned against the mistakes they were built from.

`config/profiles/otr_mac_mps.json` shipped for months naming `viz_mxc_mandala`
(needs pycairo, which publishes no macOS wheel), `z_image_turbo` for all three
image roles (12.3 GB, OOMs at 16 GB; its int8 build hits `aten::_int_mm`, which
MPS does not implement), a required `OTR_GOOGLE_API_KEY` for images that are
local, and an `llm.lane_allowlist` MISSING "transformers" -- which is
NO-FALLBACK enforced, so `--profile otr_mac_mps` would have refused the very
writer the platform was proven with. Every one of those was a foot-gun aimed at
the one reader most likely to trust the profile over the docs.

None of it was caught by a test, because `tests/test_capability_profiles.py`
drives a fixed `TIERS` tuple that does not include either Mac profile. This file
is the missing coverage, and it is deliberately written against the SPECIFIC
failures rather than as a generic schema check -- a schema check would have
passed on all four of them.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)

from nodes._otr_image_engines import registry as ireg  # noqa: E402
from nodes._otr_shared import capability_profiles as cp  # noqa: E402
from nodes._otr_video_engines import registry as vreg  # noqa: E402

MAC_PROFILES = ("otr_mac_mps", "otr_mac_adiff")

#: Engines a Mac profile must not SELECT BY DEFAULT, with the reason. Note the
#: two reasons are different in kind, and neither is visible to
#: ``cp.availability()`` -- which is exactly why this dict exists rather than
#: leaning on the device rows.
#:
#: ``viz_mxc_mandala``'s row says ["cuda","cpu","mps"] and that row is CORRECT:
#: it is pure CPU vector graphics and genuinely runs on any device. What stops
#: it here is a missing SYSTEM LIBRARY (pycairo publishes no macOS wheel), which
#: is a different axis from the device, and deliberately not a
#: ``required_toolchain`` -- registry.py:315 explains that pycairo is kept out
#: of the main requirements so a box without libcairo cannot break any OTHER
#: engine's install. It is reachable after `brew install cairo pkg-config`; it
#: must simply not be the DEFAULT a Mac reader lands on.
#:
#: ``z_image_turbo`` is a hardware fact rather than an install step: 12.3 GB
#: bf16 needs ~20.4 GiB in the KSampler at 16 GB, and the int8 build hits
#: aten::_int_mm, which MPS does not implement. No amount of installing fixes
#: that one.
BANNED_ON_MAC = {
    "viz_mxc_mandala": ("pycairo has no macOS wheel, so assert_usable refuses "
                        "until `brew install cairo pkg-config` -- not a "
                        "default a Mac reader should land on"),
    "z_image_turbo": "12.3 GB bf16 OOMs at 16 GB; int8 hits aten::_int_mm",
}


#: Engines a DRAFT Mac profile may name while their own registry row still says
#: they do not run here. Each entry is a promise that somebody looked.
DRAFT_KNOWN_INADMISSIBLE = {
    "animatediff15_v3_haunted_video": (
        "row is [\"cuda\"], reverted from [\"cuda\",\"mps\"] on 2026-09-08 "
        "because no clip had landed. The registry comment says to add mps only "
        "after one does; the profile and the row move together, on a receipt."),
}


def _profile(name):
    with open(os.path.join(_REPO, "config", "profiles", name + ".json"),
               encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def decls():
    return {**vreg.CAPABILITIES, **ireg.CAPABILITIES}


@pytest.mark.parametrize("name", MAC_PROFILES)
def test_it_is_a_mac_mps_profile_at_all(name):
    p = _profile(name)
    assert p["platform"] == "mac"
    assert p["device_backend"] == "mps"
    assert p["gpu_vendor"] == "apple"
    assert p["video"]["device_policy"] == "mps"
    assert p["allow_sidecars"] is False, (
        "every sidecar engine in the pack is pinned to cu128 torch with a "
        "PowerShell-only installer; allowing them here promises nothing")


@pytest.mark.parametrize("name", MAC_PROFILES)
def test_no_role_names_an_engine_measured_broken_on_this_platform(name):
    """The mandala and z_image foot-guns, pinned by name and by reason."""
    p = _profile(name)
    for role, engine in p["role_overrides"].items():
        assert engine not in BANNED_ON_MAC, (
            "%s.%s = %s -- %s" % (name, role, engine, BANNED_ON_MAC.get(engine)))


@pytest.mark.parametrize("name", MAC_PROFILES)
def test_the_writer_lane_it_uses_is_the_lane_it_allows(name):
    """THE SILENT ONE. `lane_allowlist` is enforced NO-FALLBACK in
    `_otr_model_loader`, and otr_mac_mps omitted "transformers" while setting
    quant_policy "none" with a plain HF repo id -- i.e. it named a transformers
    writer and then refused the transformers lane. Nothing else in the repo
    cross-checks these two fields against each other."""
    llm = _profile(name)["llm"]
    allow = list(llm.get("lane_allowlist") or [])
    assert allow, "an empty allowlist refuses every writer"
    quant = llm.get("quant_policy")
    model = str(llm.get("creative_model") or "")
    if quant == "none" and "/" in model and not model.lower().endswith("gguf"):
        assert "transformers" in allow, (
            "%s runs %s at quant_policy=none, which is the transformers lane, "
            "but the allowlist is %r -- the profile would refuse its own "
            "writer" % (name, model, allow))


@pytest.mark.parametrize("name", MAC_PROFILES)
def test_it_does_not_demand_a_key_it_has_no_use_for(name):
    """otr_mac_mps required OTR_GOOGLE_API_KEY while every image role was a
    LOCAL engine. A required key that buys nothing is a barrier with no
    payload."""
    p = _profile(name)
    required = list((p.get("preflight") or {}).get("required_keys") or [])
    if not required:
        return
    cloudish = {e for e in p["role_overrides"].values()
                if e.startswith(("cloud_", "google_", "ideo"))}
    assert cloudish, (
        "%s requires %r but every role is a local engine (%r) -- nothing will "
        "consume that key" % (name, required, sorted(set(
            p["role_overrides"].values()))))


@pytest.mark.parametrize("name", MAC_PROFILES)
def test_a_profile_naming_a_weighted_engine_declares_something_to_fetch(name,
                                                                        decls):
    """preflight.required_models is how a reader learns what to download.

    THIS DELIBERATELY DOES NOT MATCH THE TWO LISTS ITEM BY ITEM, and the reason
    is a trap worth stating. `model_requirements` holds S5 WIZARD ASSET IDS, not
    filenames -- `sd15` declares "sd15-v1-5-pruned-emaonly-fp16" while its
    loader opens "v1-5-pruned-emaonly-fp16.safetensors", and `wan_ti2v` declares
    "wan2.2-ti2v-5b" for a file called "Wan2.2-TI2V-5B-Q5_K_M.gguf". The first
    version of this test compared stems and failed on both Mac profiles, which
    is the same mistake that made the unified-memory weight guard inert earlier
    the same day (PBUG-20260908-02's commit). Two names for one artifact, and
    only one of them is a path.

    So the assertion is the one that holds without a mapping table: a profile
    that selects any engine carrying weights must tell the reader to fetch
    something. Empty is the failure this catches -- otr_mac_mps shipped with
    required_models empty while naming z_image_turbo, a ~20 GB download."""
    p = _profile(name)
    named = list((p.get("preflight") or {}).get("required_models") or [])
    weighted = {role: eng for role, eng in p["role_overrides"].items()
                if (decls.get(eng) or {}).get("model_requirements")}
    if not weighted:
        return
    assert named, (
        "%s selects weighted engine(s) %r but preflight.required_models is "
        "empty -- a reader is told to fetch nothing and finds out mid-run"
        % (name, sorted(set(weighted.values()))))


def test_the_adiff_profile_differs_from_the_base_one_only_where_it_should():
    """otr_mac_adiff exists to change the VIDEO lane and the frame budget, and
    nothing else. If it drifts from otr_mac_mps on the substrate -- voices,
    music, writer, device policy -- the two stop being comparable and a result
    on one says nothing about the other."""
    base, adiff = _profile("otr_mac_mps"), _profile("otr_mac_adiff")
    assert base["slot_overrides"]["voice_bank"] == \
        adiff["slot_overrides"]["voice_bank"]
    assert base["slot_overrides"]["music_engine"] == \
        adiff["slot_overrides"]["music_engine"]
    assert base["llm"]["creative_model"] == adiff["llm"]["creative_model"]
    assert base["llm"]["quant_policy"] == adiff["llm"]["quant_policy"]
    assert base["audio"] == adiff["audio"]
    assert base["video"]["dtype_policy"] == adiff["video"]["dtype_policy"]

    for role in ("announcer_visual", "music_visual", "character_visual"):
        assert adiff["role_overrides"][role] == "animatediff15_v3_haunted_video"
        assert base["role_overrides"][role] != adiff["role_overrides"][role]


def test_the_adiff_render_cap_is_the_reason_the_profile_exists():
    """MEASURED 2026-09-08. On the bare canonical, ghost_signal planned 125
    latents against AnimateDiff-Evolved's 16-frame context window -- about eight
    sliding windows per sampler step, 124 s/step, ~41 minutes for one clip. Per
    window the Mac is only ~1.5x slower than the 4060 that renders this lane in
    3-3.6 minutes; the whole gap is clip length.

    THE KNOB IS `video.max_render_frames`, NOT `render.frame_budget`, and the
    first version of this profile set the wrong one. It shipped with
    frame_budget 17 and the very next run still planned 125 latents -- caught by
    watching the log rather than by any test. capability_profiles.py:148 states
    the distinction outright: frame_budget is "the soak/single harness per-clip
    frame count (every 16GB tier declares 25 there and must NOT be capped to
    it)", while the planner reads video.max_render_frames through
    otr_shot_lock._stamp_coverage_plan. Two plausible names, one of which does
    nothing here."""
    p = _profile("otr_mac_adiff")
    cap = (p.get("video") or {}).get("max_render_frames")
    assert cap, (
        "otr_mac_adiff has no video.max_render_frames -- render.frame_budget "
        "does NOT cap the coverage planner, so the lane will plan long clips "
        "and sample them in many sliding windows")
    assert 0 < cap <= 33, (
        "max_render_frames %r puts this back into multi-window sampling" % cap)


def test_the_adiff_profile_stays_draft_until_a_clip_lands():
    """`animatediff15_v3_haunted_video` declares ["cuda"]. That row was reverted
    from ["cuda","mps"] on 2026-09-08 because nothing had proven otherwise, and
    the registry comment says to add "mps" only after a clip lands in otr/obs/.

    So this profile cannot be `shipping` while the row it depends on says the
    lane does not run here. When the receipt exists, BOTH move together -- and
    this test is what makes that a deliberate pair rather than a half-edit."""
    row = vreg.CAPABILITIES["animatediff15_v3_haunted_video"]
    claims_mps = "mps" in list(row["device_backends"])
    status = _profile("otr_mac_adiff")["status"]
    if not claims_mps:
        assert status == "draft", (
            "otr_mac_adiff is %r while its own video engine's row still says "
            "cuda-only. Promote the row on a receipt first." % status)


@pytest.mark.parametrize("name", MAC_PROFILES)
def test_availability_is_computed_and_its_verdict_is_recorded(name, decls):
    """Not an assertion that everything is OK -- an assertion that we KNOW.

    `availability()` is the only consumer of device_backends outside the
    registries, and it is not a render gate. A role it reports as requires_cuda
    is allowed here ONLY while the profile is a draft; a shipping profile whose
    own roles are inadmissible is incoherent."""
    p = _profile(name)
    verdicts = cp.availability(p, decls)
    bad = {role: (eng, verdicts.get(eng))
           for role, eng in p["role_overrides"].items()
           if verdicts.get(eng) != cp.REASON_OK}
    if p["status"] == "shipping":
        assert not bad, (
            "%s is shipping but these roles are inadmissible: %r" % (name, bad))
        return

    # A DRAFT PROFILE STILL HAS TO KNOW WHAT IT IS CARRYING. An earlier version
    # of this test asserted nothing at all on the draft branch, which a Sonnet
    # mutation audit correctly called a no-op: otr_mac_adiff really does have
    # three requires_cuda roles and the test sailed past them. "Draft" licenses
    # a KNOWN inadmissible engine, not an unexamined one.
    unexpected = {role: v for role, v in bad.items()
                  if v[0] not in DRAFT_KNOWN_INADMISSIBLE}
    assert not unexpected, (
        "%s is a draft carrying inadmissible role(s) nobody has accounted for: "
        "%r. Either the engine earned its backend row, or this dict should say "
        "why it has not." % (name, unexpected))
