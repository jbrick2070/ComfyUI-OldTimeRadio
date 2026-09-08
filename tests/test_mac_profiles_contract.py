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

MAC_PROFILES = ("otr_mac_mps",)

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


def test_the_ghost_lane_stays_out_of_the_planner_cap_list():
    """Guards the other half: if someone adds this lane to PLANNING_CAP_ENGINES
    to make the cap 'work', the profile's cap becomes live and the beat starts
    jump-cutting. Both halves have to stay true together."""
    from nodes._otr_video_engines import frame_contract as fc
    assert "animatediff15_v3_haunted_video" not in fc.PLANNING_CAP_ENGINES, (
        "the ghost lane declares max_frames=0 and continuity=NONE; capping it "
        "produces ~15 jump-cut segments and ~240 latents where one continuous "
        "beat sampled 125")


@pytest.mark.parametrize("name", MAC_PROFILES)
def test_availability_is_computed_and_every_role_is_admissible(name, decls):
    """`availability()` is the only consumer of device_backends outside the
    registries. `otr_mac_mps` is `shipping`, so every role it names must be
    admissible -- a shipping profile whose own roles are inadmissible is
    incoherent.

    This used to carry a `draft` branch for `otr_mac_adiff`, which a Sonnet
    mutation audit caught asserting nothing at all. That profile is deleted --
    it encoded three dropdown settings and a preflight list, and the operator's
    point stands that the dropdowns ARE the path -- so the branch is gone with
    it rather than left as dead code waiting to hide the next no-op."""
    p = _profile(name)
    verdicts = cp.availability(p, decls)
    bad = {role: (eng, verdicts.get(eng))
           for role, eng in p["role_overrides"].items()
           if verdicts.get(eng) != cp.REASON_OK}
    assert not bad, (
        "%s (%s) names inadmissible role(s): %r"
        % (name, p["status"], bad))
