"""S0a characterization fixture -- lock the three still-plan outputs at HEAD.

Chunk S0a of the per-model still-plans build
(``docs/2026-07-25-still-plans-locked-build-spec.md``, locked at
``84328aa1``). No production code moves in S0a -- this file is the
CURRENT-HEAD ground truth that every later chunk is diffed against.

The parity contract, verbatim from the spec (section 10):

  Three outputs per engine, because they disagree today:

  1. **Authored** -- ``derive_image_prompts`` objects + ``required_scene_targets``.
  2. **Materialized** -- ``still_consumer_capabilities``, ``roles_requiring_stills``,
     ``mesh_fodder_roles_from_video_policy``, and the effective engine per role.
  3. **Render-validated** -- what the render path validates or RAISES on:
     ``_still_spine_requires_scene``, the ``requires_mesh_fodder`` branch, the
     ``family == 'audio_driven_face'`` portrait branch, the LTX-I2V gate, and
     the IA2V portrait gate.

  Configuration matrix, not just the default env -- the change is a ROUTING
  change: hosts off/on; ``*=viz_*``; forced ``ltx_audio_in``; cloud avatar
  excluded from the local redirect; LTX-I2V off/on; IA2V vs a non-talking
  recipe; malformed force map; incomplete policy.

The fixture (``tests/fixtures/still_plan_head_parity.json``) is regenerated
by running this file as a script::

    .venv\\Scripts\\python.exe tests\\test_still_plan_parity.py --regenerate

Pure CPU: no ComfyUI runtime, no GPU, no image model, no LLM.
"""

from __future__ import annotations

import json
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_FIXTURE_PATH = _HERE / "fixtures" / "still_plan_head_parity.json"


# ---------------------------------------------------------------------------
# Bootstrap: allow ``python tests/test_still_plan_parity.py --regenerate``
# to work outside a pytest session. Mirrors the tests/conftest.py stubs so
# node imports resolve identically in both contexts.
# ---------------------------------------------------------------------------
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OTR_TEST_MODE", "1")
if "folder_paths" not in sys.modules:
    _fp_stub = types.ModuleType("folder_paths")
    _fp_stub.get_output_directory = lambda: str(Path.cwd())  # noqa: E731
    _fp_stub.get_filename_list = lambda *a, **kw: []  # noqa: E731
    sys.modules["folder_paths"] = _fp_stub


from nodes import otr_image_director as imgdir  # noqa: E402
from nodes import otr_image_gen_dispatcher as disp  # noqa: E402
from nodes import otr_meta_brief_image_prompt as mbp  # noqa: E402
from nodes._otr_shared import role_slots as _role_slots  # noqa: E402
from nodes._otr_video_engines import registry as _vreg  # noqa: E402
import nodes._otr_video_engines as _video_pkg  # noqa: F401,E402
from nodes._otr_video_engines import render_driver as _rd  # noqa: E402
from nodes.otr_video_director import OTRVideoDirector as _VD  # noqa: E402


# ---------------------------------------------------------------------------
# The characterization corpus -- one representative episode covering the
# three still-plan-relevant shapes (opening beat, character beat, closing
# music beat) with one named cast row to exercise the portrait branch.
# ---------------------------------------------------------------------------
_LINES = (
    {"line_id": "b001", "speaker_role": "announcer", "char_id": "announcer",
     "start_s": 8.4, "dur_s": 4.0},
    {"line_id": "b002", "speaker_role": "char_voice", "char_id": "c01",
     "start_s": 12.4, "dur_s": 6.0},
    {"line_id": "b005", "speaker_role": "music_close", "char_id": "music_close",
     "start_s": 18.4, "dur_s": 5.0},
)
_CAST = ({"char_id": "c01", "name": "BABA",
          "portrait_prompt": "a tall weathered spacer with a scar"},)
_META = {"episode_id": "ep_still_plan_parity",
         "story_brief_terms": {"setting": ["a derelict orbital station"]}}


#: Every env key any configuration touches. Overridden together so a prior
#: configuration cannot leak into the next.
_ENV_KEYS = (
    "OTR_FORCE_ENGINE_MAP",
    "OTR_ENABLE_HUMO_HOSTS",
    "OTR_ENABLE_LTX_I2V",
    "OTR_LTX_AV_RECIPE",
    "OTR_LTX_AV_UNET",
)


#: The configuration matrix. Each entry is (label, env-overrides). ``env``
#: overrides are applied on top of a clean baseline (every ``_ENV_KEYS`` key
#: cleared). Order is stable so the fixture is deterministic.
_CONFIGURATIONS = (
    ("default", {}),
    ("hosts_on", {"OTR_ENABLE_HUMO_HOSTS": "1"}),
    ("force_all_viz_camera", {"OTR_FORCE_ENGINE_MAP": "*=viz_camera"}),
    ("force_all_ltx_audio_in", {"OTR_FORCE_ENGINE_MAP": "*=ltx_audio_in"}),
    ("force_bookends_cloud_kling",
     {"OTR_FORCE_ENGINE_MAP":
      "announcer_visual=cloud_kling_avatar,music_visual=cloud_kling_avatar"}),
    ("ltx_i2v_off", {"OTR_ENABLE_LTX_I2V": "0"}),
    ("ia2v_non_talking", {"OTR_LTX_AV_RECIPE": "distilled_native"}),
    ("malformed_force_map", {"OTR_FORCE_ENGINE_MAP": "not-a-role-pair"}),
)


#: The three video roles ShotLock will emit for our corpus, and the beat each
#: belongs to. Used to key render-time gate decisions per (engine, role).
_ROLE_BEAT_PAIRS = (
    ("announcer_visual", "b001"),
    ("character_video", "b002"),
    ("music_visual", "b005"),
)


# ---------------------------------------------------------------------------
# Helpers -- pure over inputs, and all env reads happen inside the callee at
# call time (dispatcher force-map, radio-host redirect, LTX-I2V gate, IA2V
# recipe probe). No module-scope caches to invalidate.
# ---------------------------------------------------------------------------
@contextmanager
def _env_override(env_map):
    """Set exactly the ``_ENV_KEYS`` env; restore prior values on exit."""
    prior = {k: os.environ.get(k) for k in _ENV_KEYS}
    try:
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        for k, v in env_map.items():
            os.environ[k] = v
        yield
    finally:
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
            if prior[k] is not None:
                os.environ[k] = prior[k]


def _policy_for(engine_id):
    """The v2 video policy an operator's OTR_VideoDirector would emit for a
    single-engine episode (every video slot points at ``engine_id``)."""
    video_models = {slot: {"engine_id": engine_id, "custom": False}
                    for slot in _role_slots.ROLE_TO_VIDEO_SLOT.values()}
    return {"policy_version": 2, "video_models": video_models}


def _authored_row(engine_id, policy):
    """The producer's decisions: objects it authors + the required-target
    receipt it publishes. This is what the image dispatcher materializes."""
    vm = policy["video_models"]
    aspects = _VD._role_aspects(vm)
    talking = _VD._role_talking(vm)
    mesh_roles = imgdir.mesh_fodder_roles_from_video_policy(policy)
    row = {"aspects": dict(sorted(aspects.items())),
           "talking": dict(sorted(talking.items()))}
    try:
        payload, warnings = mbp.derive_image_prompts(
            list(_CAST), dict(_META), llm_fn=None, lines=list(_LINES),
            still_aspects=aspects,
            mesh_fodder_roles=mesh_roles,
            talking_roles=talking, still_word_roles=None,
            video_models=vm,
        )
    except Exception as exc:  # noqa: BLE001 -- record, never mask
        row["error"] = "%s: %s" % (type(exc).__name__, exc)
        return row
    row["required_target_ids"] = sorted(
        str(r.get("object_id") or "")
        for r in (payload.get("required_scene_targets") or []))
    row["objects"] = sorted(
        "%s|%s|%sx%s" % (o.get("object_id"), o.get("kind"),
                         o.get("w"), o.get("h"))
        for o in (payload.get("objects") or []))
    row["warnings"] = sorted(str(w) for w in (warnings or []))
    return row


def _materialized_row(engine_id, policy):
    """The dispatcher / image-director read of the policy: what the ledger
    will carry through to render dispatch."""
    caps = disp.still_consumer_capabilities(policy)
    stills = disp.roles_requiring_stills(policy)
    mesh = imgdir.mesh_fodder_roles_from_video_policy(policy)
    vm = policy["video_models"]
    effective = {
        role: disp._effective_video_engine_for_role(
            role, _role_slots.engine_id_for_role(vm, role))
        for role in _role_slots.ROLE_TO_VIDEO_SLOT
    }
    return {
        "capabilities": (dict(sorted(caps.items()))
                         if isinstance(caps, dict) else None),
        "roles_requiring_stills": (sorted(stills) if stills is not None
                                   else None),
        "mesh_fodder_roles": list(mesh),
        "effective_engine": dict(sorted(effective.items())),
    }



def _render_decisions_row(engine_id, policy):
    """The render path's own gates, evaluated per (role, beat). The five
    mechanisms the parity contract cites: ``_still_spine_requires_scene``,
    ``requires_mesh_fodder``, ``family == 'audio_driven_face'``, the
    LTX-I2V gate (``render_driver.py:1801-1817``), and the IA2V portrait
    gate (``render_driver.py:1709-1721``). Plus the wide LTX radio-face
    gate (``render_driver.py:1761``) that shares the IA2V register."""
    vm = policy["video_models"]
    out = {}
    for role, beat_id in _ROLE_BEAT_PAIRS:
        picked = _role_slots.engine_id_for_role(vm, role)
        eff = disp._effective_video_engine_for_role(role, picked)
        family = _rd.engine_family(eff, "")
        try:
            engine = _vreg.get_engine(eff) if _vreg.is_registered(eff) else None
        except Exception:  # noqa: BLE001
            engine = None
        requires_mesh_fodder = bool(engine and getattr(
            engine, "requires_mesh_fodder", False))
        shot = {"engine_id": eff, "family": family, "role": role,
                "beat_id": beat_id}
        requires_scene = bool(_rd._still_spine_requires_scene(
            shot, eff, family))
        ia2v_active = bool(_rd._ia2v_talking_register_active(eff))
        ltx_i2v_gate = (
            eff == "ltx_video"
            and os.environ.get("OTR_ENABLE_LTX_I2V", "1") == "1")
        ia2v_portrait_gate = (
            eff == "ltx_audio_in"
            and role == "character_video"
            and ia2v_active)
        ltx_radio_face_gate = (
            eff == "ltx_audio_in"
            and role in ("announcer_visual", "music_visual")
            and ia2v_active)
        out[role] = {
            "effective_engine": eff,
            "family": family or "",
            "requires_scene": requires_scene,
            "requires_mesh_fodder": requires_mesh_fodder,
            "family_is_audio_driven_face": (family == "audio_driven_face"),
            "ltx_i2v_gate": bool(ltx_i2v_gate),
            "ia2v_portrait_gate": bool(ia2v_portrait_gate),
            "ltx_radio_face_gate": bool(ltx_radio_face_gate),
        }
    return dict(sorted(out.items()))


def _engine_row(engine_id, cfg_env):
    """One (engine, configuration) cell of the parity matrix."""
    with _env_override(cfg_env):
        policy = _policy_for(engine_id)
        return {
            "authored": _authored_row(engine_id, policy),
            "materialized": _materialized_row(engine_id, policy),
            "render_decisions": _render_decisions_row(engine_id, policy),
        }


def _special_cases():
    """Named observations of the dispatcher against MALFORMED / INCOMPLETE
    policy inputs. These do not iterate over engines because the fail-closed
    boundary is the policy structure, not the picked engine.

    ``incomplete_policy_no_video_models`` proves the ``None`` third state
    that the dispatcher preserves for unknown coverage; ``policy_version_v1``
    proves that ``dispatch_images`` and the still helpers still key off the
    policy_version integer and reject a stale v1 policy (this is the
    boundary S0b flips to require v3).
    """
    with _env_override({}):
        no_vm = {"policy_version": 2, "video_models": None}
        v1 = {"policy_version": 1, "video_models": {}}
        empty_vm = {"policy_version": 2, "video_models": {}}
        empty_slot = {"policy_version": 2, "video_models": {
            "announcer_video_model": {"engine_id": "", "custom": False},
            "music_video_model": {"engine_id": "", "custom": False},
            "character_video_model": {"engine_id": "", "custom": False},
        }}
        return {
            "incomplete_policy_no_video_models": {
                "capabilities": disp.still_consumer_capabilities(no_vm),
                "roles_requiring_stills": (
                    sorted(disp.roles_requiring_stills(no_vm))
                    if disp.roles_requiring_stills(no_vm) is not None
                    else None),
                "mesh_fodder_roles": list(
                    imgdir.mesh_fodder_roles_from_video_policy(no_vm)),
            },
            "policy_version_v1_empty_models": {
                # The current dispatcher inspects video_models regardless of
                # policy_version; capture what it returns so S0b's flip to
                # v3-only rejection is a named delta.
                "capabilities": disp.still_consumer_capabilities(v1),
                "roles_requiring_stills": (
                    sorted(disp.roles_requiring_stills(v1))
                    if disp.roles_requiring_stills(v1) is not None
                    else None),
                "mesh_fodder_roles": list(
                    imgdir.mesh_fodder_roles_from_video_policy(v1)),
            },
            "empty_video_models_dict": {
                "capabilities": disp.still_consumer_capabilities(empty_vm),
                "roles_requiring_stills": (
                    sorted(disp.roles_requiring_stills(empty_vm))
                    if disp.roles_requiring_stills(empty_vm) is not None
                    else None),
                "mesh_fodder_roles": list(
                    imgdir.mesh_fodder_roles_from_video_policy(empty_vm)),
            },
            "empty_engine_id_per_slot": {
                "capabilities": disp.still_consumer_capabilities(empty_slot),
                "roles_requiring_stills": (
                    sorted(disp.roles_requiring_stills(empty_slot))
                    if disp.roles_requiring_stills(empty_slot) is not None
                    else None),
                "mesh_fodder_roles": list(
                    imgdir.mesh_fodder_roles_from_video_policy(empty_slot)),
            },
        }



def _all_registered_video_engines():
    """Every internal engine id the video registry currently owns, sorted.
    Matches the spec's 31-engine count until an operator decision changes
    the registry."""
    return sorted(_vreg._VIDEO_REGISTRY._registry)


def _required_scene_targets_receipt(engine_id):
    """Materialize the exact ``required_scene_targets`` list the image phase
    publishes for a single-engine episode. The spec REQUIRES the fixture to
    carry a real receipt so it never inherits the ``OTR_TEST_MODE`` validator
    skip at ``otr_video_render_batch.py:311-321``; this returns the receipt
    a ledger would carry into ``validate_and_repair_still_spine``."""
    with _env_override({}):
        vm = _policy_for(engine_id)["video_models"]
        aspects = _VD._role_aspects(vm)
        talking = _VD._role_talking(vm)
        mesh_roles = imgdir.mesh_fodder_roles_from_video_policy(
            _policy_for(engine_id))
        payload, _warnings = mbp.derive_image_prompts(
            list(_CAST), dict(_META), llm_fn=None, lines=list(_LINES),
            still_aspects=aspects, mesh_fodder_roles=mesh_roles,
            talking_roles=talking, still_word_roles=None, video_models=vm,
        )
    return [
        {"object_id": str(r.get("object_id") or ""),
         "kind": str(r.get("kind") or ""),
         "beat_id": str(r.get("beat_id") or "")}
        for r in (payload.get("required_scene_targets") or [])
    ]


def _generate_matrix():
    """Return the whole parity fixture as a plain Python dict, ready to
    ``json.dump``. Deterministic: sorted keys everywhere, stable engine and
    configuration order."""
    engines = _all_registered_video_engines()
    matrix = {}
    for cfg_label, cfg_env in _CONFIGURATIONS:
        cfg_rows = {}
        for engine_id in engines:
            cfg_rows[engine_id] = _engine_row(engine_id, cfg_env)
        matrix[cfg_label] = cfg_rows
    receipts = {engine_id: _required_scene_targets_receipt(engine_id)
                for engine_id in engines}
    return {
        "version": 1,
        "spec_reference": "docs/2026-07-25-still-plans-locked-build-spec.md",
        "chunk": "S0a",
        "configurations": [
            {"label": label, "env": dict(env)}
            for label, env in _CONFIGURATIONS],
        "role_beat_pairs": [{"role": r, "beat_id": b}
                            for r, b in _ROLE_BEAT_PAIRS],
        "engines": engines,
        "matrix": matrix,
        "required_scene_targets_by_engine": receipts,
        "special_cases": _special_cases(),
    }


# ---------------------------------------------------------------------------
# Tests. The parity fixture is BOTH the artifact under review AND the truth
# for the migration -- so every produced row must diff to zero against it.
# ---------------------------------------------------------------------------
def _load_fixture():
    if not _FIXTURE_PATH.exists():
        raise pytest.skip.Exception(
            "still-plan parity fixture missing; regenerate with "
            "`python tests/test_still_plan_parity.py --regenerate`")
    with _FIXTURE_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_registry_matches_fixture_engine_list():
    """The fixture is keyed by engine id; a new / removed engine breaks
    parity immediately. The spec pins the count at 31; the assertion is
    on-parity, not on the constant, so an operator adding a 32nd engine
    tells us here first."""
    fixture = _load_fixture()
    assert _all_registered_video_engines() == fixture["engines"]


def test_configurations_match_fixture():
    """A configuration reorder / rename must be a deliberate fixture change
    -- otherwise the per-cell diff below silently loses coverage."""
    fixture = _load_fixture()
    got = [{"label": lbl, "env": dict(env)} for lbl, env in _CONFIGURATIONS]
    assert got == fixture["configurations"]
    assert [{"role": r, "beat_id": b} for r, b in _ROLE_BEAT_PAIRS] \
        == fixture["role_beat_pairs"]


def test_still_plan_parity_matches_head():
    """The three outputs -- authored, materialized, render-validated --
    across the whole configuration matrix, engine by engine. Any drift here
    is either an intentional spec chunk landing (record the delta explicitly
    per section 10's transition rule) or a regression."""
    fixture = _load_fixture()
    got = _generate_matrix()
    # Iterate for a legible diff message: pytest shows the first failing
    # (config, engine, section) triple instead of a page of nested dicts.
    for cfg_label, cfg_rows in got["matrix"].items():
        assert cfg_label in fixture["matrix"], cfg_label
        for engine_id, row in cfg_rows.items():
            fixture_row = fixture["matrix"][cfg_label].get(engine_id)
            assert fixture_row is not None, (cfg_label, engine_id)
            for section in ("authored", "materialized", "render_decisions"):
                assert row[section] == fixture_row[section], (
                    cfg_label, engine_id, section)
    # Whole-object equality catches any missing configuration or engine in
    # the generated side.
    assert got["matrix"] == fixture["matrix"]


def test_required_scene_targets_receipts_match_head():
    """The receipt that carries into ``validate_and_repair_still_spine``.
    The spec: 'the fixture must carry a real required_scene_targets receipt
    so it never inherits the OTR_TEST_MODE validator skip.'"""
    fixture = _load_fixture()
    got = {engine_id: _required_scene_targets_receipt(engine_id)
           for engine_id in _all_registered_video_engines()}
    assert got == fixture["required_scene_targets_by_engine"]


def test_special_cases_match_head():
    """Named observations against malformed / stale policy inputs. S0b's
    routing freeze changes some of these on purpose (v1 rejection tightens
    to v3); this test names the current answer so that flip is a documented
    delta rather than a silent behaviour change."""
    fixture = _load_fixture()
    assert _special_cases() == fixture["special_cases"]


# ---------------------------------------------------------------------------
# Regeneration entry point. Run as::
#
#   .venv\Scripts\python.exe tests\test_still_plan_parity.py --regenerate
#
# The write is deterministic (sort_keys=True, UTF-8 no BOM, LF line
# endings) so a re-run at unchanged HEAD produces a byte-identical file.
# ---------------------------------------------------------------------------
def _write_fixture(matrix):
    _FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(matrix, indent=1, sort_keys=True,
                         ensure_ascii=False) + "\n"
    # Explicit UTF-8 no BOM, LF line endings -- the standing project rule.
    with _FIXTURE_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(payload)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--regenerate":
        matrix = _generate_matrix()
        _write_fixture(matrix)
        print("wrote %s (%d engines x %d configurations)"
              % (_FIXTURE_PATH, len(matrix["engines"]),
                 len(matrix["configurations"])))
    else:
        raise SystemExit(
            "usage: python tests/test_still_plan_parity.py --regenerate")
